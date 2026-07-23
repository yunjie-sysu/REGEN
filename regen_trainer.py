# REGEN Trainer (Independent framework — no longer depends on DPOTrainer)
#
# Loss function (only the positive sample term is retained):
#   L = coef * pol
#   coef = - advantage * rho
#   rho  = ρ(w, b) for negative samples, 1 for positive samples
#   w    = clip(π/μ, 0, regen_alpha)
#
# Data requirements (from trajectory datasets):
#   prompt / response / nll_seq / advantage / origin_label

from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, Tuple, Union

import torch
import torch.nn as nn
from transformers import Trainer, PreTrainedModel


@dataclass
class REGENConfig:
    """Configuration for REGEN Trainer.

    Hyper-parameters:
        regen_alpha            : float, upper cap for IS weight before tapering (default 1.0).
        regen_clip_ratio_max   : float, hard upper clip on raw IS ratio (default 10.0).
        regen_length_normalize : bool,  divide log-probs by sequence length (default False).
        regen_require_old_nll  : bool,  if True (default), missing nll → raise.
        regen_use_dataset_advantage : bool, if True (default), use dataset advantage column.
        regen_default_advantage : float, fallback advantage (default 1.0).
    """
    regen_alpha: float = field(default=1.0)
    regen_clip_ratio_max: float = field(default=10.0)
    regen_length_normalize: bool = field(default=False)
    regen_require_old_nll: bool = field(default=True)
    regen_use_dataset_advantage: bool = field(default=True)
    regen_default_advantage: float = field(default=1.0)

    # Tokenization
    max_length: int = field(default=1024)
    max_prompt_length: int = field(default=512)
    truncation_mode: str = field(default="keep_end")
    label_pad_token_id: int = field(default=-100)
    padding_value: int = field(default=0)


class REGENTrainer(Trainer):
    """REGEN trainer: loss retains only the positive sample term ``coef * pol``,
    and adjusts the importance weight based on the sample's origin (``origin_label``)
    in the original data.
    """

    # ---- Column name constants ----
    NLL_KEY = "nll_seq"
    ADVANTAGE_KEY = "advantage"
    ORIGIN_LABEL_KEY = "origin_label"
    ORIGIN_IS_NEG_KEY = "origin_is_negative"
    OLD_LOGPS_KEY = "old_logps"

    def __init__(self, *args, regen_config: REGENConfig = None, **kwargs):
        if regen_config is None:
            regen_config = REGENConfig()
        self.regen_config = regen_config

        # Extract regen-specific params from kwargs (for backward compatibility)
        self.regen_alpha = kwargs.pop("regen_alpha", regen_config.regen_alpha)
        self.regen_clip_ratio_max = kwargs.pop("regen_clip_ratio_max", regen_config.regen_clip_ratio_max)
        self.regen_length_normalize = kwargs.pop("regen_length_normalize", regen_config.regen_length_normalize)
        self.regen_require_old_nll = kwargs.pop("regen_require_old_nll", regen_config.regen_require_old_nll)
        self.regen_use_dataset_advantage = kwargs.pop("regen_use_dataset_advantage", regen_config.regen_use_dataset_advantage)
        self.regen_default_advantage = kwargs.pop("regen_default_advantage", regen_config.regen_default_advantage)

        super().__init__(*args, **kwargs)

        # 1) Inject old_logps column into train/eval datasets (from nll_seq)
        self._inject_old_logps_columns()
        # 2) Set up custom data collator
        self._setup_data_collator()

    def _remove_unused_columns(self, dataset, description=""):
        """Override to preserve REGEN-specific columns that are not model inputs."""
        # Don't remove any columns — our collator handles popping non-model fields
        return dataset

    # ------------------------------------------------------------------ #
    # Step 1: dataset.map → inject old_logps column
    # ------------------------------------------------------------------ #
    def _inject_old_logps_columns(self):
        from datasets import Dataset

        nll_key = self.NLL_KEY
        out_key = self.OLD_LOGPS_KEY
        require = self.regen_require_old_nll

        def _add(ds):
            if ds is None:
                return ds
            if not isinstance(ds, Dataset):
                return ds
            cols = ds.column_names
            if nll_key not in cols:
                if require:
                    raise ValueError(
                        f"[REGENTrainer] dataset is missing required column "
                        f"`{nll_key}`. Got columns: {cols}"
                    )
                return ds

            def _map_fn(ex):
                cv = ex[nll_key]
                if cv is None:
                    if require:
                        raise ValueError(
                            f"[REGENTrainer] found None in `{nll_key}`."
                        )
                    ex[out_key] = None
                else:
                    ex[out_key] = -float(cv)  # log μ = -nll
                return ex

            return ds.map(_map_fn, desc="REGEN: nll → old_logps")

        if getattr(self, "train_dataset", None) is not None:
            self.train_dataset = _add(self.train_dataset)
        if getattr(self, "eval_dataset", None) is not None:
            if isinstance(self.eval_dataset, dict):
                self.eval_dataset = {k: _add(v) for k, v in self.eval_dataset.items()}
            else:
                self.eval_dataset = _add(self.eval_dataset)

    # ------------------------------------------------------------------ #
    # Step 2: Set up custom data collator
    # ------------------------------------------------------------------ #
    def _setup_data_collator(self):
        from transformers import DataCollatorForSeq2Seq

        nll_key = self.NLL_KEY
        out_key = self.OLD_LOGPS_KEY
        adv_key = self.ADVANTAGE_KEY
        label_key = self.ORIGIN_LABEL_KEY
        out_neg = self.ORIGIN_IS_NEG_KEY
        default_adv = float(self.regen_default_advantage)

        # Use a simple collator that pads input_ids and labels
        tokenizer = self.tokenizer
        pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        label_pad_token_id = self.regen_config.label_pad_token_id

        def collator(features):
            # Extract and pop non-tokenized fields
            old_vals = []
            adv_vals = []
            is_neg_vals = []

            for f in features:
                # old policy logps
                if out_key in f and f[out_key] is not None:
                    old_vals.append(float(f.pop(out_key)))
                elif nll_key in f and f[nll_key] is not None:
                    old_vals.append(-float(f.pop(nll_key)))
                else:
                    old_vals.append(0.0)

                # advantages
                cv = f.pop(adv_key, None)
                try:
                    adv_vals.append(float(cv) if cv is not None else default_adv)
                except (TypeError, ValueError):
                    adv_vals.append(default_adv)

                # origin_label → is_negative
                lab = f.pop(label_key, None)
                if isinstance(lab, str):
                    is_neg_vals.append(1.0 if lab.strip().lower() == "negative" else 0.0)
                elif lab is None:
                    is_neg_vals.append(0.0)
                else:
                    try:
                        is_neg_vals.append(float(lab))
                    except (TypeError, ValueError):
                        is_neg_vals.append(0.0)

                # Pop nll column if still present
                f.pop(nll_key, None)

            # Now features should contain: input_ids, attention_mask, labels
            # Pad sequences
            batch = self._pad_features(features, pad_token_id, label_pad_token_id)

            # Add extra tensors
            batch[out_key] = torch.tensor(old_vals, dtype=torch.float32)
            batch[adv_key] = torch.tensor(adv_vals, dtype=torch.float32)
            batch[out_neg] = torch.tensor(is_neg_vals, dtype=torch.float32)
            return batch

        self.data_collator = collator

    def _pad_features(self, features, pad_token_id, label_pad_token_id):
        """Pad a list of feature dicts (each with input_ids, attention_mask, labels) into a batch."""
        # Get max lengths
        max_input_len = max(len(f["input_ids"]) for f in features)
        max_label_len = max(len(f["labels"]) for f in features)
        max_len = max(max_input_len, max_label_len)

        input_ids_list = []
        attention_mask_list = []
        labels_list = []

        for f in features:
            input_ids = f["input_ids"]
            attention_mask = f["attention_mask"]
            labels = f["labels"]

            # Pad on the right
            pad_len = max_len - len(input_ids)
            input_ids = input_ids + [pad_token_id] * pad_len
            attention_mask = attention_mask + [0] * pad_len
            labels = labels + [label_pad_token_id] * pad_len

            input_ids_list.append(input_ids)
            attention_mask_list.append(attention_mask)
            labels_list.append(labels)

        return {
            "input_ids": torch.tensor(input_ids_list, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask_list, dtype=torch.long),
            "labels": torch.tensor(labels_list, dtype=torch.long),
        }

    # ------------------------------------------------------------------ #
    # Tokenization: tokenize prompt + response into input_ids, attention_mask, labels
    # ------------------------------------------------------------------ #
    def tokenize_row(self, prompt: str, response: str) -> Dict:
        """Tokenize a single prompt-response pair into model inputs."""
        max_length = self.regen_config.max_length
        max_prompt_length = self.regen_config.max_prompt_length
        truncation_mode = self.regen_config.truncation_mode
        label_pad_token_id = self.regen_config.label_pad_token_id

        # Tokenize prompt (without special tokens)
        prompt_tokens = self.tokenizer(prompt, add_special_tokens=False)
        prompt_input_ids = prompt_tokens["input_ids"]
        prompt_attention_mask = prompt_tokens["attention_mask"]

        # Tokenize response (without special tokens)
        response_tokens = self.tokenizer(response, add_special_tokens=False)
        response_input_ids = response_tokens["input_ids"]
        response_attention_mask = response_tokens["attention_mask"]

        # Add BOS token if available
        if self.tokenizer.bos_token_id is not None:
            prompt_input_ids = [self.tokenizer.bos_token_id] + prompt_input_ids
            prompt_attention_mask = [1] + prompt_attention_mask

        # Truncate prompt if needed
        if len(prompt_input_ids) > max_prompt_length:
            if truncation_mode == "keep_start":
                prompt_input_ids = prompt_input_ids[:max_prompt_length]
                prompt_attention_mask = prompt_attention_mask[:max_prompt_length]
            elif truncation_mode == "keep_end":
                prompt_input_ids = prompt_input_ids[-max_prompt_length:]
                prompt_attention_mask = prompt_attention_mask[-max_prompt_length:]
            else:
                raise ValueError(f"Unknown truncation mode: {truncation_mode}")

        # Concatenate prompt + response
        input_ids = prompt_input_ids + response_input_ids
        attention_mask = prompt_attention_mask + response_attention_mask

        # Truncate total length if exceeds max_length
        if len(input_ids) > max_length:
            input_ids = input_ids[:max_length]
            attention_mask = attention_mask[:max_length]

        # Create labels: mask prompt with label_pad_token_id, keep response
        labels = [label_pad_token_id] * len(prompt_input_ids) + response_input_ids
        if len(labels) > max_length:
            labels = labels[:max_length]

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    # ------------------------------------------------------------------ #
    # Forward pass: compute policy logps and completion lengths
    # ------------------------------------------------------------------ #
    def _compute_logps(self, model, batch):
        """Forward pass that returns per-sample policy logps and completion lengths."""
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        label_pad_token_id = self.regen_config.label_pad_token_id

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
        )
        all_logits = outputs.logits

        # Align logits with labels
        if all_logits.shape[:2] != labels.shape[:2]:
            seq_len = labels.shape[1]
            all_logits = all_logits[:, -seq_len:]

        # Compute logps for completion tokens (where labels != label_pad_token_id)
        logps = []
        completion_lens = []

        for i in range(labels.shape[0]):
            # Find completion tokens (where label != pad)
            completion_mask = labels[i] != label_pad_token_id
            completion_indices = completion_mask.nonzero(as_tuple=True)[0]

            if len(completion_indices) == 0:
                logps.append(torch.tensor(0.0, device=all_logits.device))
                completion_lens.append(torch.tensor(1.0, device=all_logits.device))
                continue

            # For each completion token at position t, we need logits at position t-1
            # (because logits[t] predicts token at position t+1)
            # So shift: logits at positions [start-1:end-1] predict tokens at [start:end]
            start = completion_indices[0].item()
            end = completion_indices[-1].item() + 1

            # Get logits for predicting completion tokens
            if start == 0:
                # Can't predict the first token, skip it
                pred_logits = all_logits[i, 0:end-1, :]
                target_ids = labels[i, 1:end]
            else:
                pred_logits = all_logits[i, start-1:end-1, :]
                target_ids = labels[i, start:end]

            # Compute log probabilities
            log_probs = torch.log_softmax(pred_logits, dim=-1)
            token_log_probs = log_probs.gather(1, target_ids.unsqueeze(-1)).squeeze(-1)

            logps.append(token_log_probs.sum())
            completion_lens.append(torch.tensor(float(len(target_ids)), device=all_logits.device))

        policy_logps = torch.stack(logps)
        completion_lens = torch.stack(completion_lens).clamp(min=1).float()

        return policy_logps, completion_lens

    # ------------------------------------------------------------------ #
    # REGEN loss: only the positive sample term
    # ------------------------------------------------------------------ #
    def regen_loss(
        self,
        policy_logps: torch.FloatTensor,
        old_logps: torch.FloatTensor,
        completion_len: torch.FloatTensor,
        advantage: torch.FloatTensor = None,
        origin_is_negative: torch.FloatTensor = None,
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, Dict[str, torch.Tensor]]:
        device = policy_logps.device
        old = old_logps.to(device=device, dtype=policy_logps.dtype)

        if self.regen_length_normalize:
            pol = policy_logps / completion_len
            old_used = old / completion_len
        else:
            pol = policy_logps
            old_used = old

        log_ratio = pol - old_used

        max_log = torch.log(torch.tensor(self.regen_clip_ratio_max, device=device))
        ratio = torch.exp(torch.clamp(log_ratio, max=max_log)).detach()

        w = torch.clamp(ratio, min=0, max=self.regen_alpha)

        # Importance weight: positive samples use 1, negative samples use clipped w
        if origin_is_negative is None:
            weight = torch.ones_like(w)
        else:
            oin = origin_is_negative.to(device=device, dtype=w.dtype)
            weight = torch.where(oin > 0.5, w, torch.ones_like(w))

        # ---- Real advantage as loss coefficient ----
        if self.regen_use_dataset_advantage and advantage is not None:
            r = advantage.to(device=device, dtype=pol.dtype)
        else:
            r = torch.full_like(pol, float(self.regen_default_advantage))

        coef = - r * weight

        losses = coef * pol

        rewards = ((policy_logps - old) / completion_len).detach()

        diagnostics = {
            "regen/ratio_mean": ratio.mean().detach(),
            "regen/w_mean": w.mean().detach(),
            "regen/weight_mean": weight.mean().detach(),
            "regen/loss": losses.mean().detach(),
            "regen/frac_capped": (ratio >= self.regen_alpha).float().mean().detach(),
            "regen/completion_len_mean": completion_len.mean().detach(),
            "regen/old_logp_mean": old.mean().detach(),
            "regen/length_normalize": torch.tensor(
                1.0 if self.regen_length_normalize else 0.0, device=device
            ),
            "regen/data_advantage_mean": r.mean().detach(),
        }

        if origin_is_negative is not None:
            diagnostics["regen/frac_origin_negative"] = (
                origin_is_negative.to(device=device) > 0.5
            ).float().mean().detach()
        return losses, rewards, diagnostics

    # ------------------------------------------------------------------ #
    # Override compute_loss to plug REGEN into the training loop
    # ------------------------------------------------------------------ #
    def compute_loss(
        self,
        model,
        inputs,
        return_outputs: bool = False,
        **kwargs,
    ):
        metrics = {}

        # Forward pass to get policy logps
        policy_logps, completion_len = self._compute_logps(model, inputs)

        device = policy_logps.device

        # --- Strict validation: must obtain old policy logps from data ---
        old = inputs.get(self.OLD_LOGPS_KEY, None)
        if old is None:
            raise RuntimeError(
                f"[REGENTrainer] batch is missing `{self.OLD_LOGPS_KEY}`. "
                f"The dataset must provide `{self.NLL_KEY}` column "
                f"so that μ can be reconstructed from log μ = -nll."
            )
        if not torch.is_tensor(old):
            old = torch.tensor(old, dtype=policy_logps.dtype, device=device)
        old_logps = old.to(device=device, dtype=policy_logps.dtype)

        # Dataset real advantage (injected by collator)
        ds_advantage = inputs.get(self.ADVANTAGE_KEY, None)
        if ds_advantage is not None and not torch.is_tensor(ds_advantage):
            ds_advantage = torch.tensor(
                ds_advantage, dtype=policy_logps.dtype, device=device
            )

        # origin_label → is_negative (injected by collator)
        origin_is_negative = inputs.get(self.ORIGIN_IS_NEG_KEY, None)
        if origin_is_negative is not None and not torch.is_tensor(origin_is_negative):
            origin_is_negative = torch.tensor(
                origin_is_negative, dtype=policy_logps.dtype, device=device
            )

        losses, rewards, diag = self.regen_loss(
            policy_logps,
            old_logps,
            completion_len,
            advantage=ds_advantage,
            origin_is_negative=origin_is_negative,
        )

        # Log metrics (convert tensors to scalars)
        prefix = "eval_" if not self.model.training else ""
        metrics[f"{prefix}rewards/chosen"] = rewards.mean().item()
        metrics[f"{prefix}logps/chosen"] = policy_logps.detach().mean().item()
        for k, v in diag.items():
            if torch.is_tensor(v):
                metrics[f"{prefix}{k}"] = v.item()
            else:
                metrics[f"{prefix}{k}"] = v

        # Store metrics for logging
        self._regen_metrics = metrics

        loss = losses.mean()
        return (loss, outputs) if return_outputs else loss

    # ------------------------------------------------------------------ #
    # Override logging to include REGEN metrics
    # ------------------------------------------------------------------ #
    def log(self, logs):
        # Merge REGEN metrics into logs, converting tensors to scalars
        if hasattr(self, "_regen_metrics"):
            for k, v in self._regen_metrics.items():
                if torch.is_tensor(v):
                    logs[k] = v.item()
                else:
                    logs[k] = v
            self._regen_metrics = {}
        super().log(logs)
