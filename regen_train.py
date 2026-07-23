import logging
import random
import sys
import yaml
from dataclasses import dataclass, field

import torch
import transformers
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    TrainingArguments,
    set_seed,
)

from regen_trainer import REGENTrainer, REGENConfig

logger = logging.getLogger(__name__)

# REGEN uses single-sided trajectory data, each containing:
#   prompt / response / nll_seq / advantage / origin_label
# No placeholder rejected columns are needed — REGEN is an independent framework.

PRESERVE_COLUMNS = (
    "nll_seq",
    "advantage",
    "origin_label",
)
REQUIRED_NLL_COLUMN = "nll_seq"
ORIGIN_LABEL_COLUMN = "origin_label"


@dataclass
class ModelArguments:
    model_name_or_path: str = field(default=None)
    torch_dtype: str = field(default="bfloat16")
    attn_implementation: str = field(default=None)
    trust_remote_code: bool = field(default=False)
    model_revision: str = field(default="main")
    use_peft: bool = field(default=False)


@dataclass
class DataArguments:
    data_path: str = field(default=None)
    preprocessing_num_workers: int = field(default=12)
    truncation_side: str = field(default="left")
    auto_insert_empty_system_msg: bool = field(default=True)


@dataclass
class REGENTrainingArguments(TrainingArguments):
    """Training arguments with REGEN-specific hyper-parameters."""
    # Prompt template
    prompt: str = field(default="ours")

    # Tokenization
    max_length: int = field(default=1024)
    max_prompt_length: int = field(default=512)

    # REGEN hyper-parameters
    regen_alpha: float = field(default=1.0)
    regen_clip_ratio_max: float = field(default=10.0)
    regen_length_normalize: bool = field(default=False)
    regen_require_old_nll: bool = field(default=True)
    regen_use_dataset_advantage: bool = field(default=True)
    regen_default_advantage: float = field(default=1.0)


def apply_chat_template(
    example,
    tokenizer,
    prompt,
):
    """Apply chat template to format prompt and response."""
    if prompt == 'alpaca':
        prompt_no_input = (
            "Below is an instruction that describes a task. "
            "Write a response that appropriately completes the request.\n\n"
            "### Instruction:\n{instruction}\n\n### Response:"
        )
    elif prompt == 'deepseek-math':
        prompt_no_input = "User: {instruction}\nPlease reason step by step, and put your final answer within \\boxed{{}}.\n\nAssistant:"
    elif prompt == 'qwen2-boxed':
        prompt_no_input = (
            "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
            "<|im_start|>user\n{instruction}\nPlease reason step by step, and put your final answer within \\boxed{{}}.<|im_end|>\n"
            "<|im_start|>assistant\n"
        )
    elif prompt == 'ours':
        prompt_no_input = (
            "<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n"
            "<|im_start|>user\n{instruction}<|im_end|>\n"
            "<|im_start|>assistant\n"
        )
    else:
        raise ValueError(f"Unknown prompt template: {prompt}")

    formatted_prompt = prompt_no_input.format(instruction=example['prompt'])
    response = example['response']

    return {
        'prompt': formatted_prompt,
        'response': response,
    }


def tokenize_dataset(example, tokenizer, max_length, max_prompt_length, truncation_mode, label_pad_token_id):
    """Tokenize a prompt-response pair into model inputs."""
    prompt = example["prompt"]
    response = example["response"]

    # Tokenize prompt (without special tokens)
    prompt_tokens = tokenizer(prompt, add_special_tokens=False)
    prompt_input_ids = prompt_tokens["input_ids"]
    prompt_attention_mask = prompt_tokens["attention_mask"]

    # Tokenize response (without special tokens)
    response_tokens = tokenizer(response, add_special_tokens=False)
    response_input_ids = response_tokens["input_ids"]
    response_attention_mask = response_tokens["attention_mask"]

    # Add BOS token if available
    if tokenizer.bos_token_id is not None:
        prompt_input_ids = [tokenizer.bos_token_id] + prompt_input_ids
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


def parse_yaml_config(yaml_path):
    """Parse a YAML config file into a dict."""
    with open(yaml_path, 'r') as f:
        return yaml.safe_load(f)


def main():
    import sys
    # Check if first arg is a yaml config file
    yaml_config = {}
    args = sys.argv[1:]
    if args and (args[0].endswith('.yaml') or args[0].endswith('.yml')):
        yaml_config = parse_yaml_config(args[0])
        args = args[1:]
        sys.argv = [sys.argv[0]] + args

    parser = HfArgumentParser((ModelArguments, DataArguments, REGENTrainingArguments))

    # Parse yaml config first, then override with command line args
    if yaml_config:
        # Convert yaml config to command line args format
        yaml_args = []
        for k, v in yaml_config.items():
            if v is None:
                continue
            if isinstance(v, bool):
                yaml_args.append(f"--{k}={str(v).lower()}")
            elif isinstance(v, list):
                # For list fields, join with comma
                yaml_args.append(f"--{k}={','.join(str(item) for item in v)}")
            elif isinstance(v, dict):
                # Skip dict fields (e.g. gradient_checkpointing_kwargs)
                continue
            else:
                yaml_args.append(f"--{k}={v}")
        # Prepend yaml args, then append command line args (command line takes precedence)
        all_args = yaml_args + args
        sys.argv = [sys.argv[0]] + all_args

    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    logger.info(f"Model parameters {model_args}")
    logger.info(f"Data parameters {data_args}")
    logger.info(f"Training parameters {training_args}")

    set_seed(training_args.seed)

    # --- Load dataset ---
    if ".json" in data_args.data_path:
        raw_datasets = load_dataset("json", data_files=data_args.data_path.split("||"))
    elif ".parquet" in data_args.data_path:
        raw_datasets = load_dataset("parquet", data_files=data_args.data_path.split("||"))
    else:
        raw_datasets = load_dataset(data_args.data_path)

    feat_names = list(raw_datasets["train"].features)
    logger.info(f"[REGEN] raw dataset columns: {feat_names}")

    # --- Sanity check: NLL must exist ---
    if REQUIRED_NLL_COLUMN not in feat_names:
        msg = (
            f"[REGEN] Required column `{REQUIRED_NLL_COLUMN}` not found in dataset "
            f"({data_args.data_path})."
        )
        raise ValueError(msg)

    has_origin_label = ORIGIN_LABEL_COLUMN in feat_names
    if not has_origin_label:
        logger.warning(
            f"[REGEN] column `{ORIGIN_LABEL_COLUMN}` not found; all samples will be "
            f"treated as positive (ρ = 1)."
        )

    # --- Tokenizer ---
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=model_args.trust_remote_code,
    )
    tokenizer.truncation_side = data_args.truncation_side
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # --- Apply chat template ---
    prompt_template = training_args.prompt

    raw_datasets = raw_datasets.map(
        apply_chat_template,
        fn_kwargs={
            "tokenizer": tokenizer,
            "prompt": prompt_template,
        },
        num_proc=data_args.preprocessing_num_workers,
        desc="REGEN: applying chat template",
    )

    # --- Tokenize ---
    max_length = getattr(training_args, 'max_length', 1024)
    max_prompt_length = getattr(training_args, 'max_prompt_length', 512)
    truncation_mode = "keep_end"
    label_pad_token_id = -100

    # Only remove non-preserved columns during tokenization
    feat_names = list(raw_datasets["train"].features)
    column_names = [c for c in feat_names if c not in PRESERVE_COLUMNS and c not in ("prompt", "response")]

    raw_datasets = raw_datasets.map(
        tokenize_dataset,
        fn_kwargs={
            "tokenizer": tokenizer,
            "max_length": max_length,
            "max_prompt_length": max_prompt_length,
            "truncation_mode": truncation_mode,
            "label_pad_token_id": label_pad_token_id,
        },
        num_proc=data_args.preprocessing_num_workers,
        remove_columns=column_names,
        desc="REGEN: tokenizing",
    )

    after_cols = list(raw_datasets["train"].features)
    logger.info(f"[REGEN] columns after tokenization: {after_cols}")
    if REQUIRED_NLL_COLUMN not in after_cols:
        raise RuntimeError(
            f"[REGEN] column `{REQUIRED_NLL_COLUMN}` was dropped during tokenization; "
            f"check PRESERVE_COLUMNS handling."
        )

    for index in random.sample(range(len(raw_datasets["train"])), min(3, len(raw_datasets["train"]))):
        sample = raw_datasets["train"][index]
        logger.info(f"Sample {index}: input_ids length = {len(sample['input_ids'])}")
        if ORIGIN_LABEL_COLUMN in sample:
            logger.info(f"origin_label sample {index}: {sample[ORIGIN_LABEL_COLUMN]}")
        if REQUIRED_NLL_COLUMN in sample:
            logger.info(
                f"nll sample {index}: nll={sample[REQUIRED_NLL_COLUMN]} "
                f"(log μ={-float(sample[REQUIRED_NLL_COLUMN]) if sample[REQUIRED_NLL_COLUMN] is not None else None})"
            )

    # --- Load model ---
    torch_dtype = (
        model_args.torch_dtype if model_args.torch_dtype in ["auto", None] else getattr(torch, model_args.torch_dtype)
    )

    model_kwargs = dict(
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        torch_dtype=torch_dtype,
        use_cache=False if training_args.gradient_checkpointing else True,
    )
    if model_args.attn_implementation is not None:
        model_kwargs["attn_implementation"] = model_args.attn_implementation

    logger.info("[REGEN] Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        **model_kwargs,
    )

    # --- REGEN config ---
    regen_config = REGENConfig(
        regen_alpha=training_args.regen_alpha,
        regen_clip_ratio_max=training_args.regen_clip_ratio_max,
        regen_length_normalize=training_args.regen_length_normalize,
        regen_require_old_nll=training_args.regen_require_old_nll,
        regen_use_dataset_advantage=training_args.regen_use_dataset_advantage,
        regen_default_advantage=training_args.regen_default_advantage,
        max_length=max_length,
        max_prompt_length=max_prompt_length,
        truncation_mode=truncation_mode,
        label_pad_token_id=label_pad_token_id,
    )

    # --- REGEN trainer ---
    trainer = REGENTrainer(
        model=model,
        args=training_args,
        train_dataset=raw_datasets["train"],
        eval_dataset=raw_datasets["test"] if "test" in raw_datasets.keys() else None,
        tokenizer=tokenizer,
        regen_config=regen_config,
        # Pass regen-specific params explicitly
        regen_alpha=regen_config.regen_alpha,
        regen_clip_ratio_max=regen_config.regen_clip_ratio_max,
        regen_length_normalize=regen_config.regen_length_normalize,
        regen_require_old_nll=regen_config.regen_require_old_nll,
        regen_use_dataset_advantage=regen_config.regen_use_dataset_advantage,
        regen_default_advantage=regen_config.regen_default_advantage,
    )

    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint

    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    metrics = train_result.metrics
    metrics["train_samples"] = len(raw_datasets["train"])
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    logger.info("*** Training complete ***")
    trainer.save_model(training_args.output_dir)
    logger.info(f"Model saved to {training_args.output_dir}")

    if trainer.accelerator.is_main_process:
        trainer.model.config.use_cache = True
        trainer.model.config.save_pretrained(training_args.output_dir)

    if training_args.do_eval:
        metrics = trainer.evaluate()
        metrics["eval_samples"] = len(raw_datasets["test"])
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)


if __name__ == "__main__":
    main()
