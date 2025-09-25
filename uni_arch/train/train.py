# ------------------------------------------------------------------------
# Modified from LLaVA (https://github.com/haotian-liu/LLaVA)
# Copyright 2025 Junfeng Wu
# ------------------------------------------------------------------------
import os
from dataclasses import asdict
from datetime import datetime
import wandb
from typing import Dict, Tuple

import torch
import datasets
import transformers

from configs.args import ModelArguments, DataArguments, TrainingArguments
# from uni_arch.train.llava_trainer import LLaVATrainer

from configs import conversation as conversation_lib
from uni_arch.train.custom_trainer import CustomTrainer

from transformers import AutoModelForCausalLM, Qwen3ForCausalLM, PreTrainedModel, PreTrainedTokenizer
from datasets import load_from_disk, concatenate_datasets, load_dataset, DatasetDict, IterableDatasetDict
from uni_arch.train.data_collator import DataCollatorForSupervisedDataset, DataCollatorPacked, DataCollatorSFTPacked

from modeling.uni_x_qwen3 import UniQwen3ForCausalLM
from modeling.shared_func_module import UniQwen3Config
# from modeling.uni_share_qwen3 import UniShareQwen3ForCausalLM
from modeling.uni_share_v3_qwen2_5 import UniShareQwen2ForCausalLM as UniShareQwen3ForCausalLM
from modeling.mot_qwen import MoTQwen3ForCausalLM
from modeling.moe_qwen import HardMoeQwen3CausalLM
from tools.log import main_logger
from accelerate import Accelerator
from configs.data_features import FEATURES

local_rank = None
accelerator = Accelerator()
datasets.config.STREAMING_READ_MAX_RETRIES = 10000  # noqa


def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer,
                                data_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    data_path = data_args.data_path
    percentage = data_args.percentage
    shuffleseed = data_args.shuffleseed

    if '^^' in data_path:
        data_paths = data_path.split('^^')
        percentages = [float(p) for p in percentage.split('^^')] if '^^' in percentage else [float(percentage)] * len(
            data_paths)
        assert len(percentages) == len(data_paths)

        hgdata_list = []
        print('loading subsets...')
        for percent, hgdata_path in zip(percentages, data_paths):
            subset = load_dataset(hgdata_path, streaming=data_args.streaming_data, features=FEATURES)
            if isinstance(subset, DatasetDict) or isinstance(subset, IterableDatasetDict):
                subset = subset["train"]
            if not data_args.streaming_data:
                sub_len = subset.num_rows
                subset = subset.select(range(int(sub_len * percent)))
            hgdata_list.append(subset)
        train_dataset = concatenate_datasets(hgdata_list)
        if shuffleseed != 0:
            print('shuffling...')
            train_dataset = train_dataset.shuffle(seed=shuffleseed)
    else:
        print('loading subsets...')
        if data_args.streaming_data:
            train_dataset = load_dataset(data_path, streaming=True, features=FEATURES)
        else:
            train_dataset = load_dataset(data_path, streaming=False, features=FEATURES, num_proc=8)
        if isinstance(train_dataset, DatasetDict) or isinstance(train_dataset, IterableDatasetDict):
            train_dataset = train_dataset["train"]
        percentage = float(percentage)
        if not data_args.streaming_data:
            sub_len = train_dataset.num_rows
            train_dataset = train_dataset.select(range(int(sub_len * percentage)))
        if shuffleseed != 0:
            print('shuffling...')
            train_dataset = train_dataset.shuffle(seed=shuffleseed)

    if not data_args.streaming_data:
        print(f"Total training samples: {train_dataset.num_rows}")
        main_logger.debug(f"\n\n{train_dataset[:4]}\n\n")

    if data_args.use_data_packing == 2:
        data_collator = DataCollatorSFTPacked(tokenizer=tokenizer, data_args=data_args)
    elif data_args.use_data_packing == 1:
        data_collator = DataCollatorPacked(tokenizer=tokenizer, data_args=data_args)
    else:
        data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer, data_args=data_args)

    return dict(train_dataset=train_dataset,
                eval_dataset=None,
                data_collator=data_collator)


def load_model_and_tokenizer(model_args: ModelArguments, data_args: DataArguments,
                             training_args: TrainingArguments, attn_implementation: str = None) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
    """Load model and tokenizer with specified configurations."""
    # Load tokenizer
    tokenizer_kwargs = {
        "cache_dir": training_args.cache_dir,
        "model_max_length": training_args.model_max_length,
        "padding_side": "right",
    }
    if "gemma" not in model_args.model_name_or_path:
        tokenizer_kwargs["use_fast"] = False

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        **tokenizer_kwargs
    )
    assert tokenizer.model_max_length == training_args.model_max_length

    # Configure tokenizer and conversation template
    if data_args.add_sep_for_vis:
        from modeling import shared_func_module
        vis_sep_tokens_id = torch.tensor([tokenizer(_, add_special_tokens=False).input_ids[0] for _ in data_args.vis_sep_tokens.split(',')])
        shared_func_module.get_vision_mask = shared_func_module.reset_get_vis_mask(vis_sep_tokens_id)

    if model_args.version == "v0":
        raise NotImplementedError
    elif model_args.version == "v0.5":
        tokenizer.pad_token = tokenizer.unk_token
    else:
        tokenizer.pad_token = tokenizer.unk_token
        if model_args.version in conversation_lib.conv_templates:
            conversation_lib.default_conversation = conversation_lib.conv_templates[model_args.version]
        else:
            default_conv = "vicuna_v1" if "gemma" not in model_args.model_name_or_path else "gemma"
            conversation_lib.default_conversation = conversation_lib.conv_templates[default_conv]

    # Load model
    torch_dtype = (
        torch.bfloat16 if training_args.bf16 else torch.float16
    )
    device_map = {"": training_args.device} if not training_args.deepspeed else None

    if model_args.model_custom_cls == "auto" or model_args.model_custom_cls == "force_qwen3":
        cls = AutoModelForCausalLM if model_args.model_custom_cls == "auto" else Qwen3ForCausalLM
        model = cls.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            attn_implementation=attn_implementation,
            torch_dtype=torch_dtype,
            device_map=device_map,
        )

    elif model_args.model_custom_cls == "uni_qwen":
        cfg = UniQwen3Config.from_pretrained(model_args.model_name_or_path, attn_implementation=attn_implementation)
        vision_start_token_id = tokenizer('<|vision_start|>', add_special_tokens=False).input_ids[0]
        cfg.set_extra_args(
            # uni x params
            vision_encode_layers=model_args.vision_encode_layers,
            vision_decode_layers=model_args.vision_decode_layers,
            vision_vocab_size=8192,
            all_modal_visible=model_args.all_modal_visible,
            real_text_vocab_size=len(tokenizer),
            vision_start_id=vision_start_token_id,
            add_sep_for_vis=data_args.add_sep_for_vis,
            vis_sep_tokens=data_args.vis_sep_tokens,
            vis_sep_lens=data_args.vis_sep_lens,
            skip_connection_loc=model_args.skip_connection_loc,
            # uni share params
            add_share_ffn=model_args.add_share_ffn,
            ffn_share_size=model_args.ffn_share_size,
            ffn_vision_size=model_args.ffn_vision_size,
            use_share_attn=model_args.use_share_attn,
            use_share_ffn=model_args.use_share_ffn,
            attn_v_q_heads=model_args.attn_v_q_heads,
            # baseline moe params
            decoder_sparse_step=model_args.decoder_sparse_step,
        )

        if model_args.model_spec_module == "x":
            cls = UniQwen3ForCausalLM
        elif model_args.model_spec_module == "share":
            cls = UniShareQwen3ForCausalLM
        elif model_args.model_spec_module == "moe":
            cls = HardMoeQwen3CausalLM
        elif model_args.model_spec_module == "mot":
            cls = MoTQwen3ForCausalLM
        else:
            raise ValueError("Unsupported model spec module")

        model = cls.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            attn_implementation=attn_implementation,
            torch_dtype=torch_dtype,
            device_map=device_map,
            config=cfg,
        )
    else:
        raise ValueError("Unsupported model_custom_cls")

    return model, tokenizer


def train(attn_implementation=None):
    global local_rank

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    local_rank = training_args.local_rank

    # Prepare run name and output directory
    print(training_args.extra_tags, type(training_args.extra_tags))
    if isinstance(training_args.extra_tags, str):
        training_args.extra_tags = training_args.extra_tags.split(',')

    assert isinstance(training_args.extra_tags, list)
    tags_str = None
    if training_args.extra_tags is not None:
        tags_str = "-".join(training_args.extra_tags)
        training_args.run_name = f"{training_args.run_name}__{tags_str}"
        timestamp = datetime.now().strftime('%Y%m%d_%H%M')
        training_args.output_dir = os.path.join(training_args.output_dir, tags_str, timestamp)

    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(model_args, data_args, training_args, attn_implementation)

    # Initialize vision weights if specified
    if training_args.init_vision_with_text and hasattr(model.model, "init_vision_weights"):
        model.model.init_vision_weights()

    # Freeze layers if specified
    if training_args.unfreeze_keys != "train-all":
        freeze_tag_exist = any("freeze" in t for t in training_args.extra_tags)
        assert freeze_tag_exist, "Freeze tag is required when unfreezing specific keys."

        unfreeze_keys = set(training_args.unfreeze_keys.split(','))
        for n, p in model.named_parameters():
            p.requires_grad = any(k in n for k in unfreeze_keys)

    # Disable cache for training
    model.config.use_cache = False

    # Enable gradient checkpointing
    if training_args.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)

            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    # Prepare data module
    data_module = make_supervised_data_module(tokenizer=tokenizer,
                                              data_args=data_args)
    # Initialize wandb
    if accelerator.is_main_process:
        _cfg = {k: v for _args in [model_args, data_args, training_args] for k, v in asdict(_args).items()}
        wandb.init(
            project=training_args.project_name, name=training_args.run_name,
            config=_cfg, tags=training_args.extra_tags,
        )

    # Initialize trainer
    trainer = CustomTrainer(model=model,
                            tokenizer=tokenizer,
                            args=training_args,
                            **data_module)

    # Manually load optimizer state to continue SFT
    optimizer_path = os.path.join(model_args.model_name_or_path, "optimizer.pt")
    if os.path.exists(optimizer_path):
        # Trainer will not create a new optimizer if it already exists.
        trainer.create_optimizer()
        optimizer_state = torch.load(optimizer_path, map_location='cpu')
        trainer.optimizer.load_state_dict(optimizer_state)
        accelerator.print(f"Successfully loaded optimizer state from {optimizer_path}")

    # Start training
    if accelerator.is_main_process:
        print("[check] resume_from_checkpoint", training_args.resume_from_checkpoint)
    if "checkpoint" in training_args.resume_from_checkpoint:
        trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
    else:
        training_args.resume_from_checkpoint = None
        trainer.train()

    model.config.use_cache = True
    trainer.save_model()
    try:
        # Manually save optimizer states
        trainer._save_optimizer_and_scheduler(f"../mock/ckpts/optimizer_states/{tags_str}/{timestamp}/")  # noqa
    except Exception as e:
        print(f"[Error] save optimizer ... {e}")


if __name__ == "__main__":
    train()