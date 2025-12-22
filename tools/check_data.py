import os
from dataclasses import asdict
from datetime import datetime
import wandb
from typing import Dict

import torch

import transformers

from configs.args import ModelArguments, DataArguments, TrainingArguments
# from uni_arch.train.llava_trainer import LLaVATrainer

from configs import conversation as conversation_lib
from uni_arch.train.custom_trainer import CustomTrainer

from transformers import AutoModelForCausalLM, Qwen3ForCausalLM
from datasets import load_from_disk, concatenate_datasets, load_dataset, DatasetDict, IterableDatasetDict
from uni_arch.train.data_collator import DataCollatorForSupervisedDataset, DataCollatorPacked

from modeling.uni_x_qwen3 import UniQwen3ForCausalLM
from modeling.shared_func_module import UniQwen3Config
# from modeling.uni_share_qwen3 import UniShareQwen3ForCausalLM
from modeling.uni_share_v3_qwen2_5 import UniShareQwen2ForCausalLM as UniShareQwen3ForCausalLM
from tools.log import main_logger
from accelerate import Accelerator
from modeling import shared_func_module

local_rank = None
accelerator = Accelerator()


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer,
                                   output_dir: str):
    """Collects the state dict and dump to disk."""

    if trainer.deepspeed:
        torch.cuda.synchronize()
        trainer.save_model(output_dir)
        return

    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {
            key: value.cpu()
            for key, value in state_dict.items()
        }
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


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
            subset = load_dataset(hgdata_path, streaming=data_args.streaming_data)
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
        train_dataset = load_dataset(data_path, streaming=data_args.streaming_data)
        if isinstance(train_dataset, DatasetDict) or isinstance(train_dataset, IterableDatasetDict):
            train_dataset = train_dataset["train"]
        if not data_args.streaming_data:
            sub_len = train_dataset.num_rows
            train_dataset = train_dataset.select(range(int(sub_len * percentage)))
        percentage = float(percentage)
        if shuffleseed != 0:
            print('shuffling...')
            train_dataset = train_dataset.shuffle(seed=shuffleseed)

    if not data_args.streaming_data:
        print(f"Total training samples: {train_dataset.num_rows}")
        main_logger.debug(f"\n\n{train_dataset[:4]}\n\n")

    if data_args.use_data_packing:
        data_collator = DataCollatorPacked(tokenizer=tokenizer, data_args=data_args)
    else:
        data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer, data_args=data_args)

    return dict(train_dataset=train_dataset,
                eval_dataset=None,
                data_collator=data_collator)


def train(attn_implementation=None):
    global local_rank

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    local_rank = training_args.local_rank

    # print("total processes:", accelerator.num_processes)

    # Prepare run name and output directory
    print(training_args.extra_tags, type(training_args.extra_tags))
    if isinstance(training_args.extra_tags, str):
        training_args.extra_tags = training_args.extra_tags.split(',')

    assert isinstance(training_args.extra_tags, list)
    if training_args.extra_tags is not None:
        tags_str = "-".join(training_args.extra_tags)
        training_args.run_name = f"{training_args.run_name}__{tags_str}"
        timestamp = datetime.now().strftime('%Y%m%d_%H%M')
        training_args.output_dir = os.path.join(training_args.output_dir, tags_str, timestamp)

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

    if data_args.add_sep_for_vis:
        data_args: DataArguments
        vis_sep_tokens_id = torch.tensor([tokenizer(_, add_special_tokens=False).input_ids[0] for _ in data_args.vis_sep_tokens.split(',')])
        print("vis_sep_tokens_id", vis_sep_tokens_id, tokenizer('<|vision_pad|>', add_special_tokens=False).input_ids[0])
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
            # Fallback to a default conversation template
            default_conv = "vicuna_v1" if "gemma" not in model_args.model_name_or_path else "gemma"
            conversation_lib.default_conversation = conversation_lib.conv_templates[default_conv]

    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)
    train_dataset = data_module["train_dataset"]
    ori_vocab_size = len(tokenizer)
    vision_start_token_id = tokenizer('<|vision_start|>', add_special_tokens=False).input_ids[0]
    cnt = 0
    data_lst = []
    for sample in train_dataset:
        data_lst.append(sample)
        cnt += 1
        if cnt == 50:
            break

    collator = data_module["data_collator"]
    sample = collator(data_lst)

    input_ids = sample["input_ids"]
    attn_mask = sample["attention_mask"]
    mask = shared_func_module.get_vision_mask(input_ids, ori_vocab_size + 20000, vision_start_token_id)
    input_ids[mask] = attn_mask[mask]
    input_ids[~mask] = 0
    print('\n')
    print(input_ids.tolist())
    print('\n')


if __name__ == "__main__":
    train()