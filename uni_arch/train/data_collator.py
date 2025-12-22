import os

import torch
import numpy as np
import random
import json
import transformers
import logging

from typing import Dict, Sequence, List, Optional
from configs.constants import IGNORE_INDEX
from dataclasses import dataclass

from modeling.shared_func_module import get_flash_attention_args_tensorized
from configs.args import DataArguments
from tools.log import main_logger
from accelerate import Accelerator

accelerator = Accelerator()


# This is a placeholder for the original function, assuming it's available in the execution scope.
def random_choice_t2iprompt_from_list():
    my_list = [
        ' Generate an image based on this description.',
        ' Create an image that captures the provided description.',
        ' Based on the previous text, produce a corresponding image.',
        ' Please illustrate the above text with a picture.',
        ' Translate the given description into a image.',
        ' Construct a visual representation of the above description.',
        ' Create a image that matches the text.',
        ' Formulate a visual expression that reflects the narrative just provided.',
        ' Give a visual depiction based on the above sentences.',
        ' Create an image using the information mentioned above as guidance.',
    ]
    return random.choice(my_list)


def random_choice_t2iprompt_from_list_empower():
    my_list = [
        # --- 原始列表 ---
        ' Generate an image based on this description.',
        ' Create an image that captures the provided description.',
        ' Based on the previous text, produce a corresponding image.',
        ' Please illustrate the above text with a picture.',
        ' Translate the given description into a image.',
        ' Construct a visual representation of the above description.',
        ' Create a image that matches the text.',
        ' Formulate a visual expression that reflects the narrative just provided.',
        ' Give a visual depiction based on the above sentences.',
        ' Create an image using the information mentioned above as guidance.',

        # --- 新增的中文prompt (命令式) ---
        '根据这段描述生成一张图片。',
        '把上面的文字变成一张画。',
        '请为以上内容创作一幅图像。',
        '根据这些信息，画一张图。',
        '将上述文本可视化。',
        '为这段话配一张图。',
        '我想让你根据这段描述画一幅画。',
        '把这个故事用图片展示出来。',

        # --- 新增的问句格式prompt (中英文) ---
        'Can you generate an image for the text above?',
        'Could you create a visual representation of this description?',
        'Would it be possible to illustrate the previous sentences with a picture?',
        'What would an image of the above description look like?',
        'Are you able to create a drawing from this text?',
        '你能根据这段描述生成一张图片吗？',
        '可以把上面的文字变成一幅画吗？',
        '能将上述内容可视化吗？',
        '根据这段文字生成的图片会是什么样的？',
        '麻烦你根据这段描述创作一幅图像？',
    ]

    return random.choice(my_list)

def random_choice_i2t_prompt_from_list_empower():
    caption_prompts = [
        # --- 英文图片描述 Prompt ---
        'Describe this image.',
        'What do you see in this picture?',
        'Write a short caption for this image.',
        'Provide a detailed description of the scene in the image.',
        'Generate a textual explanation for this picture.',
        'Summarize the main elements and actions depicted in the image.',
        'Give me a descriptive caption for the attached photo.',
        'Explain what is happening in this visual.',
        'Identify the key subjects and their context in this image.',
        'Craft a brief but informative description of this photograph.',
        'What is the subject of this image?',
        'Can you describe the mood or atmosphere of this picture?',
        'Write a caption that highlights the most important aspects of this image.',
        'Tell me about this image.',
        'Produce a concise caption that accurately reflects the content of this image.',
        'Analyze this image and provide a textual summary.',

        # --- 问句格式的图片描述 Prompt (中英文) ---
        'Can you describe what is in the image?',
        'Could you generate a caption for this picture?',
        'What would be a good description for this photo?',
        'How would you summarize the content of this image in words?',
    ]
    return random.choice(caption_prompts)


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer
    data_args: None

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        raise NotImplementedError


@dataclass
class DataCollatorPacked(object):
    """
    Collate examples for supervised fine-tuning by packing sequences into a single batch item.

    This collator avoids padding by concatenating multiple sequences into a single, long
    sequence. This is efficient for training with attention mechanisms like flash attention
    that can handle block-diagonal masks.

    Attributes:
        tokenizer (transformers.PreTrainedTokenizer): The tokenizer used for encoding.
        data_args (object): An object containing data-related arguments.
        buffer (List[Dict]): A stateful buffer to hold processed samples that are waiting
                             to be packed into a batch.

    Note:
        - This collator is STATEFUL. The `DataLoader` using it should have `num_workers=0`.
        - The `attention_mask` produced is not a standard binary mask. It's a 1D tensor
          of sequence IDs, e.g., [1, 1, 1, 2, 2, 2, 3, 3, ...], which is used by flash
          attention to prevent attention between different sequences in the pack.
    """

    tokenizer: transformers.PreTrainedTokenizer
    data_args: object  # Use a more specific type if possible, e.g., a dataclass
    buffer: List[Dict[str, torch.Tensor]]

    def __init__(self, tokenizer: transformers.PreTrainedTokenizer, data_args: DataArguments):
        self.tokenizer = tokenizer
        # Get the token ID for <|vision_start|>
        self.vision_start_token_id = self.tokenizer('<|vision_start|>', add_special_tokens=False).input_ids[0]
        self.real_text_vocab_size = len(tokenizer)
        self.data_args: DataArguments = data_args
        self.buffer = []
        self.logger = logging.getLogger(__name__)
        self.add_sep_for_vis = data_args.add_sep_for_vis
        self.vis_sep_tokens = [tokenizer(_, add_special_tokens=False).input_ids[0] for _ in data_args.vis_sep_tokens.split(',')]
        self.vis_sep_lens = [int(_) for _ in data_args.vis_sep_lens.split(',')]
        self.type_for_img_text = ['image_text', 't2i', 'i2t', 'i2t_multi', 'i2t_reverse']
        self.filter_data_type = None

    def set_filter_data_type(self, expected_data_type):
        self.filter_data_type = expected_data_type

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        # 1. Process all incoming instances and add them to the internal buffer.
        for sources in instances:
            processed_instance = self._process_source(sources)
            if processed_instance is None:
                main_logger.warning(f"processed_instance is None ... ")
            if processed_instance:
                self.buffer.append(processed_instance)

        # 2. Pack sequences from the buffer into a single batch.
        pack_max_length = self.tokenizer.model_max_length

        packed_input_ids = []
        packed_labels = []
        packed_attn_mask_ids = []

        current_pack_len = 0
        sequence_id = 1
        pack_has_image_text = False  # Flag to track if the current pack contains an image.

        # Keep packing as long as the buffer has samples.
        if len(self.buffer) == 0:
            main_logger.warning("buffer为空，这并不正常。")
        
        while self.buffer:
            instance = self.buffer[0]
            instance_len = len(instance["input_ids"])

            # If the next instance overflows, truncate it to fill the remaining space.
            if current_pack_len + instance_len > pack_max_length:
                remaining_len = pack_max_length - current_pack_len
                if remaining_len > 0:
                    instance = self.buffer.pop(0)  # Consume the instance.

                    # Truncate the instance to fit.
                    packed_input_ids.append(instance["input_ids"][:remaining_len])
                    packed_labels.append(instance["labels"][:remaining_len])
                    packed_attn_mask_ids.extend([sequence_id] * remaining_len)

                    current_pack_len += instance_len
                    sequence_id += 1

                    # 这里可能因为图片被截掉所以不一定为 true
                    # if instance.get("data_type") in self.type_for_img_text:
                    #     pack_has_image_text = True

                # The pack is now full, so finalize it.
                break

            # Dequeue the sample and add it to the current pack.
            instance = self.buffer.pop(0)
            # 用来分析梯度，准备指定type的数据
            if self.filter_data_type is not None and instance['data_type'] not in self.filter_data_type:
                continue

            # Check if the instance is an image-text sample and set the flag.
            if instance.get("data_type") in self.type_for_img_text:
                pack_has_image_text = True

            packed_input_ids.append(instance["input_ids"])
            packed_labels.append(instance["labels"])
            packed_attn_mask_ids.extend([sequence_id] * instance_len)

            current_pack_len += instance_len
            sequence_id += 1

        # 3. If no sequences were packed, return an empty dict.
        if not packed_input_ids:
            raise ValueError(f"no data?? packed_input_ids={packed_input_ids}")

        # 4. Concatenate all parts and create the final batch tensors.
        input_ids_tensor = torch.cat(packed_input_ids)
        labels_tensor = torch.cat(packed_labels)

        # If the packed batch contains no image_text data, insert placeholder vision tokens.
        if not pack_has_image_text:
            # Create a random image token ID. Assumes VQ codebook size is 8192.
            # The image token ID space starts after the text tokenizer's vocabulary.
            random_image_token_id = len(self.tokenizer) + random.randint(0, 8191)

            # Append the new tokens to input_ids
            new_tokens = torch.tensor([self.vision_start_token_id, random_image_token_id], dtype=torch.long)
            input_ids_tensor[-2: ] = new_tokens
            labels_tensor[-2: ] = new_tokens

            # Also update the attention mask IDs. Assign a new sequence ID to the appended tokens.
            new_sequence_id = sequence_id  # The next available sequence ID
            # packed_attn_mask_ids.extend([new_sequence_id] * len(new_tokens))
            packed_attn_mask_ids[-2: ] = [new_sequence_id] * 2

        attention_mask_tensor = torch.tensor(packed_attn_mask_ids, dtype=torch.long)

        # Reshape to [1, N] as expected by the model for a packed sequence.
        batch = dict(
            input_ids=input_ids_tensor.unsqueeze(0),
            labels=labels_tensor.unsqueeze(0),
            attention_mask=attention_mask_tensor.unsqueeze(0),
        )
        if self.data_args.packing_bug_fix:
            # 自定义的模型不需要。。不过默认打开吧
            cumulative_seqlens_q, max_length_q = get_flash_attention_args_tensorized(attention_mask_tensor[None])
            batch.update({
                "cumulative_seqlens_q": cumulative_seqlens_q,
                "cumulative_seqlens_k": cumulative_seqlens_q,
                "max_length_q": max_length_q,
                "max_length_k": max_length_q,
            })

        return batch

    def _process_source(self, sources: Dict) -> Optional[Dict[str, torch.Tensor]]:
        """
        Processes a single data source into tokenized input_ids and labels.
        This logic is adapted directly from the original padding-based collator
        to ensure correctness.
        """
        vq_resolution = self.data_args.vq_resolution
        T2I_ratio = self.data_args.T2I_ratio
        _data_type = sources.get("data_type", "text")

        if _data_type in self.type_for_img_text:
            vqcode = json.loads(sources['vqcode_{}'.format(str(vq_resolution))])
            vqcode = torch.tensor(vqcode) + len(self.tokenizer)

            if vqcode.shape[-1] != 1024:
                main_logger.warning(f"vqcode.shape[-1] != 1024 with shape {vqcode.shape[-1]}")
                return None

            if os.environ.get("DEBUGING", 0):
                print("debug", self.add_sep_for_vis, self.vis_sep_lens, self.vis_sep_tokens)
            if self.add_sep_for_vis:
                for sep_len, sep_token in zip(self.vis_sep_lens, self.vis_sep_tokens):
                    assert vqcode.shape[0] % sep_len == 0
                    n_seg = vqcode.shape[0] // sep_len
                    vqcode = torch.cat([vqcode.view(n_seg, sep_len), torch.full(size=(n_seg, 1), fill_value=sep_token)], dim=-1).view(-1)
                    if os.environ.get("DEBUGING", 0):
                        print("debug vqcode", len(vqcode))

            if np.random.rand() < T2I_ratio:  # T2I mode
                prompt = random_choice_t2iprompt_from_list()
                if 'multi' in _data_type:
                    prompt = f"Image Size: Width is {sources['width']} Height is {sources['height']}." + prompt

                caption = sources["caption"] if sources['text'] is None else sources['text']
                text = caption + prompt

                if np.random.rand() > 0.9:
                    text = "<unconditional>"
                    if 'multi' in _data_type:
                        text = f'Image Size: Width is {sources["width"]} Height is {sources["height"]}. {text}'

                # This structure mirrors the original collator exactly
                full_text = text + f'<|vision_start|><|vision_end|>{self.tokenizer.eos_token}'
                input_ids = self.tokenizer(
                    full_text,
                    return_tensors="pt",
                    max_length=self.tokenizer.model_max_length,
                    truncation=True,
                    add_special_tokens=False,
                ).input_ids[0]

                # The point to insert vqcode is right before the last two tokens (<|vision_end|> and eos)
                instruction_len = len(input_ids) - 2
                input_ids = torch.cat([input_ids[:instruction_len], vqcode, input_ids[instruction_len:]])

            else:  # Caption mode
                caption = sources["caption"] if sources['text'] is None else sources['text']
                caption = caption + f'{self.tokenizer.eos_token}'
                instruction = '<|vision_start|><|vision_end|>The caption of this image is:'

                # Tokenize separately, just like the original collator
                caption_ids = self.tokenizer(
                    caption, return_tensors="pt", max_length=self.tokenizer.model_max_length,
                    truncation=True, add_special_tokens=False).input_ids[0]
                instruct_id = self.tokenizer(
                    instruction, return_tensors="pt", max_length=self.tokenizer.model_max_length,
                    truncation=True, add_special_tokens=False).input_ids[0]

                # Reconstruct precisely as the original collator did
                input_ids = torch.cat([
                    instruct_id[:1],  # [<|vision_start|>]
                    vqcode,
                    instruct_id[1:],  # [<|vision_end|>, The caption...]
                    caption_ids  # [CAPTION_TEXT, EOS] (strips duplicate BOS)
                ])
                instruction_len = len(input_ids) - len(caption_ids) + 1

            # Common logic for image_text type
            # Truncate if the final assembled sequence is too long
            if len(input_ids) > self.tokenizer.model_max_length:
                raise ValueError("理论上不应该出现这个情况")

            num_vis_tokens = (input_ids >= len(self.tokenizer)).sum().item()
            assert len(self.tokenizer) == 151665, "only qwen2.5 now .. "
            assert num_vis_tokens % 1024 == 0, f"{num_vis_tokens}"

            targets = input_ids.clone()
            # 我的理解从： 没必要 ignore，需要让模型学习图片的分布是什么样子的
            # 到：好像需要 ignore，来让模型对齐图片 token 和文字 token 的对应关系
            # 但是也不一定
            if self.data_args.ignore_instruction:
                targets[:instruction_len] = IGNORE_INDEX
            # Include data_type to inform the packing logic
            return dict(input_ids=input_ids, labels=targets, data_type=_data_type)

        else:  # Text pretrain mode
            if "input_ids" in sources:
                input_ids = torch.tensor(sources["input_ids"], dtype=torch.long)
            else:
                text = sources.get('text', '')
                if not text or text == 'no': return None
                input_ids = self.tokenizer(
                    text, return_tensors="pt", max_length=self.tokenizer.model_max_length, truncation=True
                ).input_ids[0]

            targets = input_ids.clone()
            # Include data_type to inform the packing logic
            return dict(input_ids=input_ids, labels=targets, data_type=_data_type)


def convert_qa_pair_to_messages(q: str, a: str):
    return [{"role": "user", "content": q}, {"role": "assistant", "content": a}]


def convert_alpaca_to_messages(instr, inp, out):
    return [{"role": "user", "content": f"{instr}\n{inp}"}, {"role": "assistant", "content": out}]


@dataclass
class DataCollatorSFTPacked(DataCollatorPacked):

    tokenizer: transformers.Qwen2Tokenizer
    data_args: object  # Use a more specific type if possible, e.g., a dataclass
    buffer: List[Dict[str, torch.Tensor]]

    def __init__(self, tokenizer: transformers.PreTrainedTokenizer, data_args: DataArguments):
        super().__init__(tokenizer, data_args)

    @torch.no_grad()
    def _process_source(self, sources: Dict) -> Optional[Dict[str, torch.Tensor]]:
        """
        Processes a single data source into tokenized input_ids and labels.
        This logic is adapted directly from the original padding-based collator
        to ensure correctness.
        """
        vq_resolution = self.data_args.vq_resolution
        _data_type = sources.get("data_type", "text")
        T2I_ratio = self.data_args.T2I_ratio

        if _data_type in self.type_for_img_text:
            vqcode = json.loads(sources['vqcode_{}'.format(str(vq_resolution))])
            vqcode = torch.tensor(vqcode, dtype=torch.long) + len(self.tokenizer)
            assert vqcode.dim() == 1
            # assert len(self.tokenizer) == 151936, f"{len(self.tokenizer)}"
            # 实际 size 是 151665
            if vqcode.shape[-1] != 1024:
                main_logger.warning(f"vqcode.shape[-1] != 1024 with shape {vqcode.shape[-1]}")
                return None

            if os.environ.get("DEBUGING", 0):
                print("debug", self.add_sep_for_vis, self.vis_sep_lens, self.vis_sep_tokens)
            if self.add_sep_for_vis:
                for sep_len, sep_token in zip(self.vis_sep_lens, self.vis_sep_tokens):
                    assert vqcode.shape[0] % sep_len == 0
                    n_seg = vqcode.shape[0] // sep_len
                    vqcode = torch.cat([vqcode.view(n_seg, sep_len), torch.full(size=(n_seg, 1), fill_value=sep_token)], dim=-1).view(-1)
                    if os.environ.get("DEBUGING", 0):
                        print("debug vqcode", len(vqcode))

            if _data_type == 't2i' or _data_type == 'image_text':
                if random.random() > T2I_ratio:
                    _data_type = 'i2t_reverse'
                    sources['conversations'] = [{}, {}]
                    sources['conversations'][0]['value'] = random_choice_i2t_prompt_from_list_empower()
                    sources['conversations'][1]['value'] = sources["caption"] if sources['text'] is None else sources['text']

            if _data_type == 't2i' or _data_type == 'image_text':  # T2I mode
                prompt = random_choice_t2iprompt_from_list_empower()
                if 'multi' in _data_type:
                    prompt = f"Image Size: Width is {sources['width']} Height is {sources['height']}." + prompt

                caption = sources["caption"] if sources['text'] is None else sources['text']
                text = caption + prompt

                if np.random.rand() < 0.1:
                    text = "<unconditional>"
                    if 'multi' in _data_type:
                        text = f'Image Size: Width is {sources["width"]} Height is {sources["height"]}. {text}'

                # This structure mirrors the original collator exactly
                full_text = text + f'<|vision_start|><|vision_end|>{self.tokenizer.eos_token}'
                input_ids = self.tokenizer(
                    full_text,
                    return_tensors="pt",
                    max_length=self.tokenizer.model_max_length,
                    truncation=True,
                    add_special_tokens=False,
                ).input_ids[0]

                # The point to insert vqcode is right before the last two tokens (<|vision_end|> and eos)
                instruction_len = len(input_ids) - 2
                input_ids = torch.cat([input_ids[:instruction_len], vqcode, input_ids[instruction_len:]])

                targets = input_ids.clone()
                if self.data_args.ignore_instruction:
                    targets[:instruction_len] = IGNORE_INDEX

            else:  # VQA Mode
                # 假设，肯定是偶数human，奇数ai
                assert self.data_args.ignore_instruction, 'ignore之后效果好不少'
                if sources['conversations'] is None or len(sources['conversations']) == 0:
                    assert sources['text'] is not None
                    sources['conversations'] = []
                    convs = sources['text'].split('^^^turn_split^^^')
                    for turn in convs:
                        human, ai = turn.split('^^^pair_split^^^')
                        if len(human) == 0 and len(convs) == 1:
                            human = random_choice_i2t_prompt_from_list_empower()
                        sources['conversations'].append({'from': 'human', 'value': human})
                        sources['conversations'].append({'from': 'assistant', 'value': ai})

                def _tkn(text):
                    # 如果 text 为空，好像会产生float返回
                    if text is None or len(text) == 0:
                        main_logger.error(f"数据为空 {_data_type} {len(sources['conversations'])}")
                    return self.tokenizer(text, add_special_tokens=False, return_tensors='pt').input_ids[0].to(dtype=torch.long)

                input_ids = torch.cat([_tkn('<|vision_start|>'), vqcode, _tkn('<|vision_end|>')], dim=-1)
                assert input_ids.shape[-1] == 1026 and input_ids.dim() == 1
                targets = input_ids.clone().fill_(IGNORE_INDEX)
                for chat_i, mes in enumerate(sources['conversations']):
                    this_mes_ids = _tkn('\n' * random.randint(1, 3) + mes['value'])
                    input_ids = torch.cat([input_ids, this_mes_ids], dim=-1)
                    if chat_i % 2 == 0: # human no grad
                        targets = torch.cat([targets, torch.empty_like(this_mes_ids).fill_(IGNORE_INDEX)])
                    else:
                        targets = torch.cat([targets, this_mes_ids.clone()])

            if len(input_ids) > self.tokenizer.model_max_length:
                main_logger.error(f"len(input_ids) > self.tokenizer.model_max_length, current len(input_ids) = {len(input_ids)}")
                input_ids = input_ids[:self.tokenizer.model_max_length]
                targets = targets[:self.tokenizer.model_max_length]

            num_vis_tokens = (input_ids >= len(self.tokenizer)).sum().item()
            assert len(self.tokenizer) == 151665, "only qwen2.5 now .. "
            assert num_vis_tokens % 1024 == 0, f"{num_vis_tokens}"
            assert targets.dtype == input_ids.dtype == torch.long, f"{targets.dtype} {input_ids.dtype} {vqcode.dtype}"
            # Include data_type to inform the packing logic
            return dict(input_ids=input_ids, labels=targets, data_type=_data_type)

        else:  # Text pretrain mode
            if "input_ids" in sources:
                input_ids = torch.tensor(sources["input_ids"], dtype=torch.long)
                targets = input_ids.clone()
            else:
                text = sources.get('text', None)
                instruction_len = 0
                if text is None and 'instruction' in sources:
                    # alpaca type data
                    instr, inp, out = sources['instruction'], sources['input'], sources['output']
                    _newline = '\n'
                    no_grad_text = f"{instr}{_newline*random.randint(1, 3)}{inp}"
                    grad_text = f"{_newline*random.randint(1, 3)}{out}{self.tokenizer.eos_token}"
                    input_ids_no_grad = self.tokenizer(
                        no_grad_text, return_tensors="pt", max_length=self.tokenizer.model_max_length, truncation=True, add_special_tokens=False
                    ).input_ids[0]
                    input_ids_grad = self.tokenizer(
                        grad_text, return_tensors="pt", max_length=self.tokenizer.model_max_length, truncation=True, add_special_tokens=False
                    ).input_ids[0]
                    instruction_len = input_ids_no_grad.shape[-1]
                    input_ids = torch.cat([input_ids_no_grad, input_ids_grad], dim=-1)
                    targets = input_ids.clone()
                    if self.data_args.ignore_instruction:
                        targets[:instruction_len] = IGNORE_INDEX

                elif text is None and 'conversations' in sources:
                    def _tkn(_x):
                        # 如果 text 为空，好像会产生float返回
                        if _x is None or len(_x) == 0:
                            main_logger.error(f"数据为空 {_data_type} {len(sources['conversations'])}")
                        return self.tokenizer(_x, add_special_tokens=False, return_tensors='pt').input_ids[0].to(dtype=torch.long)

                    input_ids = torch.zeros(0, dtype=torch.long, device='cpu')
                    targets = input_ids.clone()
                    for chat_i, mes in enumerate(sources['conversations']):
                        this_mes_ids = _tkn('\n' * random.randint(1, 3) + mes['value'])
                        input_ids = torch.cat([input_ids, this_mes_ids], dim=-1)
                        if chat_i % 2 == 0:  # human no grad
                            targets = torch.cat([targets, torch.empty_like(this_mes_ids).fill_(IGNORE_INDEX)])
                        else:
                            targets = torch.cat([targets, this_mes_ids.clone()])

                else:
                    input_ids = self.tokenizer(
                        f"{text}{self.tokenizer.eos_token}", return_tensors="pt", max_length=self.tokenizer.model_max_length,
                        truncation=True, add_special_tokens=False,
                    ).input_ids[0]
                    targets = input_ids.clone()
                    if self.data_args.ignore_instruction:
                        targets[:instruction_len] = IGNORE_INDEX

            return dict(input_ids=input_ids, labels=targets, data_type=_data_type)
