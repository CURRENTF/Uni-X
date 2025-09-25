import fire
import torch
import os
import base64
import io

from torch.nn import functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM, LogitsProcessor, LogitsProcessorList, Qwen2Tokenizer, Qwen2ForCausalLM
from typing import Optional, List, Union
from data_process.vqgan.image_tokenizer import ImageTokenizer
from PIL import Image
from tools.log import main_logger
from modeling.uni_x_qwen3 import UniQwen3ForCausalLMInference


# --- 1. 核心逻辑：模态约束 LogitsProcessor ---
# LogitsProcessor是Hugging Face提供的一个钩子（hook），允许我们在生成过程的每一步修改模型的输出logits。
class ModalLogitsProcessor(LogitsProcessor):
    """
    一个 LogitsProcessor，用于根据指定的模态（文本或图像）约束词汇表的输出。
    它通过将不希望生成的词元（token）的概率分数（logits）设置为负无穷大来实现这一点。
    """

    def __init__(self,
                 ori_vocab_size: int,
                 modal_to_generate: str = 'text',
                 vis_sep_token: int = -1,
                 vis_sep_len: int = 32):
        """
        初始化处理器。

        Args:
            ori_vocab_size (int): 原始文本词汇表的大小。这是区分文本和图像词元的关键。
            modal_to_generate (str): 'text' 或 'image'，指定要生成的模态。
            vis_sep_token (int): 视觉分隔符的token id，如果模型设计中有这个特殊token。
            vis_sep_len (int): 视觉分隔符的插入周期（例如，每32个图像token插入一个）。
        """
        if modal_to_generate not in ['text', 'image']:
            raise ValueError("`modal_to_generate` 必须是 'text' 或 'image'")

        self.ori_vocab_size = ori_vocab_size
        self.modal_to_generate = modal_to_generate
        self.vis_sep_token = vis_sep_token
        self.vis_sep_len = vis_sep_len
        self.cnt = 0  # 计数器，用于追踪生成图像token的步数

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        """
        在每个生成步骤中被调用，用于修改 logits (scores)。

        Args:
            input_ids (torch.LongTensor): 到目前为止已生成的词元ID。
            scores (torch.FloatTensor): 当前步骤的原始 logits 张量。

        Returns:
            torch.FloatTensor: 修改后的 logits 张量。
        """
        self.cnt += 1

        if self.modal_to_generate == 'text':
            # --- 文本生成模式 ---
            # ID大于等于ori_vocab_size的都是图像词元。
            scores[:, self.ori_vocab_size:] = -float('inf')

        elif self.modal_to_generate == 'image':
            # --- 图像生成模式 ---
            # 1. 检查是否需要强制插入视觉分隔符（如果模型需要）。
            # 例如，每 (vis_sep_len + 1) 步就强制生成一个分隔符。
            if self.vis_sep_token != -1 and self.cnt % (self.vis_sep_len + 1) == 0:
                # 将所有token的logit设为-inf，除了视觉分隔符token。
                scores[:] = -float('inf')
                scores[:, self.vis_sep_token] = 0.0  # 将分隔符的logit设为0，确保它被选中。
            else:
                # 2. 正常生成图像token时，阻止模型生成任何文本词元。
                # 文本词元的ID小于ori_vocab_size。
                scores[:, :self.ori_vocab_size] = -float('inf')

        return scores


class SimpleTokenConstrainedLogitsProcessor(LogitsProcessor):
    def __init__(self, acceptable_tokens: torch.Tensor):
        # self.acceptable_tokens = acceptable_tokens
        self.banned_tokens = ~acceptable_tokens
        # assert acceptable_tokens.sum().item() == 5, f"为了 debug {acceptable_tokens.sum().item()}"

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        scores[:, self.banned_tokens] = float('-inf')
        return scores


def get_tensor_of_acceptable_tokens(tokens: List, vocab_size: int) -> torch.Tensor:
    """
    将一个逗号分隔的token ID字符串解析为一个布尔张量。
    张量中对应ID的位置为True，表示该token是可接受的。
    """
    # tokens 通过 ',' 分隔
    # 创建一个全为 False 的布尔张量，代表整个词汇表
    acceptable_mask = torch.zeros(vocab_size, dtype=torch.bool)
    # 使用张量索引一次性将所有有效ID的位置设置为True
    acceptable_mask[tokens] = True
    return acceptable_mask


# --- 2. API 工厂函数 ---
def any_modal_chat_api(
        model: Qwen2ForCausalLM,
        tokenizer: Qwen2Tokenizer,
        img_tokenizer: ImageTokenizer,
        ori_vocab_size: int,
        vis_sep_token: int = -1,
        vis_sep_len: int = 32
):
    """
    一个API工厂函数（Factory Function）。
    它接收模型等核心组件作为参数，并返回一个配置好的、可随时使用的 `generate` 函数。
    这种模式避免了每次调用生成函数时都重复传入模型和分词器。

    Args:
        model (AutoModelForCausalLM): 预训练的多模态大语言模型。
        tokenizer (AutoTokenizer): 对应的分词器。
        img_tokenizer (ImageTokenizer): 用于处理图像 tokenization 的 VQGAN 分词器。
        ori_vocab_size (int): 原始文本词汇表的大小。
        vis_sep_token (int): 视觉分隔符的 token id，如果没有则为 -1。
        vis_sep_len (int): 视觉分隔符的插入周期。

    Returns:
        function: 一个功能强大的、闭包了模型等信息的 generate 函数。
    """
    device = model.device  # 从模型中获取设备信息（CPU 或 CUDA）
    tokenizer.padding_side = 'left'

    def generate(
            # --- 输入选项 ---
            prompts: List[str],
            imgs_base64: Optional[List[List[str]]] = None,  # 一个prompt里可能有多张图片
            img_placeholder_in_prompt: str = '<img>',  # 应 assert 图片数目与 placeholder 的数目一致
            unconditional_prompt: str = '<unconditional>',

            # --- 生成控制参数 ---
            modal_to_generate: str = "text",
            max_new_tokens: int = 1024,
            min_new_tokens: int = 0,
            # 如果为None时，不启用
            acceptable_tokens: Optional[str] = None,
            prompts_has_img_tokens: int = 1,

            # --- CFG (Classifier-Free Guidance) 参数 ---
            cfg_scale: float = 1.0,

            # --- 采样策略参数 ---
            do_sample: bool = True,
            temperature: float = 1.0,
            top_k: int = 50,
            top_p: float = 0.95,

            # --- 结束标志 ---
            eos_token_id: Optional[Union[int, List[int]]] = None,
            decode_in_the_end: bool = True,
            **kwargs,
    ) -> Union[torch.Tensor, List[str], List[Image.Image]]:
        """
        根据指定模态（文本或图像）生成词元序列。
        支持直接输入prompts列表或预分词的input_ids。
        """
        # --- 1. 输入校验与处理 ---
        use_cfg = (cfg_scale > 1.0 and modal_to_generate == "image")  # 判断是否启用CFG

        if imgs_base64 is not None:
            # 处理多模态输入：将文本和图像的 token 交错拼接
            assert len(prompts) == len(imgs_base64), "提示(prompts)的数量必须与图像列表(imgs_base64)的数量一致。"

            all_combined_ids = []
            for prompt, img_list_b64 in zip(prompts, imgs_base64):
                if prompts_has_img_tokens:
                    assert prompt.count(img_placeholder_in_prompt) == len(img_list_b64), \
                        f"每个提示中的图像占位符数量必须与提供的图像数量一致。{prompt.count(img_placeholder_in_prompt)}, {len(img_list_b64)}, {prompt}"
                else:
                    raise NotImplementedError

                # 解码 base64 并使用 VQGAN tokenizer 将图像转换为 token
                img_tokens_list = []
                for img_b64 in img_list_b64:
                    img_bytes = base64.b64decode(img_b64)
                    pil_img = Image.open(io.BytesIO(img_bytes))
                    if os.environ.get("DEBUG", False):
                        pil_img.save('test_samples/test_input.jpg')
                    img_toks = torch.tensor(img_tokenizer.img_tokens_from_pil(pil_img), device=device, dtype=torch.long)
                    if os.environ.get("DEBUG", False):
                        _img = img_tokenizer.pil_from_img_toks(img_toks)
                        _img.save('test_samples/test_input_encoded.jpg')
                    img_tokens_list.append(img_toks + ori_vocab_size)

                # 将提示文本按占位符分割，并分别 tokenize
                text_parts = prompt.split(img_placeholder_in_prompt)
                text_tokens_list = [torch.tensor(tokenizer.encode(part, add_special_tokens=False), device=device, dtype=torch.long) for part in text_parts]

                # 交错合并文本 token 和图像 token
                combined_ids = [text_tokens_list[0]]
                for i, img_toks in enumerate(img_tokens_list):
                    combined_ids.append(img_toks)
                    combined_ids.append(text_tokens_list[i + 1])

                all_combined_ids.append(torch.cat(combined_ids))

            # 手动进行左填充，以创建 batch
            max_len = max(len(ids) for ids in all_combined_ids)
            input_ids = torch.full((len(all_combined_ids), max_len), tokenizer.pad_token_id, dtype=torch.long, device=device)
            attention_mask = torch.zeros_like(input_ids)
            for i, ids in enumerate(all_combined_ids):
                seq_len = len(ids)
                input_ids[i, -seq_len:] = ids
                attention_mask[i, -seq_len:] = 1
        else:
            # 纯文本输入
            inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(device)
            input_ids = inputs.input_ids
            attention_mask = inputs.attention_mask

        # --- 2. CFG 输入准备 ---
        if use_cfg:
            # 对于CFG，我们需要有条件和无条件两组输入。
            # 核心是保证它们的序列长度一致，以便在batch维度上拼接。
            batch_size = input_ids.shape[0]
            uncond_prompts = [unconditional_prompt] * batch_size

            uncond_inputs = tokenizer(uncond_prompts, return_tensors="pt", padding=True).to(device)
            unconditional_input_ids = uncond_inputs.input_ids
            unconditional_attention_mask = uncond_inputs.attention_mask

            # 手动对齐序列长度 (左填充或截断)，以匹配有条件输入的长度。
            cond_len = input_ids.shape[1]
            uncond_len = unconditional_input_ids.shape[1]
            if cond_len > uncond_len:
                pad_len = cond_len - uncond_len
                padding = torch.full((batch_size, pad_len), tokenizer.pad_token_id, dtype=torch.long, device=device)
                unconditional_input_ids = torch.cat([padding, unconditional_input_ids], dim=1)
                unconditional_attention_mask = torch.cat([torch.zeros_like(padding), unconditional_attention_mask], dim=1)
            elif uncond_len > cond_len:
                main_logger.warning(f"不合理，推理时出现了 uncond_len > cond_len")
                unconditional_input_ids = unconditional_input_ids[:, -cond_len:]
                unconditional_attention_mask = unconditional_attention_mask[:, -cond_len:]

            # 将有条件和无条件批次在第0维度（batch维度）上拼接起来
            model_input_ids = torch.cat([input_ids, unconditional_input_ids], dim=0)
            model_attention_mask = torch.cat([attention_mask, unconditional_attention_mask], dim=0)
        else:
            # 不使用CFG时，输入就是原始输入
            model_input_ids = input_ids
            model_attention_mask = attention_mask

        batch_size = input_ids.shape[0]

        # 设置结束符ID
        if eos_token_id is None:
            eos_token_id = tokenizer.eos_token_id
        if isinstance(eos_token_id, int):
            eos_token_id = [eos_token_id]  # 统一处理为列表

        # --- 3. 设置 Logits Processors ---
        # LogitsProcessorList 用于管理一个或多个处理器
        processors = LogitsProcessorList()

        # 根据 acceptable_tokens 参数添加约束处理器
        # assert acceptable_tokens is not None, "to debug"
        if acceptable_tokens is not None:
            acceptable_tokens = [tokenizer(_, add_special_tokens=False).input_ids[0] for _ in acceptable_tokens.split(',')]
            # acceptable_tokens = tokenizer(acceptable_tokens, add_special_tokens=False).input_ids
            acceptable_mask = get_tensor_of_acceptable_tokens(acceptable_tokens, model.vocab_size).to(device)
            processors.append(SimpleTokenConstrainedLogitsProcessor(acceptable_mask))

        # 添加我们自定义的模态约束处理器
        processors.append(ModalLogitsProcessor(ori_vocab_size, modal_to_generate, vis_sep_token, vis_sep_len))
        # 根据采样参数，添加Hugging Face提供的标准处理器
        if do_sample:
            if temperature > 0 and temperature != 1.0:
                from transformers.generation.logits_process import TemperatureLogitsWarper
                processors.append(TemperatureLogitsWarper(temperature))
            if top_k > 0:
                from transformers.generation.logits_process import TopKLogitsWarper
                processors.append(TopKLogitsWarper(top_k))
            if top_p < 1.0:
                from transformers.generation.logits_process import TopPLogitsWarper
                processors.append(TopPLogitsWarper(top_p))
            if min_new_tokens > 0:
                from transformers.generation.logits_process import MinLengthLogitsProcessor
                assert eos_token_id is not None
                processors.append(MinLengthLogitsProcessor(min_new_tokens + input_ids.shape[-1], eos_token_id, device))

        unfinished_sequences = torch.ones(batch_size, dtype=torch.long, device=device)
        # --- 4. 核心自回归解码循环 (Manual Autoregressive Loop) ---
        with torch.no_grad():  # 推理时关闭梯度计算，节省显存和计算资源
            # 第一次模型调用：传入完整输入，计算并缓存所有的键/值（Key/Value Cache）
            outputs = model(input_ids=model_input_ids, attention_mask=model_attention_mask, use_cache=True)
            past_key_values = outputs.past_key_values

            # 初始化生成的序列，初始内容为输入本身
            generated_tokens = model_input_ids

            # 循环生成新的token，直到达到最大长度
            for step in range(max_new_tokens):
                # 从模型输出中获取最后一个时间步的logits
                logits = outputs.logits[:, -1, :]

                # 如果启用CFG，根据公式调整logits
                if use_cfg:
                    # 将合并的logits分割回有条件和无条件两部分
                    cond_logits, uncond_logits = torch.split(logits, batch_size, dim=0)
                    # CFG公式: guidance_scale * (conditional - unconditional) + unconditional
                    logits = uncond_logits + (cond_logits - uncond_logits) * cfg_scale

                # 应用所有处理器（模态约束、采样等）来修改logits
                processed_logits = processors(generated_tokens, logits)

                # 从修改后的logits中选择下一个token
                if do_sample:
                    # 采样：根据概率分布随机选择
                    probs = F.softmax(processed_logits, dim=-1)
                    next_tokens = torch.multinomial(probs, num_samples=1)
                else:
                    # 贪心解码：直接选择概率最高的token
                    next_tokens = torch.argmax(processed_logits, dim=-1).unsqueeze(-1)

                # 如果使用了CFG，需要为无条件部分也准备一个相同的next_tokens，以保持张量形状一致
                if use_cfg:
                    next_input_ids = torch.cat([next_tokens, next_tokens], dim=0)
                else:
                    next_input_ids = next_tokens

                # 将新生成的token添加到有条件序列的末尾
                generated_tokens = torch.cat([generated_tokens[:batch_size], next_tokens], dim=1)

                # 检查是否所有序列都已生成结束符
                if eos_token_id is not None:
                    # an elegant way to handle both single and multiple eos_token_id
                    produced_eos = torch.isin(next_tokens.squeeze(-1), torch.tensor(eos_token_id, device=device))
                    unfinished_sequences = unfinished_sequences & ~produced_eos
                    # 如果所有序列都已结束，则提前终止循环
                    if unfinished_sequences.max() == 0:
                        break

                # 准备下一次迭代的输入，这时输入只有一个token，但会利用之前缓存的past_key_values
                model_attention_mask = torch.cat([model_attention_mask, torch.ones_like(next_input_ids[:, -1:])], dim=-1)
                outputs = model(input_ids=next_input_ids, attention_mask=model_attention_mask, past_key_values=past_key_values, use_cache=True)
                past_key_values = outputs.past_key_values

        # 返回最终生成的完整序列（仅有条件部分）
        final_outputs = generated_tokens[:batch_size]
        if decode_in_the_end:
            if modal_to_generate == "text":
                # 仅解码新生成的部分
                output_tokens = generated_tokens[:, input_ids.shape[1]:]
                generated_tokens_list = output_tokens.tolist()
                eos_token_id_set = set(eos_token_id)

                decoded_texts = []
                for tokens in generated_tokens_list:
                    # 寻找第一个eos token的位置
                    stop_index = -1
                    for i, token_id in enumerate(tokens):
                        if token_id in eos_token_id_set:
                            stop_index = i
                            break
                    # 如果找到了eos，就截断
                    if stop_index != -1:
                        tokens = tokens[:stop_index]
                    decoded_texts.append(tokenizer.decode(tokens, skip_special_tokens=True))
                final_outputs = decoded_texts
            else:  # modal_to_generate == 'image'
                # 提取新生成的图像 token
                output_tokens = final_outputs[:, input_ids.shape[1]:]

                decoded_images = []
                for tokens in output_tokens:
                    # 确保生成的 token 数量符合预期，以构成完整的图像
                    assert len(tokens) == max_new_tokens, f"期望 {max_new_tokens} 个图像 token，但实际生成了 {len(tokens)} 个。"
                    tokens = tokens - ori_vocab_size
                    assert tokens.min() >= 0 and tokens.max() < 8192
                    # 使用图像分词器将 token 解码为 PIL 图像
                    pil_image = img_tokenizer.pil_from_img_toks(tokens)
                    decoded_images.append(pil_image)
                final_outputs = decoded_images

        return final_outputs

    return generate


def _test(
        model_path: str = "../models/Qwen2.5-1.5B-AddTokens",
        model_cls: str = "uni_x",
        vqgan_path: str = "data_process/vqgan/",
        max_new_tokens: int = 128,
        cfg_scale: float = 4.0,
        test_img_path: str = "test_samples/test.jpg",
        vqa_test_prompt: str = 'The caption of this image is:',
):
    """
    一个简单的测试函数，用于验证 any_modal_chat_api 的核心功能。
    它会执行两个核心任务：
    1. 图生文（Image Captioning）: 输入一张图片和文本提示，生成图片的描述。
    2. 文生图（Text-to-Image）: 输入一段描述性文本，生成对应的图片。
    """
    # --- 0. 环境设置 ---
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"当前使用的设备: {device}")

    # --- 1. 加载模型与分词器 ---
    print("正在加载模型和分词器...")
    # 加载大语言模型 (LLM) 和其文本分词器
    if model_cls == "auto":
        _cls = AutoModelForCausalLM
    elif model_cls == "uni_x":
        _cls = UniQwen3ForCausalLMInference
    else:
        raise ValueError("error model_cls")

    model = _cls.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    # 加载 VQGAN 图像分词器
    img_tokenizer = ImageTokenizer(ckpt_path=os.path.join(vqgan_path, 'vqgan.ckpt'), cfg_path=os.path.join(vqgan_path, 'vqgan.yaml'), device=device)
    # 确保 pad token 已设置，这对于左填充至关重要
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # --- 2. 创建 API 生成函数 ---
    ori_vocab_size = len(tokenizer)
    print(f"原始文本词汇表大小: {ori_vocab_size}")

    # 使用工厂函数创建一个配置好的 `generate` 闭包函数
    generate_fn = any_modal_chat_api(
        model=model,
        tokenizer=tokenizer,
        img_tokenizer=img_tokenizer,
        ori_vocab_size=ori_vocab_size,
    )

    # --- 3. 测试任务一：图生文 (Image Captioning) ---
    print("\n---【测试 1: 图生文】---")
    # 准备输入图片，如果找不到测试图片则创建一个纯色图片代替
    try:
        image = Image.open(test_img_path).convert("RGB")
        print(f"成功从路径加载测试图片: {test_img_path}")
    except FileNotFoundError:
        print(f"未在 '{test_img_path}' 找到测试图片，将创建一个红色256x256的图片用于测试。")
        image = Image.new('RGB', (256, 256), color='red')

    # 将图片编码为 base64 字符串
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")
    img_b64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

    # 准备图生文的 prompt
    caption_prompt = f"<|vision_start|><img><|vision_end|> {vqa_test_prompt}"
    # 调用生成函数
    generated_texts = generate_fn(
        prompts=[caption_prompt, caption_prompt, caption_prompt],  # 测试bs>1
        imgs_base64=[[img_b64], [img_b64], [img_b64]],
        modal_to_generate="text",
        max_new_tokens=max_new_tokens,
        cfg_scale=cfg_scale,
    )
    print(f"输入 Prompt: {caption_prompt}")
    print(f"模型生成描述: {generated_texts}")

    # --- 4. 测试任务二：文生图 (Text-to-Image) ---
    print("\n---【测试 2: 文生图】---")
    # 准备文生图的 prompt
    text_prompt = "一名面包师正在面包店里把刚烤好的面包从烤箱中取出来"
    gen_prompt = " Generate an image based on this description. <|vision_start|>"
    text_to_image_prompt = text_prompt + gen_prompt

    # 对于图像生成，max_new_tokens 必须是图像分词器产生的 token 精确数量
    # 例如，对于一个 256x256 的图像和一个 16x 降采样的 VQGAN，token 数量是 (256/16)^2 = 256
    num_image_tokens = 1024

    # 调用生成函数
    generated_images = generate_fn(
        prompts=[text_to_image_prompt],
        modal_to_generate="image",
        max_new_tokens=num_image_tokens,  # 必须是图像 token 的确切数量
        cfg_scale=cfg_scale
    )

    # 保存生成的图片
    if generated_images:
        output_image = generated_images[0]
        output_path = "test_samples/generated_image_by_model.png"
        output_image.save(output_path)
        print(f"输入 Prompt: {text_to_image_prompt}")
        print(f"图片生成成功，已保存至: {output_path}")
    else:
        print("图片生成失败。")

    print("\n--- 所有测试已完成 ---")


# --- 7. 示例用法 ---
if __name__ == '__main__':
    fire.Fire(_test)