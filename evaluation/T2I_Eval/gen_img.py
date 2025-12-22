import torch
from torch.cuda.nvtx import range_start
from transformers import AutoTokenizer, AutoModelForCausalLM, LogitsProcessor
from modeling.moe_qwen import HardMoeQwen3CausalLM
from modeling.mot_qwen import MoTQwen3ForCausalLM
from modeling.uni_x_qwen3 import UniQwen3ForCausalLMInference
from modeling import shared_func_module
import os
from tqdm import tqdm, trange
import argparse
from torch.nn import functional as F

# 从自定义模块中导入 ImageTokenizer，它负责将图像编码/解码为离散的词元
from data_process.vqgan.image_tokenizer import ImageTokenizer
import numpy as np
from tools.log import main_logger
from PIL import Image  # 导入PIL库用于图像处理

# 全局变量，用于存储原始文本词汇表的大小
Ori_Text_Token_Num = None
MAX_RETRIES = 5


def split_list(input_list, chunk_size):
    """一个辅助函数，将一个长列表切分成多个指定大小的小列表"""
    return [input_list[i:i + chunk_size] for i in range(0, len(input_list), chunk_size)]


def create_image_grid(images, rows, cols):
    """将一组PIL图像拼接成一个网格图像"""
    # 所有图像尺寸相同
    w, h = images[0].size
    # 创建一个新的空白图像用于存放网格
    grid = Image.new('RGB', size=(cols * w, rows * h))
    # 将每张小图依次粘贴到网格的正确位置
    for i, img in enumerate(images):
        grid.paste(img, box=(i % cols * w, i // cols * h))
    return grid


def main(args):
    # --- 1. 初始化和设置 ---
    LLM_pth = args.model_path
    text_set_id = args.chunk_idx
    cfg_scale = args.cfg_scale
    tau = args.tau
    topk = args.topk
    topp = args.topp
    num_chunks = args.num_chunks
    batch_size = args.batch_size
    image_save_pth = '{}/'.format(args.save_path)
    local_rank = os.environ.get("CUDA_VISIBLE_DEVICES", -1)
    # assert int(text_set_id) == int(local_rank)

    # --- 2. 加载模型和分词器 ---
    # 加载文本分词器，padding_side='left' 是为了让Causal LM的生成更方便
    tokenizer = AutoTokenizer.from_pretrained(LLM_pth, padding_side='left')
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 加载多模态大语言模型（VQ-LLM）
    if args.model_cls == "auto":
        model_cls = AutoModelForCausalLM
    elif args.model_cls == "uni_qwen":
        model_cls = UniQwen3ForCausalLMInference
    elif args.model_cls == "hard_moe":
        model_cls = HardMoeQwen3CausalLM
    elif args.model_cls == "mot":
        model_cls = MoTQwen3ForCausalLM
    else:
        raise ValueError("model cls")
    for i in range(MAX_RETRIES):
        try:
            model = model_cls.from_pretrained(
                LLM_pth,
                torch_dtype=torch.bfloat16,  # 使用 bfloat16 减少显存占用并加速
                device_map="auto",
            )
            break
        except Exception as e:
            if i == MAX_RETRIES - 1:
                raise e

    print("[debug] ", model.config)

    vis_sep_token = -1
    if getattr(model.config, "add_sep_for_vis", False):
        vis_sep_tokens_id = torch.tensor([tokenizer(_, add_special_tokens=False).input_ids[0] for _ in model.config.vis_sep_tokens.split(',')])
        shared_func_module.get_vision_mask = shared_func_module.reset_get_vis_mask(vis_sep_tokens_id)
        assert vis_sep_tokens_id.numel() == 1 and vis_sep_tokens_id.dim() == 1
        vis_sep_token = vis_sep_tokens_id[0].item()
        vis_sep_len = int(model.config.vis_sep_lens)

    # 定义VQGAN（图像分词器）的配置文件和权重路径
    vqgan_cfg_path = "data_process/vqgan/vqgan.yaml"
    vqgan_ckpt_path = "data_process/vqgan/vqgan.ckpt"
    # 初始化图像分词器（解码器）
    image_tokenizer = ImageTokenizer(cfg_path=vqgan_cfg_path, ckpt_path=vqgan_ckpt_path, device="cuda", )
    # 记录纯文本词汇表的大小，这是区分文本和图像词元的关键
    global Ori_Text_Token_Num
    Ori_Text_Token_Num = ori_vocab_size = len(tokenizer)
    # total_vocab_size = ori_vocab_size + 8192
    # vocab_ids = torch.arange(0, total_vocab_size, device=model.device)
    # vis_vocab_mask = shared_func_module.get_vision_mask(vocab_ids, ori_vocab_size, model.config.vision_start_id)
    # text_vocab_mask = ~vis_vocab_mask

    # --- 3. 准备输入数据 ---
    with open(args.prompts_path, 'r') as f:
        lines = f.readlines()

    # 并行化处理：将所有提示分成 N 块，当前任务只处理其中一块
    chunked_lines = np.array_split(lines, num_chunks)
    subset_lines = chunked_lines[text_set_id].tolist()
    line_num_bf_this_chunk = sum([len(chunk) for _, chunk in enumerate(chunked_lines) if _ < text_set_id])

    all_prompts = []
    if args.prompts_path.endswith('txt'):
        # 遍历所有原始提示
        for index, line in enumerate(subset_lines):
            # 根据 num_gen_repeats 参数，为每个提示添加一个或多个生成任务
            index = index + line_num_bf_this_chunk
            for i in range(args.num_gen_repeats):
                # 如果只生成一次，保持原始索引；否则，添加后缀以创建唯一索引
                img_index = str(index) if args.num_gen_repeats == 1 else f"{index}_{i}"
                all_prompts.append({'Index': img_index, 'Prompt': line.strip()})
    elif args.prompts_path.endswith('jsonl'):
        import json
        for index, line in enumerate(subset_lines):
            # 根据 num_gen_repeats 参数，为每个提示添加一个或多个生成任务
            l_dic = json.loads(line)
            for i in range(args.num_gen_repeats):
                # 如果只生成一次，保持原始索引；否则，添加后缀以创建唯一索引
                l_index = l_dic['Index']
                img_index = l_index if args.num_gen_repeats == 1 else f"{l_index}_{i}"
                all_prompts.append({'Index': img_index, 'Prompt': l_dic['Prompt'].strip()})
    else:
        raise ValueError("prompt文件格式不正确")

    # 将当前任务块再切分成更小的批次
    if args.grid_img:
        assert batch_size % 8 == 0, 'ensure grid img ok'
    chunk_inputs = split_list(all_prompts, batch_size)

    # --- 4. 核心生成循环 ---
    for chunk in tqdm(chunk_inputs, desc=f"<local rank: {local_rank}>"):  # 使用tqdm显示处理进度

        # 准备有条件（conditional）和无条件（unconditional）的输入，用于CFG
        text_inputs = [v['Prompt'] for v in chunk]
        uncondition_text_inputs = ['<unconditional><|vision_start|>'] * len(text_inputs)
        # 为每个文本提示添加后缀，引导模型开始生成图像
        for i in range(len(text_inputs)):
            text_inputs[i] = text_inputs[i] + ' Generate an image based on this description.<|vision_start|>'

        if cfg_scale > 1.0:
            # 如果使用CFG，将有条件和无条件的输入一起编码
            # 模型内部会利用这个结构进行引导式生成
            model_inputs = tokenizer(text_inputs + uncondition_text_inputs, return_tensors="pt", padding=True).to('cuda')
        else:
            # 否则只编码有条件的输入
            model_inputs = tokenizer(text_inputs, return_tensors="pt", padding=True).to('cuda')

        with torch.no_grad():  # 在推理时关闭梯度计算，节省资源
            # --- 代码开始：手动实现支持 CFG 的自回归解码 ---
            input_ids = model_inputs['input_ids']
            attention_mask = model_inputs['attention_mask']
            cur_len = input_ids.shape[1]
            max_new_tokens = 1024  # 定义生成图像词元的数量

            # 定义一个 LogitsProcessor，强制模型只生成图像词元 (ID >= Ori_Text_Token_Num)
            class ImageTokenLogitsProcessor(LogitsProcessor):
                def __init__(self, min_token_id: int):
                    self.min_token_id = min_token_id
                    self.cnt = 0

                def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
                    self.cnt += 1
                    main_logger.debug(f"{self.cnt}:: vis sep token {scores[:, vis_sep_token].tolist()} ## mean score {scores.mean()} ## max score {scores.max()}")
                    if vis_sep_token != -1 and self.cnt % (vis_sep_len + 1) == 0:
                        scores[:] = -float('inf')
                        scores[:, vis_sep_token] = 0.0
                    else:
                        scores[:, :self.min_token_id] = -float('inf')
                    return scores

            image_processor = ImageTokenLogitsProcessor(ori_vocab_size)

            # 1. 首次前向传播，获得初始的 KV Cache
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=True)
            past_key_values = outputs.past_key_values
            generated_tokens = []

            # 2. 自回归生成每个图像词元
            pbar = tqdm(total=max_new_tokens, desc=f'GPU{local_rank} 产生img tokens ... ')
            idx = 0
            while idx < max_new_tokens:
                pbar.update(1)
                idx += 1

                # 获取最后一步的 logits
                logits = outputs.logits[:, -1, :]

                # 如果启用 CFG，则计算引导后的 logits
                if cfg_scale > 1.0:
                    cond_logits, uncond_logits = torch.split(logits, len(chunk), dim=0)
                    logits = uncond_logits + (cond_logits - uncond_logits) * cfg_scale

                # 强制只生成图像词元
                logits = image_processor(None, logits)

                # 应用采样策略 (Temperature, Top-K, Top-P)
                if tau > 0: logits = logits / tau
                if topk > 0:
                    v, _ = torch.topk(logits, min(topk, logits.size(-1)))
                    logits[logits < v[:, -1:]] = -float('inf')
                if topp < 1.0:
                    probs = F.softmax(logits, dim=-1)
                    sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
                    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                    sorted_indices_to_remove = cumulative_probs > topp
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0  # bs, vocab_sz
                    # Scatter to original indices
                    indices_to_remove = torch.zeros_like(logits, dtype=torch.bool)
                    indices_to_remove.scatter_(
                        dim=1,
                        index=sorted_indices,
                        src=sorted_indices_to_remove
                    )
                    logits = logits.masked_fill(indices_to_remove, float('-inf'))

                # 从处理后的 logits 中采样
                probs = F.softmax(logits, dim=-1)
                next_tokens = torch.multinomial(probs, num_samples=1)

                is_vis_sep = (next_tokens == vis_sep_token).sum().item()
                assert is_vis_sep == next_tokens.numel() or is_vis_sep == 0
                if is_vis_sep == 0:
                    generated_tokens.append(next_tokens)
                else:
                    max_new_tokens += 1
                    main_logger.debug(f"at {idx}, max_new_tokens += 1")

                # 准备下一次迭代的输入 (利用 KV Cache)
                if cfg_scale > 1.0:
                    # 对于 CFG，为有条件和无条件两路复制相同的输入
                    next_input_ids = torch.cat([next_tokens, next_tokens], dim=0)
                else:
                    next_input_ids = next_tokens

                attention_mask = torch.cat([attention_mask, torch.ones_like(attention_mask[:, -1:])], dim=-1)
                outputs = model(input_ids=next_input_ids, attention_mask=attention_mask, past_key_values=past_key_values, use_cache=True)
                past_key_values = outputs.past_key_values

            main_logger.debug(f"final max_new_tokens {max_new_tokens}")
            # 将所有生成的词元拼接成一个张量
            final_tokens = torch.cat(generated_tokens, dim=1)
            # --- 代码完成 ---

        # --- 5. 图像词元后处理 ---
        image_vq_id = final_tokens
        # 关键步骤：将全局词元ID转换为图像词元ID
        # 通过减去文本词汇表的大小，得到在图像词汇表中的相对ID
        image_vq_id = image_vq_id - ori_vocab_size
        # 确保ID在有效范围内（VQGAN的码本大小通常是8192，即0-8191）
        image_vq_id = torch.clamp(image_vq_id, min=0, max=8191)

        image_list = []
        for index, generate_id in enumerate(image_vq_id):
            image_list.append(generate_id.tolist())

        # --- 6. 解码图像并保存 ---
        if not os.path.exists(image_save_pth):
            os.makedirs(image_save_pth)  # 如果保存目录不存在，则创建

        # 检查是否启用网格图像保存
        grid_img_enabled = args.grid_img and args.num_gen_repeats > 1
        grid_size = 0
        if grid_img_enabled:
            grid_size = int(np.sqrt(args.num_gen_repeats))
            if grid_size * grid_size != args.num_gen_repeats:
                main_logger.warning("num_gen_repeats 不是完全平方数，已禁用网格图像保存功能。")
                grid_img_enabled = False

        if grid_img_enabled:
            # 按 num_gen_repeats 分组处理图像
            assert len(image_list) % args.num_gen_repeats == 0
            for i in range(0, len(image_list), args.num_gen_repeats):
                image_codes_group = image_list[i:i + args.num_gen_repeats]
                datainfo_group = chunk[i:i + args.num_gen_repeats]
                # 获取组内第一个元素的基本索引作为参照 获取原始索引 (a_b_0 --> a_b)
                base_index = "_".join(datainfo_group[0]['Index'].split('_')[:-1])

                # 遍历组内所有元素，确保它们的基本索引都与参照相同
                for item_info in datainfo_group:
                    item_base_index = "_".join(item_info['Index'].split('_')[:-1])
                    assert item_base_index == base_index, \
                        f"A group for grid image contains items from different original prompts. Expected base index '{base_index}', but found item with index '{item_info['Index']}'."

                pil_images = []
                for vq_code in image_codes_group:
                    latents = torch.tensor(vq_code).to('cuda')
                    rec_img = image_tokenizer.pil_from_img_toks(latents)
                    pil_images.append(rec_img)

                # 创建并保存网格图
                grid = create_image_grid(pil_images, grid_size, grid_size)
                grid.save(f'{image_save_pth}/{base_index}.png')
        else:
            # 原始逻辑：逐个保存图像
            for datainfo, vq_code in zip(chunk, image_list):
                idx = datainfo['Index']  # 获取图片的文件名索引
                main_logger.debug(f"{idx}::: {vq_code}")
                latents = torch.tensor(vq_code).to('cuda')
                # 使用图像解码器将离散的词元序列转换回PIL图像对象
                rec_img = image_tokenizer.pil_from_img_toks(latents)
                # 保存图像
                rec_img.save('{}/{}.png'.format(image_save_pth, str(idx)))


def get_args_parser():
    """定义和解析命令行参数"""
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
    # --- 模型与路径参数 ---
    parser.add_argument('--model_path', type=str, default='/path/to/uni_arch_models//uni_arch_V1_7B/', help="预训练模型文件路径")
    parser.add_argument('--model_cls', type=str, default='auto')
    parser.add_argument('--save_path', type=str, default='GenAI_Bench_527_results', help="生成图片保存的文件夹")
    parser.add_argument('--prompts_path', type=str, default='evaluation/T2I_Eval/prompts.txt')
    # --- 硬件与性能参数 ---
    parser.add_argument('--batch_size', type=int, default=4, help="每批处理的提示数量")
    # --- 并行处理参数 ---
    parser.add_argument('--chunk_idx', type=int, default=0, help="当前处理的数据块索引（用于并行化）")
    parser.add_argument('--num_chunks', type=int, default=8, help="总共要切分成的数据块数量（用于并行化）")
    # --- 生成策略参数 ---
    parser.add_argument('--cfg_scale', type=float, default=7.0, help="Classifier-Free Guidance (CFG) 的尺度因子，控制图像与文本的关联度")
    parser.add_argument('--tau', type=float, default=0.99, help="采样时的温度 (temperature) 参数，控制随机性")
    parser.add_argument('--topk', type=int, default=4096, help="Top-K 采样参数")
    parser.add_argument('--topp', type=float, default=0.96, help="Top-P (nucleus) 采样参数")
    # num_gen_repeats 参数，用于指定每个提示的重复生成次数
    parser.add_argument('--num_gen_repeats', type=int, default=1, help="每个提示重复生成的次数")
    parser.add_argument('--grid_img', type=int, default=0, help='当num_gen_repeats为平方数时，把多张图片以正方形网格形式拼成一张保存')

    return parser


if __name__ == '__main__':
    # 主程序入口
    # 创建参数解析器
    parser = argparse.ArgumentParser('image path check script', parents=[get_args_parser()])
    # 解析命令行参数
    args = parser.parse_args()
    # 调用主函数
    main(args)