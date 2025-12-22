import os
import json
import argparse
import io
import base64

import torch
import numpy as np
import PIL
from PIL import Image, ImageFile
from tqdm import tqdm
from torch.utils.data import IterableDataset, DataLoader
from torchvision import transforms

from vqgan.image_tokenizer import ImageTokenizer

# --- 全局设置 ---
ImageFile.LOAD_TRUNCATED_IMAGES = True
torch.set_grad_enabled(False)


def _whiten_transparency(img: PIL.Image.Image) -> PIL.Image.Image:
    """如果图像有透明通道，将其与白色背景融合。"""
    if img.mode == "RGB":
        return img

    vals_rgba = np.array(img.convert("RGBA"))

    # 如果没有实际的透明像素，直接转换为 RGB
    if not (vals_rgba[:, :, 3] < 255).any():
        return img.convert("RGB")

    # 使用 Alpha 通道与白色背景进行混合
    alpha = vals_rgba[:, :, 3] / 255.0
    vals_rgb = (1 - alpha[:, :, np.newaxis]) * 255 + alpha[:, :, np.newaxis] * vals_rgba[:, :, :3]
    return PIL.Image.fromarray(vals_rgb.astype("uint8"), "RGB")


def center_crop_image(ori_image, tgt_width=512, tgt_height=512):
    """
    按比例缩放后进行中心裁剪，以获得目标尺寸的图像。
    """
    Width, Height = ori_image.size
    factor = min(Width, Height) / min(tgt_width, tgt_height)
    input_image = ori_image.resize((int(Width / factor), int(Height / factor)), PIL.Image.LANCZOS)
    resize_width, resize_height = input_image.size

    left = (resize_width - tgt_width) // 2
    top = (resize_height - tgt_height) // 2
    right = (resize_width + tgt_width) // 2
    bottom = (resize_height + tgt_height) // 2
    input_image = input_image.crop((left, top, right, bottom))
    return input_image


# --- 流式数据集 ---
class ImageTextDataset(IterableDataset):
    """
    一个使用 yield 实现流式读取的图文数据集。
    在 __iter__ 方法中逐行读取文件并 yield 数据，支持多进程加载。
    """

    def __init__(self, file_paths, image_size=512):
        super().__init__()
        self.file_paths = file_paths
        self.image_size = image_size

        # 预处理管道，确保输出的 Tensor 数值范围为 [-1, 1]
        self.transform = transforms.Compose([
            transforms.ToTensor(),  # PIL [0, 255] -> Tensor [0, 1]
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Tensor [0, 1] -> Tensor [-1, 1]
        ])

    def _process_line(self, line):
        """处理单行数据，从 base64 解码图像并应用所有预处理。"""
        try:
            fields = line.strip().split('\t')
            caption_json_str, base64_str = fields[-6], fields[-3]

            caption_info = json.loads(caption_json_str)
            text = ""
            for k, v in caption_info.items():
                if len(v) > len(text):
                    text = v

            image_bytes = base64.b64decode(base64_str)

            # 1. 从字节流打开图像
            raw_image = PIL.Image.open(io.BytesIO(image_bytes))

            # 2. 处理透明度
            ori_image = _whiten_transparency(raw_image)

            # 3. 过滤掉比例过大的图像
            width, height = ori_image.size
            if max(width, height) / min(width, height) > 2.5:
                return None

            # 4. 中心裁剪
            processed_image = center_crop_image(ori_image, self.image_size, self.image_size)

            # 5. 校验图像尺寸，确保处理后的图像大小正确
            assert processed_image.size == (self.image_size, self.image_size), \
                f"Image size is {processed_image.size} after cropping, expected ({self.image_size}, {self.image_size})."

            # 6. 应用 transform，生成最终的 Tensor
            return self.transform(processed_image), text
        except Exception:
            # 捕获任何潜在的错误（如损坏的数据行）
            return None

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()

        if worker_info is None:
            # 单进程模式：处理所有分配的文件
            files_to_process = self.file_paths
        else:
            # 多进程模式：每个 worker 处理一部分文件
            worker_id = worker_info.id
            num_workers = worker_info.num_workers
            files_to_process = self.file_paths[worker_id::num_workers]

        for file_path in files_to_process:
            with open(file_path, 'r', encoding="utf-8") as f:
                for line in f:
                    processed_item = self._process_line(line)
                    if processed_item is not None:
                        yield processed_item


# --- 自定义 Collate Function ---
def collate_fn(batch):
    """过滤掉数据加载过程中产生的 None 样本，然后将有效样本打包。"""
    batch = [item for item in batch if item is not None]
    if not batch:
        return None, None
    images, texts = zip(*batch)
    image_batch = torch.stack(images, dim=0)
    return image_batch, list(texts)


# --- 主函数 ---
def main(args):
    # 初始化 VQGAN Tokenizer
    vqgan_cfg_path = os.path.join(args.vqgan_path, "vqgan.yaml")
    vqgan_ckpt_path = os.path.join(args.vqgan_path, "vqgan.ckpt")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    image_tokenizer = ImageTokenizer(cfg_path=vqgan_cfg_path, ckpt_path=vqgan_ckpt_path, device=device)
    num_tokens = 1024

    # 根据总分块数和当前块索引，确定本进程要处理的文件列表
    all_files = sorted([os.path.join(args.input_dir, f) for f in os.listdir(args.input_dir) if f.startswith('part-')])
    chunked_files = np.array_split(all_files, args.num_chunks)
    files_to_process = chunked_files[args.chunk_idx].tolist()

    if not files_to_process:
        print(f"Rank {args.chunk_idx} has no files to process. Exiting.")
        return

    print(
        f"Rank {args.chunk_idx} processing {len(files_to_process)} files (showing first 3): {files_to_process[:3]}...")

    # 实例化数据集和数据加载器
    dataset = ImageTextDataset(files_to_process, image_size=512)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_processes,
        collate_fn=collate_fn,
        pin_memory=True  # 如果在 GPU 上运行，推荐开启
    )

    valid_pair_list = []
    # 对于 IterableDataset，tqdm 无法预知总长度，所以不会显示进度条和剩余时间
    pbar = tqdm(dataloader, desc=f'Rank {args.chunk_idx} Processing')

    for image_batch, text_batch in pbar:
        if image_batch is None:
            continue

        # 获取当前批次的实际大小，以正确处理最后一个不满的批次
        current_batch_size = image_batch.shape[0]

        # 将预处理好的图像批次移动到目标设备
        image_batch = image_batch.to(device=device, dtype=image_tokenizer._dtype)

        with torch.no_grad():
            # 对整个 batch 进行编码
            _, _, [_, _, vqcode_batch] = image_tokenizer._vq_model.encode(image_batch)

            # 使用当前批次的实际大小进行 reshape
            vqcode_batch = vqcode_batch.view(current_batch_size, -1)
            vqcode_batch = vqcode_batch.cpu().tolist()

        # 整理结果
        for text, vqcode in zip(text_batch, vqcode_batch):
            # 校验 token 数量，确保数据处理流程无误
            assert len(vqcode) == num_tokens, f"Incorrect token count: got {len(vqcode)}, expected {num_tokens}."
            new_anno = {
                'data_type': 'image_text',
                'text': text,
                'length': len(text) + len(vqcode) * 4,
                'vqcode_512': json.dumps(vqcode),
                'vqcode_multi768': 'no',
                'width': 'no',
                'height': 'no',
            }
            valid_pair_list.append(new_anno)

        if args.debug:
            break

    # 结果统计和保存
    valid_images = len(valid_pair_list)
    print(f'\nFinished processing. Kept {valid_images} valid image-text pairs in rank {args.chunk_idx}.')

    os.makedirs(args.temp_path, exist_ok=True)
    output_filename = os.path.join(args.temp_path, f'{args.chunk_idx:06d}.jsonl')
    with open(output_filename, 'w', encoding='utf-8') as f:
        for item in valid_pair_list:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    print(f"Results for rank {args.chunk_idx} saved to {output_filename}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Process image-text pairs from chunked files using DataLoader in batch mode.')
    parser.add_argument('--input_dir', type=str, required=True,
                        help='Directory containing input data files (e.g., part-xxxxx)')
    parser.add_argument('--temp_path', type=str, required=True, help='Path to save converted jsonl files')
    parser.add_argument('--vqgan_path', type=str, default='data_process/vqgan',
                        help='Path to the directory containing vqgan.yaml and vqgan.ckpt')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for GPU processing')
    parser.add_argument('--chunk_idx', type=int, default=0, help='Chunk id to process (e.g., from 0 to num_chunks-1)')
    parser.add_argument('--num_chunks', type=int, default=8,
                        help='Total number of chunks, should match the number of processes you run in parallel')
    parser.add_argument('--num_processes', type=int, default=8, help='Number of worker processes for DataLoader')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode to process only one batch')

    args = parser.parse_args()
    main(args)