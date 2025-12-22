SUBSET_NAMES = ['CoSyn_400k_chart', 'CoSyn_400k_chemical', 'CoSyn_400k_circuit', 'CoSyn_400k_diagram', 'CoSyn_400k_document',
                'CoSyn_400k_graphic', 'CoSyn_400k_math', 'CoSyn_400k_music', 'CoSyn_400k_nutrition', 'CoSyn_400k_table', 'DoclingMatix',
                'LLaVA_Instruct_150K', 'SynthChartNet', 'SynthCodeNet', 'SynthFormulaNet', 'Unichart', 'a_okvqa', 'aguvis-stage-1',
                'ai2d_merged', 'alfworldgpt', 'allava_laion', 'allava_vflan', 'aokvqa', 'art', 'arxivqa', 'bentham',
                'blockdiagramcomputerized', 'blockdiagramhandwritten', 'cambrian(filtered)_processed', 'captcha', 'chart2text',
                'chartqa', 'chinesememe', 'chrome_writting', 'clevr', 'clevr_math', 'clevr_math(mathv360k)', 'coco_colors',
                'cocoqa', 'cocotext', 'ctw', 'datik', 'datikz', 'densefusion_1m', 'diagram_image_to_text', 'docvqa',
                'drivelm', 'dvqa', 'est_vqa', 'face_emotion', 'figureqa', 'figureqa(mathv360k)', 'finqa', 'funsd',
                'geo170k(align)', 'geo170k(qa)', 'geo3k', 'geometry3k(mathv360k)', 'geomverse', 'geoqa+(mathv360k)',
                'geos(mathv360k)', 'google_landmarks', 'groundui', 'handwriting_forms', 'hateful_memes', 'hitab', 'hme100k',
                'hw_squad', 'iam', 'iconqa', 'iconqa(mathv360k)', 'idk', 'iiit5k', 'image_textualization(filtered)', 'imgur5k',
                'indoor_qa', 'infographic(gpt4v)', 'infographic_vqa', 'infographic_vqa_llava_format', 'intergps',
                'invoices_receipts', 'k12_printing', 'laion_gpt4v', 'latex_handwritten', 'latexformulas', 'llavar_gpt4_20k',
                'lnqa', 'localized_narratives', 'lrv_chart', 'lrv_normal(filtered)', 'lvis_instruct4v', 'mapqa',
                'mapqa(mathv360k)', 'maptext', 'mathwriting-google', 'mavis_math_metagen', 'mavis_math_rule_geo', 'memotion',
                'mimic_cgd', 'mmc_instruct', 'mmevol', 'mmra', 'mmsoc_memotion', 'multihiertt', 'nlvr2', 'objects365_qa', 'ocrvqa',
                'olmOCR-mix-0225-books', 'olmOCR-mix-0225-documents', 'oodvqa', 'orand_car_a', 'pathvqa', 'pdfvqa',
                'plotqa', 'pmc_vqa(mathv360k)', 'raven', 'rendered_text', 'robut_sqa', 'robut_wikisql', 'robut_wtq',
                'scienceqa', 'scienceqa(nona_context)', 'screen2words', 'screenqa', 'sharegpt4o', 'sharegpt4v(coco)',
                'sharegpt4v(knowledge)', 'sharegpt4v(llava)', 'sharegpt4v(sam)', 'sketchyvqa', 'slidevqa', 'spark', 'spatialsense',
                'spot_the_diff', 'sroie', 'st_vqa', 'sujet_finance', 'super_clevr(mathv360k)', 'svrd', 'synthdog', 'tabmwp',
                'tabmwp(mathv360k)', 'tal_ocr_eng', 'tallyqa', 'tat_dqa', 'tat_qa', 'tqa',
                'unigeo(mathv360k)', 'ureader_cap', 'ureader_ie', 'ureader_kg_processed',
                'ureader_qa_processed', 'vision_flan(filtered)', 'vistext', 'visual7w', 'visualmrc',
                'visualwebinstruct(filtered)', 'vizwiz(mathv360k)', 'vqaonbd', 'vqarad', 'vqav2', 'vsr', 'websight',
                'wildvision', 'wordart', 'yesbut']
ROOT_PATH = '../finevision'
TEMP_SAVE_PATH = '../mock/datasets/finevision_vq'

import os
import json
import argparse
import io
import glob
import shutil
import time
import threading
import queue

import torch
import numpy as np
import pandas as pd # 导入 pandas 用于读取 parquet 文件
import PIL
from PIL import Image, ImageFile
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
# from datasets import load_dataset # 改用 pandas，不再需要
from accelerate import Accelerator, DataLoaderConfiguration

from vqgan.image_tokenizer import ImageTokenizer

# --- 全局设置 ---
ImageFile.LOAD_TRUNCATED_IMAGES = True  # 允许加载截断的图像文件
torch.set_grad_enabled(False)  # 全局禁用梯度计算，节省显存和计算资源
# 初始化 Accelerator，用于简化分布式训练和混合精度
accelerator = Accelerator(dataloader_config=DataLoaderConfiguration(dispatch_batches=False))
MAX_RETRIES = 10  # 文件操作失败时的最大重试次数


def _whiten_transparency(img: PIL.Image.Image) -> PIL.Image.Image:
    """如果图像有透明通道(RGBA)，将其与白色背景融合，转换为RGB。"""
    if img.mode == "RGB":
        return img
    # 检查是否存在透明像素
    vals_rgba = np.array(img.convert("RGBA"))
    if not (vals_rgba[:, :, 3] < 255).any():
        return img.convert("RGB")

    # 使用alpha通道进行加权平均，实现与白色背景的融合
    alpha = vals_rgba[:, :, 3] / 255.0
    vals_rgb = (1 - alpha[:, :, np.newaxis]) * 255 + alpha[:, :, np.newaxis] * vals_rgba[:, :, :3]
    return PIL.Image.fromarray(vals_rgb.astype("uint8"), "RGB")


def center_crop_image(ori_image, tgt_width=512, tgt_height=512):
    """按比例缩放后进行中心裁剪，以获得目标尺寸的图像。"""
    Width, Height = ori_image.size
    # 计算缩放因子，保持图像的宽高比
    factor = min(Width, Height) / min(tgt_width, tgt_height)
    input_image = ori_image.resize((int(Width / factor), int(Height / factor)), PIL.Image.LANCZOS)

    # 计算中心裁剪的坐标
    resize_width, resize_height = input_image.size
    left = (resize_width - tgt_width) // 2
    top = (resize_height - tgt_height) // 2
    right = (resize_width + tgt_width) // 2
    bottom = (resize_height + tgt_height) // 2
    input_image = input_image.crop((left, top, right, bottom))
    return input_image


class FinevisionDataset(Dataset):
    """一个用于处理 FineVision 数据集的 Dataset，适配 Pandas DataFrame。"""

    def __init__(self, df: pd.DataFrame, image_size=512, device='cpu'):
        super().__init__()
        self.df = df # 直接使用 Pandas DataFrame 作为数据源
        self.image_size = image_size
        # 定义图像预处理流程：转为Tensor并归一化到[-1, 1]
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        self.device = device

    def _process_item(self, item):
        """处理单个数据项，包括图像预处理和文本拼接。"""
        try:
            # 从 DataFrame 的行中提取图像和文本信息
            image_info = item['images'][0]
            # 从bytes中读取图片
            image_bytes = image_info.get('bytes') if isinstance(image_info, dict) else None
            if not image_bytes: return None
            raw_image = Image.open(io.BytesIO(image_bytes))

            # 预处理图像
            ori_image = _whiten_transparency(raw_image)
            width, height = ori_image.size
            if max(width, height) / min(width, height) > 2.5: return None  # 过滤掉长宽比过大的图片
            processed_image = center_crop_image(ori_image, self.image_size, self.image_size)
            assert processed_image.size == (self.image_size, self.image_size)

            # 将多轮对话拼接成一个字符串
            turns = [f"{conv.get('user', '')}^^^pair_split^^^{conv.get('assistant', '')}" for conv in item['texts']]
            text = "^^^turn_split^^^".join(turns)

            return self.transform(processed_image), text
        except Exception as e:
            print(f"处理数据时发生错误: {e}")
            # 任何处理异常都返回None，由collate_fn过滤
            return None
            # raise

    def __len__(self):
        return len(self.df) # 返回 DataFrame 的长度

    def __getitem__(self, idx):
        # 通过 iloc 获取指定索引的行
        return self._process_item(self.df.iloc[idx])


def collate_fn(batch):
    """过滤掉数据加载过程中产生的 None 样本，然后将有效样本打包。"""
    batch = [item for item in batch if item is not None]
    if not batch: return None, None
    images, texts = zip(*batch)
    return torch.stack(images, dim=0), list(texts)


def copy_with_retry(src, dst, max_retries=MAX_RETRIES, delay=5):
    """带重试机制的文件复制函数，用于处理网络文件系统(NFS)的不稳定性。"""
    for i in range(max_retries):
        try:
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copy(src, dst)
            return True
        except Exception as e:
            print(f"复制失败 (尝试 {i + 1}/{max_retries}): {src} -> {dst}。错误: {e}")
            time.sleep(delay)
    print(f"放弃复制: {src}")
    return False


# --- 流水线线程函数 ---

def fetcher_thread(files_to_fetch, q, process_cache_dir):
    """[阶段1: 获取] 后台从NFS复制原始文件到本地缓存，以加速后续读取。"""
    for nfs_path in files_to_fetch:
        filename = os.path.basename(nfs_path)
        subset = os.path.basename(os.path.dirname(nfs_path))
        local_path = os.path.join(process_cache_dir, subset, filename)
        if copy_with_retry(nfs_path, local_path):
            q.put((nfs_path, local_path))  # 将（NFS路径，本地路径）元组放入队列
    q.put(None)  # 结束标志


def uploader_thread(q):
    """[阶段3: 上传] 后台将本地的处理结果复制回NFS，并清理本地文件。"""
    while True:
        item = q.get()
        if item is None: break  # 收到结束标志后退出
        local_result_path, final_nfs_path = item
        if copy_with_retry(local_result_path, final_nfs_path):
            os.remove(local_result_path)  # 上传成功后删除本地临时文件


# --- 主函数 ---
def main(args):
    """
    主处理流程：
    采用三阶段流水线模式（获取 -> 处理 -> 上传）来最大化效率。
    - Fetcher线程：负责从网络存储（NFS）下载数据到本地。
    - 主线程：利用GPU进行核心的数据处理（图像VQ编码）。
    - Uploader线程：负责将处理完的结果上传回网络存储。
    这种方式可以使得I/O操作和GPU计算并行进行，减少等待时间。
    """
    # 阶段 0: 初始化
    # 详细注释: 脚本开始运行时，首先进行环境和模型的初始化设置。
    # 运行状态描述: 打印任务开始信息，加载 VQGAN 模型，并显示所用的设备。
    print("--- 任务开始：初始化模型和环境 ---")
    vqgan_cfg_path = os.path.join(args.vqgan_path, "vqgan.yaml")
    vqgan_ckpt_path = os.path.join(args.vqgan_path, "vqgan.ckpt")
    device = accelerator.device
    image_tokenizer = ImageTokenizer(cfg_path=vqgan_cfg_path, ckpt_path=vqgan_ckpt_path, device=device)
    # num_tokens = 1024  # VQ编码后的token数量，未使用，可移除
    print(f"VQGAN模型已加载到设备: {device}")

    # 阶段 0: 创建临时目录
    # 详细注释: 为每个分布式进程创建独立的本地临时目录，用于暂存下载的原始数据和生成的中间结果。
    #           这样做可以避免多进程之间发生文件读写冲突。主进程会负责预先清理可能存在的旧目录。
    # 运行状态描述: 显示正在清理和创建临时目录。
    temp_cache_dir = './temp_parquet_cache'  # 存放从NFS下载的原始数据
    temp_save_dir = './temp_jsonl_output'  # 存放处理后的结果
    process_cache_dir = os.path.join(temp_cache_dir, f'process_{accelerator.process_index}')
    process_save_dir = os.path.join(temp_save_dir, f'process_{accelerator.process_index}')

    # 主进程负责清理旧的临时目录
    if accelerator.is_main_process:
        print("主进程正在清理旧的临时目录...")
        if os.path.exists(temp_cache_dir): shutil.rmtree(temp_cache_dir)
        if os.path.exists(temp_save_dir): shutil.rmtree(temp_save_dir)

    accelerator.wait_for_everyone()  # 等待所有进程同步，确保清理完成后再创建
    os.makedirs(process_cache_dir, exist_ok=True)
    os.makedirs(process_save_dir, exist_ok=True)
    print(f"进程 {accelerator.process_index} 的临时目录已创建。")

    # 阶段 0: 分配文件
    # 详细注释: 扫描所有待处理的数据子集，生成一个总的文件列表。然后根据当前进程的ID，
    #           从列表中均匀地、非重复地选取一部分文件进行处理，实现任务的静态分配。
    # 运行状态描述: 打印当前进程分配到的文件数量。
    all_parquet_files = sorted([p for s in SUBSET_NAMES for p in glob.glob(os.path.join(ROOT_PATH, s, '*.parquet'))])
    files_to_process = all_parquet_files[accelerator.process_index::accelerator.num_processes]
    print(f"进程 {accelerator.process_index} 分配到 {len(files_to_process)} 个文件。")

    # --- 初始化三阶段流水线 ---
    # 详细注释: 创建两个线程安全的队列，`fetch_queue` 用于存放已下载到本地等待处理的文件信息，
    #           `upload_queue` 用于存放已处理完等待上传到NFS的结果。同时启动后台的下载和上传线程。
    # 运行状态描述: 显示流水线队列和后台线程已启动。
    fetch_queue = queue.Queue(maxsize=20)  # 缓存20个待处理文件
    upload_queue = queue.Queue(maxsize=20)  # 缓存20个待上传结果
    print(f"进程 {accelerator.process_index}: 流水线队列已初始化。")

    # 启动获取线程和上传线程，它们将在后台独立运行
    fetcher = threading.Thread(target=fetcher_thread, args=(files_to_process, fetch_queue, process_cache_dir))
    uploader = threading.Thread(target=uploader_thread, args=(upload_queue,))
    fetcher.start()
    uploader.start()
    print(f"进程 {accelerator.process_index}: Fetcher和Uploader后台线程已启动。")

    # [阶段2: 处理] 主线程只负责核心的GPU计算任务
    # 详细注释: 这是脚本的核心处理循环。主线程会不断从 `fetch_queue` 中获取任务，
    #           加载数据、进行VQ编码、保存结果，然后将结果信息放入 `upload_queue`。
    # 运行状态描述: 使用 tqdm 进度条实时显示处理进度和当前正在处理的文件名。
    pbar = tqdm(total=len(files_to_process), desc=f'Process {accelerator.process_index}', disable=not accelerator.is_main_process)
    while True:
        # 从获取队列中取一个任务，如果队列为空则阻塞等待
        item = fetch_queue.get()
        if item is None: break  # 收到结束标志，说明所有文件都已处理完毕

        parquet_path_nfs, local_parquet_path = item
        try:
            # 加载本地的parquet文件
            # 使用 pandas 读取 parquet 文件，更稳定高效
            df = pd.read_parquet(local_parquet_path)
            dataset = FinevisionDataset(df, image_size=512, device=accelerator.device)
            dataloader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_processes, collate_fn=collate_fn, pin_memory=True)
            # dataloader = accelerator.prepare(dataloader)   # 不应该再共享，进程间是独立的！

            valid_pair_list = []
            # 遍历数据批次，进行VQ编码
            for image_batch, text_batch in dataloader:
                if image_batch is None: continue  # 跳过无效批次
                with torch.no_grad():
                    # 核心步骤：使用VQGAN模型将图像编码为离散的token序列
                    _, _, [_, _, vqcode_batch] = image_tokenizer._vq_model.encode(image_batch.to(dtype=image_tokenizer._dtype, device=accelerator.device))
                    vqcode_batch = vqcode_batch.view(image_batch.shape[0], -1).cpu().tolist()
                    # print('debug', '?' * 20, image_batch.shape[0])

                # 组装结果
                for text, vqcode in zip(text_batch, vqcode_batch):
                    valid_pair_list.append({
                        'data_type': 'i2t_multi', 'text': text, 'length': len(text) + len(vqcode) * 4,
                        'vqcode_512': json.dumps(vqcode), 'vqcode_multi768': 'no', 'width': 'no', 'height': 'no',
                    })
            # print('debug', '?' * 20, "parquet end..")

            # 如果当前文件处理后有有效数据，则保存到本地
            if valid_pair_list:
                # print('debug', '?'*20, len(valid_pair_list))
                subset_name = os.path.basename(os.path.dirname(parquet_path_nfs))
                output_basename = f'{os.path.splitext(os.path.basename(parquet_path_nfs))[0]}.jsonl'
                os.makedirs(os.path.join(process_save_dir, subset_name), exist_ok=True)
                local_jsonl_path = os.path.join(process_save_dir, subset_name, output_basename)
                with open(local_jsonl_path, 'w', encoding='utf-8') as f:
                    for line in valid_pair_list: f.write(json.dumps(line, ensure_ascii=False) + '\n')

                # 将本地结果路径放入上传队列，交由Uploader线程处理
                final_save_dir = os.path.join(TEMP_SAVE_PATH, subset_name)
                os.makedirs(final_save_dir, exist_ok=True)
                final_nfs_path = os.path.join(final_save_dir, output_basename)
                upload_queue.put((local_jsonl_path, final_nfs_path))

        except Exception as e:
            # raise
            print(f"处理文件 {local_parquet_path} 时发生错误: {e}")
        finally:
            # 清理已处理完的本地原始文件
            if os.path.exists(local_parquet_path): os.remove(local_parquet_path)
            # 更新主进程的进度条
            if accelerator.is_main_process:
                pbar.set_postfix_str(f"正在处理: {os.path.basename(parquet_path_nfs)}")
                pbar.update(1)

        # 调试模式下，只处理一个文件就退出
        if args.debug:
            print("Debug 模式: 已处理一个文件，即将退出。")
            break

    # --- 清理和同步 ---
    # 详细注释: 所有文件处理完毕后，主线程向上传队列发送一个结束信号，并等待后台线程完成所有任务。
    #           最后，所有进程同步，由主进程负责彻底清理所有临时文件和目录。
    # 运行状态描述: 显示正在等待后台线程结束，以及任务完成后的最终清理信息。
    pbar.close()

    # 处理流程结束，向上传队列发送结束标志，通知上传线程可以结束了
    upload_queue.put(None)

    # 等待所有后台线程执行完毕
    print(f"进程 {accelerator.process_index}: 等待后台线程结束...")
    fetcher.join()
    uploader.join()

    # 等待所有分布式进程都到达这个点
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        print("--- 所有文件处理完毕 ---")
        # 主进程最后清理所有临时目录
        if os.path.exists(temp_cache_dir): shutil.rmtree(temp_cache_dir)
        if os.path.exists(temp_save_dir): shutil.rmtree(temp_save_dir)
        print("临时目录已清理，任务完成。")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process FineVision datasets into VQ-Tokens.')
    parser.add_argument('--vqgan_path', type=str, default='data_process/vqgan', help='Path to VQGAN model directory')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for GPU processing')
    parser.add_argument('--num_processes', type=int, default=16, help='Number of worker processes for DataLoader')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode to process only one file')

    args = parser.parse_args()
    main(args)