# -*- coding: utf-8 -*-

"""
api_server.py

一个符合OpenAI API标准的多GPU、批处理推理服务器。

功能:
- 使用 FastAPI 启动一个兼容 OpenAI `/v1/chat/completions` 接口的Web服务。
- 启动时根据 --num-gpus 参数创建多个工作进程，每个进程在独立的GPU上加载模型。
- 主进程接收API请求，并将其放入异步队列。
- 一个独立的协程任务负责从队列中取出请求，等待?秒或达到最大批处理大小（max-batch-size）来组合批处理任务。
- 主进程通过简单的轮询机制，将批处理任务分发给不同的工作进程。
- 工作进程执行 `any_modal_chat_api` 中的生成逻辑。
- 主进程通过一个独立的线程从工作进程收集结果，并将其返回给对应的API请求。

运行方式:
请确保已安装所有依赖 (fastapi, uvicorn, transformers, torch, etc.)。
python api_server.py --model-path /path/to/your/model --vqgan-path /path/to/your/vqgan --num-gpus 2 --max-batch-size 8

API 请求示例 (使用 curl):
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "uni-model",
    "messages": [
      {
        "role": "user",
        "content": [
          {"type": "text", "text": "这张图片里有什么?"},
          {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..."}}
        ]
      }
    ]
  }'
"""

import argparse
import asyncio
import base64
import io
import logging
import hashlib
import os
import re
import threading
import time
import uuid
import traceback
from typing import Any, Dict, List, Optional, Union
from concurrent.futures import ThreadPoolExecutor
from PIL import Image

import torch
import torch.multiprocessing as mp
import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM

# 导入项目中的相关模块
from data_process.vqgan.image_tokenizer import ImageTokenizer
from evaluation.uni_infer import any_modal_chat_api
from modeling.uni_x_qwen3 import UniQwen3ForCausalLMInference
from tools.data_translator import translate_text_api

# --- 全局变量 ---
MAX_RETRIES = 5
# 用于在主进程中暂存待处理的API请求
request_queue = asyncio.Queue()
# 用于存储每个批处理任务对应的 aiohttp futures
results = {}


# --- OpenAI API 数据模型 ---
class ImageUrl(BaseModel):
    url: str


class ChatMessageContentPart(BaseModel):
    type: str
    text: Optional[str] = None
    image_url: Optional[ImageUrl] = None


class ChatMessage(BaseModel):
    role: str
    content: Union[str, List[ChatMessageContentPart]]


class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    max_tokens: Optional[int] = 1024
    min_new_tokens: Optional[int] = 1
    temperature: Optional[float] = 1.0
    top_p: Optional[float] = 0.95
    constrained_tokens: Optional[str] = None


# --- 工作进程 ---
def worker_process(
        worker_id: int,
        model_path: str,
        vqgan_path: str,
        task_queue: mp.Queue,
        result_queue: mp.Queue,
        args: argparse.Namespace,  # 接收主进程的命令行参数
):
    """
    工作进程函数。
    它在指定的GPU上加载模型，并从任务队列中循环获取并处理批处理任务。
    """
    device = f"cuda:{worker_id}"
    print(f"[工作进程 {worker_id}] 在 {device} 上初始化...")
    from tools.log import create_logger
    logger = create_logger("api_server", f"./outputs/api_server_{worker_id}.log")

    try:
        # 1. 加载模型和分词器
        tokenizer, model, img_tokenizer = None, None, None
        for _ in range(MAX_RETRIES):
            try:
                tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
                if tokenizer.pad_token is None:
                    tokenizer.pad_token = tokenizer.eos_token

                if args.model_cls == 'uni_qwen':
                    model_cls = UniQwen3ForCausalLMInference
                elif args.model_cls == 'auto':
                    model_cls = AutoModelForCausalLM
                else:
                    raise ValueError("model_cls in [uni_qwen, auto]")

                model = model_cls.from_pretrained(
                    model_path,
                    torch_dtype=torch.bfloat16,
                    device_map=device,
                    trust_remote_code=True,
                )
                img_tokenizer = ImageTokenizer(
                    ckpt_path=os.path.join(vqgan_path, "vqgan.ckpt"),
                    cfg_path=os.path.join(vqgan_path, "vqgan.yaml"),
                    device=device,
                )
                break
            except Exception as e:
                if _ == MAX_RETRIES - 1:
                    raise e

        ori_vocab_size = len(tokenizer)

        # 2. 创建生成函数
        generate_fn = any_modal_chat_api(
            model=model,
            tokenizer=tokenizer,
            img_tokenizer=img_tokenizer,
            ori_vocab_size=ori_vocab_size,
        )
        print(f"[工作进程 {worker_id}] 初始化完成。")

        # 3. 主循环，等待并处理任务
        while True:
            batch_id, batch_data = task_queue.get()
            if batch_data is None:  # 接收到结束信号
                break

            print(
                f"[工作进程 {worker_id}] 接收到批处理任务 {batch_id}，包含 {len(batch_data)} 个请求。"
            )

            # 准备generate函数的输入
            # 注意: 输入文本的翻译已移至主进程的 batching_loop 中，以避免阻塞GPU工作进程
            prompts = [item["prompt"] for item in batch_data]
            imgs_base64 = [item["images"] for item in batch_data]

            if not imgs_base64 or imgs_base64[0] is None:
                imgs_base64 = None
            else:
                # 确保批处理中的所有图像类型一致
                for img in imgs_base64:
                    assert img is not None, "要求batch内输入类型一致"

            # 一个批次中的所有请求使用相同的生成参数
            gen_params = batch_data[0]["gen_params"]

            # 执行生成
            outputs = generate_fn(
                prompts=prompts,
                imgs_base64=imgs_base64,
                modal_to_generate="text",  # 当前固定为文本生成，可根据需要进行参数化
                max_new_tokens=gen_params["max_new_tokens"],
                min_new_tokens=gen_params["min_new_tokens"],
                temperature=gen_params["temperature"],
                top_p=gen_params["top_p"],
                do_sample=(gen_params["temperature"] > 0.0),
                acceptable_tokens=gen_params["acceptable_tokens"],
            )

            # 记录请求和响应的文本内容
            for img64, item, output in zip(imgs_base64, prompts, outputs):
                md5_hash = hashlib.md5(img64[0].encode('utf-8')).hexdigest()
                _img_str = f'{md5_hash}.jpg'
                if os.environ.get("SERVER_DEBUG", 0):
                    _b = base64.b64decode(img64[0])
                    _img = Image.open(io.BytesIO(_b))
                    os.makedirs('./outputs/test_imgs/', exist_ok=True)
                    _img.save(f'./outputs/test_imgs/{_img_str}')

                logger.debug(f"Img[0]: {img64[0][:10]} === {_img_str}")
                logger.debug(f"Prompt: {item} -> Output: {output} \n\n {'=' * 20}")

            # 注意: 输出文本的翻译已移至主进程的 result_handling_loop 中
            # 这样可以立即释放GPU工作进程来处理下一个任务

            # 将每个请求的原始ID与生成结果打包
            batch_results = [
                {"request_id": item["request_id"], "output": output}
                for item, output in zip(batch_data, outputs)
            ]

            # 将结果放入结果队列
            result_queue.put((batch_id, batch_results))
            print(f"[工作进程 {worker_id}] 完成处理批处理任务 {batch_id}。")

    except Exception as e:
        traceback.print_exc()
        print(f"[工作进程 {worker_id}] 发生严重错误: {e}")


# --- 主进程中的批处理和结果处理 ---
async def batching_loop(app: FastAPI, task_queues: List[mp.Queue]):
    """
    一个异步协程，它持续地从全局请求队列中拉取请求，
    将它们组合成批次，然后分发给工作进程。
    """
    args = app.state.args  # 从 app state 获取命令行参数
    worker_idx = 0
    while True:
        # 期望形成批次
        await asyncio.sleep(float(os.environ.get('SERVER_SLEEP', 0.5)))

        batch_to_process = []
        # 从队列中取出请求，直到队列为空或达到最大批次大小
        while not request_queue.empty() and len(batch_to_process) < args.max_batch_size:
            request_item = await request_queue.get()
            batch_to_process.append(request_item)

        if not batch_to_process:
            continue

        # 准备要发送给工作进程的数据
        batch_id = str(uuid.uuid4())
        futures = []
        batch_data = []
        for request_id, req_data, gen_params, future in batch_to_process:
            batch_data.append(
                {
                    "request_id": request_id,
                    "prompt": req_data["prompt"],
                    "images": req_data["images"],
                    "gen_params": gen_params,
                }
            )
            futures.append(future)

        # [异步翻译] 如果启用了翻译，则在分发任务前，在此处异步翻译所有输入
        if args.translate_input:
            prompts_to_translate = [item["prompt"] for item in batch_data]

            def translate_with_fallback(p):
                try:
                    return translate_text_api(p, source_language="英文", target_language="中文", max_tokens=1024)[0]
                except Exception as e:
                    print(f"[分发器] 输入翻译失败 (回退到原文): {e}")
                    return p

            # 在线程池中执行同步的翻译函数，避免阻塞事件循环
            loop = asyncio.get_running_loop()
            tasks = [loop.run_in_executor(None, translate_with_fallback, p) for p in prompts_to_translate]
            translated_prompts = await asyncio.gather(*tasks)

            # 将翻译后的文本更新回 batch_data
            for i, translated_prompt in enumerate(translated_prompts):
                # Future-TODO 翻译并不稳定，因此这里可能会把特殊token给删除掉，需要加回来
                if '<img>' not in translated_prompt:
                    translated_prompt = '<|vision_start|><img><|vision_end|>\n' + translated_prompt
                batch_data[i]["prompt"] = translated_prompt

            print(f"[分发器] 完成批处理 {batch_id} 的输入翻译。")

        # 存储futures，以便在收到结果后可以正确地返回响应
        results[batch_id] = futures

        # 使用轮询方式将任务分发给一个工作进程
        task_queues[worker_idx].put((batch_id, batch_data))
        print(f"[分发器] 发送批处理任务 {batch_id} 到工作进程 {worker_idx}。")
        worker_idx = (worker_idx + 1) % len(task_queues)


def result_handling_loop(result_queue: mp.Queue, args: argparse.Namespace):
    """
    一个在独立线程中运行的同步函数。
    它监听来自工作进程的结果队列，并解析对应的 asyncio futures。
    """
    while True:
        batch_id, batch_results = result_queue.get()

        if batch_id not in results:
            continue

        # [并发翻译] 如果启用翻译，则在此处并发翻译所有模型输出
        # 这发生在专用的结果处理线程中，不会阻塞GPU工作进程或主事件循环
        if args.translate_output:
            outputs_to_translate = [res["output"] for res in batch_results]

            def translate_output_with_fallback(text_to_translate):
                # 目前只考虑输出判断题和多选题答案，这样可以不再调取翻译接口
                text = text_to_translate.strip()
                if not text:
                    return ""

                # 优先匹配 A,B,C,D 等多选题选项
                match = re.search(r'[A-D]', text)
                if match:
                    return match.group(0)

                # 再匹配“是/否”等判断题选项
                if '是' in text:
                    return 'yes'
                if '否' in text or '不' in text:
                    return 'no'

                # 如果没有匹配到特定模式，返回原始文本
                return text_to_translate

            # 在线程池中并发执行翻译
            with ThreadPoolExecutor(max_workers=len(outputs_to_translate)) as executor:
                translated_outputs = list(executor.map(translate_output_with_fallback, outputs_to_translate))

            # 将翻译结果更新回 batch_results
            for i, translated_output in enumerate(translated_outputs):
                batch_results[i]["output"] = translated_output

            print(f"[结果处理器] 完成批处理任务 {batch_id} 的输出翻译。")

        futures = results.pop(batch_id)

        # 创建一个 request_id -> result 的映射，方便查找
        result_map = {res["request_id"]: res["output"] for res in batch_results}

        # 遍历这个批次中的所有 futures
        # 因为工作进程保证了批处理内请求的顺序，所以 futures 和 batch_results 的顺序是一致的
        for i, future in enumerate(futures):
            # 从 batch_results 中获取原始的 request_id
            request_id = batch_results[i]["request_id"]
            output = result_map.get(
                request_id, "错误：未找到此请求的结果。"
            )

            # 使用 call_soon_threadsafe 在主事件循环中安全地设置 future 的结果
            # 这是在线程和 asyncio 之间进行通信的标准方式
            future.get_loop().call_soon_threadsafe(future.set_result, output)
        print(f"[结果处理器] 处理完批处理任务 {batch_id} 的结果。")


# --- FastAPI 应用 ---
app = FastAPI()


@app.post("/v1/chat/completions")
async def create_chat_completion(request: ChatCompletionRequest):
    """
    兼容 OpenAI 的聊天补全接口。
    """
    request_id = str(uuid.uuid4())
    args = app.state.args  # 获取启动参数

    # 1. 解析输入
    # 通常，用户最新的消息是当前的提示
    user_message = request.messages[-1]
    prompt_text = ""
    images_b64 = []

    if isinstance(user_message.content, str):
        prompt_text = user_message.content
    else:  # 多模态输入的 content 是一个列表
        full_prompt = ""
        for part in user_message.content:
            if part.type == "input_text" or part.type == "text":
                full_prompt += part.text.replace('<img>', '').replace('<image>', '')
            elif part.type == "input_image" or part.type == 'image_url':
                # 图片格式为 "data:image/jpeg;base64,{b64_string}"
                b64_string = part.image_url.url.split(",")[-1]
                # _64 = base64.b64decode(b64_string)
                # _img = Image.open(io.BytesIO(_64))  # Future-TODO 后面移除，这里为了测试能不能decode
                images_b64.append(b64_string)
                full_prompt = "<|vision_start|><img><|vision_end|>\n" + full_prompt  # 在文本中插入占位符和对应起止
                # Future-TODO 因为目前模型没有训练过图文交错，所以暂时将图片放在最前面。
            else:
                print(f"[error] wrong input type: {part.type}")
        # 模型需要通过这种明确的指令来让其直接生成答案
        env_prompt = os.environ.get('ENV_SERVER_PROMPT', None)
        if env_prompt is None:
            prompt_text = full_prompt + '\n' + ('根据图片和问题，我的最终答案为: ' if args.translate_input
                                                else 'Based on the image and the question, my final answer is: ')
        else:
            prompt_text = full_prompt + '\n' + env_prompt

    # 2. 准备要放入队列的数据
    request_data = {"prompt": prompt_text, "images": images_b64 if len(images_b64) > 0 else None}

    # 如果启动参数设置了生成参数，则覆盖请求中的参数
    gen_params = {
        "max_new_tokens": args.max_tokens if args.max_tokens is not None else request.max_tokens,
        "min_new_tokens": args.min_new_tokens if args.min_new_tokens is not None else request.min_new_tokens,
        "temperature": args.temperature if args.temperature is not None else request.temperature,
        "top_p": args.top_p if args.top_p is not None else request.top_p,
        "acceptable_tokens": args.acceptable_tokens if args.acceptable_tokens != 'all' else None,
    }

    # 3. 创建一个 future，用于稍后接收结果
    loop = asyncio.get_running_loop()
    future = loop.create_future()

    # 4. 将请求放入队列，然后等待 future 完成
    await request_queue.put((request_id, request_data, gen_params, future))
    result = await future

    # 5. 格式化并返回 OpenAI 风格的响应
    response_data = {
        "id": "chatcmpl-" + request_id,
        "object": "chat.completion",
        "created": int(time.time()),
        "model": request.model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": result},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 0,  # 占位符
            "completion_tokens": 0,  # 占位符
            "total_tokens": 0,  # 占位符
        },
    }

    return JSONResponse(content=response_data)


# --- 主程序入口 ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="多模态 API 服务器")
    parser.add_argument(
        "--host", type=str, default="0.0.0.0", help="服务器绑定的主机地址。"
    )
    parser.add_argument("--port", type=int, default=33218, help="服务器绑定的端口。")
    parser.add_argument(
        "--num-gpus",
        type=int,
        default=torch.cuda.device_count(),
        help="要使用的GPU数量。",
    )
    parser.add_argument(
        "--max-batch-size", type=int, default=16, help="推理的最大批处理大小。"
    )
    parser.add_argument("--model-path", type=str, required=True, help="多模态模型的路径。")
    parser.add_argument("--model-cls", type=str, default='uni_qwen', help="多模态模型的class。")
    parser.add_argument(
        "--vqgan-path", type=str, default='data_process/vqgan', help="VQGAN 模型的目录路径。"
    )
    # 1,1对应输入翻译，输出也翻译，1,0代表只翻译输入，以此类推
    parser.add_argument(
        "--translate", type=str, default='1,1', help="是否翻译prompts, 格式为 '1,0', '0,1' 等"
    )
    # 添加用于覆盖生成参数的命令行选项
    parser.add_argument("--max-tokens", type=int, default=None, help="强制覆盖所有请求的 max_tokens。")
    parser.add_argument("--min-new-tokens", type=int, default=None, help="强制覆盖所有请求的 min_new_tokens。")
    parser.add_argument("--temperature", type=float, default=0.0, help="强制覆盖所有请求的 temperature。")
    parser.add_argument("--top-p", type=float, default=None, help="强制覆盖所有请求的 top_p。")
    parser.add_argument("--acceptable-tokens", type=str, default="all", help="设置tokens constrained")  # 是,否,yes,no

    args = parser.parse_args()

    # 解析 translate 参数
    try:
        in_translate, out_translate = map(int, args.translate.split(','))
        args.translate_input = bool(in_translate)
        args.translate_output = bool(out_translate)
    except ValueError:
        raise ValueError("`--translate` 参数格式不正确。请使用 '1,1', '1,0', '0,1', or '0,0'。")

    # 将命令行参数附加到 app 实例，以便在 FastAPI 事件处理程序中访问
    app.state.args = args

    # 当启用输入翻译时，检查并发量是否会超出外部API限制
    if args.translate_input:
        # translate_text_api 是一个外部API，其并发限制为100
        # 因此，最大并发请求数 (max-batch-size * num-gpus) 必须小于等于200 (很难跑到这个量)
        max_concurrency = args.max_batch_size * args.num_gpus
        if max_concurrency > 200:
            raise ValueError(
                f"并发限制错误: 当启用输入翻译时, 'max-batch-size' * 'num-gpus' 的值不能超过 100。"
                f"当前值为: {args.max_batch_size} * {args.num_gpus} = {max_concurrency}"
            )

    # 设置多进程启动方法为 "spawn"，以避免CUDA在fork模式下的问题
    mp.set_start_method("spawn", force=True)

    # 创建用于进程间通信的队列
    task_queues = [mp.Queue() for _ in range(args.num_gpus)]
    result_queue = mp.Queue()

    # 启动工作进程
    processes = []
    for i in range(args.num_gpus):
        p = mp.Process(
            target=worker_process,
            # 将命令行参数传递给工作进程
            args=(i, args.model_path, args.vqgan_path, task_queues[i], result_queue, args),
            daemon=True,
        )
        p.start()
        processes.append(p)

    # 在主进程中启动结果处理线程
    result_thread = threading.Thread(
        # 将命令行参数传递给结果处理循环
        target=result_handling_loop, args=(result_queue, args), daemon=True
    )
    result_thread.start()


    # 注册 FastAPI 的启动事件，用于启动批处理协程
    @app.on_event("startup")
    async def startup_event():
        # 将 app 实例和任务队列传递给批处理循环
        asyncio.create_task(batching_loop(app, task_queues))


    # 启动 FastAPI 服务器
    print(f"服务器将在 http://{args.host}:{args.port} 上运行")
    uvicorn.run(app, host=args.host, port=args.port)