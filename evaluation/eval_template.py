import os
import subprocess
import sys
import time
from typing import List, Dict, Any
import re  # 导入 re 模块用于正则表达式
from tools.log import create_logger
import signal
import datetime  # 用于生成时间戳
import glob  # 用于查找文件
import shutil  # 用于移动文件

# --- 核心配置区 ---
# 在这里集中配置所有要运行的实验
# 每个字典代表一个完整的实验流程（生成 + 评估）

# --- 全局默认配置 (如果实验配置中未指定，则使用此处的默认值) ---
DEFAULT_MODEL_CLS = "uni_qwen"
DEFAULT_CFG = 4.0
DEFAULT_BATCH_SIZE = 160
DEFAULT_NUM_GEN_REPEATS = 4
main_logger = create_logger("evaluation_all", './outputs/eval_all_using_template.log')

# --- 文本评测配置 ---
TEXT_BATCH_SIZE = 64
NUM_FEW_SHOT = 0
TASKS = "mmlu,arc_challenge,arc_easy,boolq,winogrande"
# MODEL_ARGS 中的 {ckpt} 将被动态替换为每个实验的 model_path
MODEL_ARGS = 'pretrained={ckpt},model_type={model_type},architectures="[{arch}]"'
# Hugging Face 配置 (lm-eval 会使用)
HF_HOME = "../hf_cache"
HF_TOKEN = ""  # 替换为 Hugging Face Token

# 图像理解测评配置
VIS_UND_TASKS = ["MMBench_DEV_EN", "MMBench_DEV_CN", "POPE", "MME"]
GEN_IMG_TASKS = ["dpg_bench", "geneval", "genai"]
SUPPORTED_TASKS = [*GEN_IMG_TASKS, "text", "vis_und"]

# --- 示例实验列表 ---
# 注意： geneval, dpg 需要在不同实验里
EXPERIMENTS = [
    # 原始 Python 脚本中的 DPG-Bench 评估 (中文)
    # {
    #     "name": "DPG-Bench Evaluation (zh)",
    #     "model_path": "../mock/ckpts/Qwen2.5-3B-uni-X/3B-unix-x12_6-try_sota-big-SFT-ignore_ins/20250912_2341/checkpoint-7800",
    #     "dpg_bench_prompts_path": "evaluation/T2I_Eval/dpg_bench/dpg_prompts_zh_fixed.jsonl",
    #     "eval_type": ["dpg_bench", "text"], # 指定评估类型：dpg_bench图像评估 + text文本评估
    #     # 其他参数将使用上面的全局默认值
    # },
    # vis und
    # {
    #     "name": "Visual Understanding Evaluation",
    #     "model_path": "path/to/your/model",
    #     "eval_type": ["vis_und"],
    #     # 可以自定义服务器和客户端参数
    #     "vis_und_port": 33218,
    #     "vis_und_api_nproc": 80,
    # },
]

try:
    from evaluation.exp import EXPERIMENTS  # noqa
except Exception as e:
    print("需要新建 exp.py 存放实验list ")
    raise e
try:
    # for quick debug
    from evaluation.exp import VIS_UND_TASKS  # noqa

    print(f"强制设置了 VIS_UND TASKS {VIS_UND_TASKS}")
except:
    pass


def gen_save_path(model_path: str, exp_config: Dict[str, Any], eval_task) -> str:
    """
    根据模型路径和实验配置，自动生成图像保存路径。
    """
    # 使用正则表达式从模型路径中解析出 model_tag 和 checkpoint_id
    # e.g., .../3B-unix-sft_v3/20250913_1654/checkpoint-5200
    match = re.search(r'([^/]+)/(\d{8}_\d{4})/?([^/]+)?$', model_path.rstrip('/'))
    if not match:
        # 如果正则不匹配，使用一个备用方案，直接用最后一级目录名
        model_name = os.path.basename(model_path.rstrip('/'))
    else:
        model_tag = match.group(1)
        ckpt_id = match.group(3)
        # 组合 model_tag 和 ckpt_id
        model_name = f"{model_tag}{'____' + ckpt_id if ckpt_id else ''}"

    # 确定评估类型对应的结果文件夹
    eval_folder_name = "unknown_results"
    if "dpg_bench" in eval_task:
        eval_folder_name = "dpg_results"
    elif "geneval" in eval_task:
        eval_folder_name = "geneval_results"
    elif "genai" in eval_task:
        eval_folder_name = "genai_results"
    else:
        raise ValueError(f"不需要为task {eval_task} 生成保存路径")

    # 从 prompts_path 中推断语言
    lang = ''
    for k in exp_config:
        if 'prompts_path' in k:
            prompts_path = exp_config[k]
            if "zh" in prompts_path:
                lang = "zh"
            else:
                lang = 'en'
            break

    # 获取 CFG 和生成次数配置
    cfg = exp_config.get('cfg', DEFAULT_CFG)
    num_repeats = exp_config.get('num_gen_repeats', DEFAULT_NUM_GEN_REPEATS)

    # 组装最终的文件名
    save_name = f"{model_name}_____{lang}_cfg{cfg}_{num_repeats}run"

    # 返回完整的保存路径
    return os.path.join("evaluation", "T2I_Eval", eval_folder_name, save_name)


def run_command(command: List[str], env: dict = None, **kwargs):
    """辅助函数：运行一个外部命令并检查其是否成功执行。"""
    main_logger.info(f"🚀 运行命令: {' '.join(command)}")
    try:
        # 使用 logger 捕获子进程的输出
        process = subprocess.Popen(command, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, **kwargs)
        for line in iter(process.stdout.readline, ''):
            main_logger.info(line.strip())
        process.wait()
        if process.returncode != 0:
            raise subprocess.CalledProcessError(process.returncode, command)
        main_logger.info("✅ 命令成功完成。")
    except FileNotFoundError:
        main_logger.error(f"❌ 错误: 命令 '{command[0]}' 未找到。请确保它已安装并且在系统的 PATH 中。")
        sys.exit(1)
    except subprocess.CalledProcessError as e:
        main_logger.error(f"❌ 命令执行失败，退出码: {e.returncode}")
        sys.exit(1)
    except KeyboardInterrupt:
        main_logger.info("\n🛑 用户中断了脚本。")
        sys.exit(1)


def run_generation(exp_config: Dict[str, Any], gpu_list: List[str]):
    """为单个实验配置并行执行图像生成任务。"""
    eval_types = exp_config.get('eval_type', [])
    image_eval_tasks = [t for t in eval_types if t in GEN_IMG_TASKS]

    if not image_eval_tasks:
        main_logger.info("⏭️  未配置图像评估类型，跳过图像生成。")
        return

    # 存储每个生成任务的保存路径，供后续评估步骤使用
    if 'generated_paths' not in exp_config:
        exp_config['generated_paths'] = {}

    for task in image_eval_tasks:
        main_logger.info(f"\n{'=' * 20} 开始为评估类型 '{task}' 生成图像 {'=' * 20}")

        # 为当前任务创建临时配置
        task_config = exp_config.copy()
        task_config['eval_type'] = [task]

        chunks = len(gpu_list)
        model_path = task_config['model_path']
        save_path = gen_save_path(model_path, task_config, task)
        exp_config['generated_paths'][task] = save_path

        prompts_path = task_config[f'{task}_prompts_path']
        model_cls = task_config.get('model_cls', DEFAULT_MODEL_CLS)
        cfg_scale = task_config.get('cfg', DEFAULT_CFG)
        batch_size = task_config.get('batch_size', DEFAULT_BATCH_SIZE)
        grid_img = 1 if task == "dpg_bench" else 0
        num_gen_repeats = task_config.get('num_gen_repeats', DEFAULT_NUM_GEN_REPEATS)

        main_logger.info(f"🎨 开始为实验 '{task_config.get('name', 'Untitled')}' [{task}] 并行生成图像...")
        main_logger.info(f"  - 模型路径: {model_path}")
        main_logger.info(f"  - 结果保存路径: {save_path}")
        main_logger.info("-" * 50)

        processes = []
        for idx in range(chunks):
            gpu_id = gpu_list[idx]
            env = os.environ.copy()
            env['PYTHONPATH'] = '.'
            env['CUDA_VISIBLE_DEVICES'] = gpu_id

            command = [
                sys.executable,
                "evaluation/T2I_Eval/gen_img.py",
                "--model_path", model_path,
                "--model_cls", model_cls,
                "--save_path", save_path,
                "--cfg_scale", str(cfg_scale),
                "--prompts_path", prompts_path,
                "--batch_size", str(batch_size),
                "--num_chunks", str(chunks),
                "--chunk_idx", str(idx),
                "--grid_img", str(grid_img),
                "--num_gen_repeats", str(num_gen_repeats),
            ]

            main_logger.info(f"  - 启动任务 {idx + 1}/{chunks} on GPU {gpu_id}...")
            # 将子进程的输出重定向到日志文件
            log_file = f"./outputs/gen_{task}_{idx}.log"
            proc = subprocess.Popen(command, env=env, stdout=open(log_file, 'w'), stderr=subprocess.STDOUT)
            processes.append(proc)

        main_logger.info(f"\n⏳ 等待 '{task}' 的所有图像生成任务完成，这可能需要一些时间...")
        for i, p in enumerate(processes):
            p.wait()
            if p.returncode != 0:
                main_logger.error(f"  - ❌ 生成任务 {i + 1}/{chunks} 失败，退出码: {p.returncode}。详情请查看 ./outputs/gen_{task}_{i}.log")
                sys.exit(1)
            else:
                main_logger.info(f"  - 任务 {i + 1}/{chunks} 已完成。")

        main_logger.info(f"\n✅ '{task}' 的图像生成任务均已完成！")


def run_gen_img_evaluation(exp_config: Dict[str, Any], gpu_list: List[str]):
    """根据实验配置执行相应的图像评估任务。"""
    eval_types = exp_config.get('eval_type', [])
    if not eval_types:
        main_logger.info("⏭️ 未指定 'eval_type'，跳过评估步骤。")
        return

    generated_paths = exp_config.get('generated_paths', {})
    num_gen_repeats = exp_config.get('num_gen_repeats', DEFAULT_NUM_GEN_REPEATS)
    chunks = len(gpu_list)
    env = os.environ.copy()
    env['PYTHONPATH'] = '.'

    for eval_type in eval_types:

        if eval_type not in GEN_IMG_TASKS:
            continue

        save_path = generated_paths[eval_type]

        path_info = f"'{save_path}'"
        main_logger.info(f"📊 开始为 {path_info} 执行 '{eval_type}' 评估...")

        if eval_type == "dpg_bench":
            command = [
                "accelerate", "launch",
                "--num_processes", str(chunks),
                "--mixed_precision", 'fp16',
                "evaluation/T2I_Eval/dpg_bench/compute_dpg_bench.py",
                "--image-root-path", save_path,
                "--resolution", "512",
                "--pic-num", str(num_gen_repeats)
            ]
            main_logger.info(f"🚀 在后台启动 dpg_bench 评估: {' '.join(command)}")
            subprocess.Popen(command, env=env)
            main_logger.info("✅ dpg_bench 已在后台启动，脚本将继续执行。")

        elif eval_type == "geneval":
            command = ["bash", "evaluation/T2I_Eval/geneval/process.sh", save_path]
            run_command(command, env=env)

        elif eval_type == "genai":
            pass
            # Future-TODO
        else:
            raise NotImplementedError


def run_vis_und_evaluation(exp_config: Dict[str, Any], gpu_list: List[str]):
    """
    执行视觉理解评估，包含启动API服务器和运行评估客户端。
    此版本经过修改，通过管理进程组来确保能可靠地终止API服务器及其所有子进程。
    """
    if "vis_und" not in exp_config.get("eval_type", []):
        return

    # 归档上一次的评估结果
    source_dir = "./outputs/my_custom_model"
    files_to_archive = glob.glob(os.path.join(source_dir, "my*"))
    if files_to_archive:
        # 使用当前时间创建归档文件夹名称
        time_str = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        archive_dir = os.path.join(source_dir, f"archive_{time_str}")
        os.makedirs(archive_dir, exist_ok=True)
        main_logger.info(f"📦 找到旧的评估结果，正在归档到: {archive_dir}")
        for f in files_to_archive:
            try:
                # 移动文件到归档目录
                shutil.move(f, archive_dir)
            except Exception as e:
                main_logger.warning(f"  - 归档文件 {f} 失败: {e}")
        main_logger.info("✅ 归档完成。")

    main_logger.info("🧠 开始执行视觉理解评估...")
    model_path = exp_config['model_path']
    server_gpu = ','.join(gpu_list)
    port = exp_config.get("vis_und_port", 33218)

    server_params = {
        "num-gpus": exp_config.get("vis_und_server_gpus", 1),
        "max-tokens": exp_config.get("vis_und_max_tokens", 3),
        "min-new-tokens": exp_config.get("vis_und_min_new_tokens", 1),
        "acceptable-tokens": exp_config.get("vis_und_acceptable_tokens", "all"),  # 默认还是支持全部tokens
        "translate": exp_config.get("vis_und_translate", '0,0'),
        "max-batch-size": exp_config.get("vis_und_max_batch_size", 80),
        "model-cls": exp_config.get("model_cls", DEFAULT_MODEL_CLS)
    }
    client_model_name = exp_config.get("vis_und_model_name", "my_custom_model")
    client_api_nproc = exp_config.get("vis_und_api_nproc", 80)
    judge_model = exp_config.get('vis_und_judge_model', 'exact_matching')

    server_env = os.environ.copy()
    server_env.update({
        'PYTHONPATH': '.', 'CUDA_VISIBLE_DEVICES': server_gpu,
        'SERVER_SLEEP': str(exp_config.get("vis_und_server_sleep", 0.5))
    })
    if "tr_glm_api_key" in exp_config:
        server_env['TR_GLM_API_KEY'] = exp_config["tr_glm_api_key"]

    server_command = [sys.executable, "evaluation/api_server.py", "--model-path", model_path, "--port", str(port)]
    for key, value in server_params.items():
        server_command.append(f"--{key}")
        server_command.append(f"{value}")
    main_logger.info(f"启动指令: {' '.join(server_command)}")

    server_process = None
    try:
        main_logger.info(f"🚀 在 GPU {server_gpu} 上后台启动 API 服务器...")
        # 【关键修改 1】: 使用 preexec_fn=os.setsid 将服务器进程放入一个新的进程组。
        # 这使得我们可以向整个组（包括所有孙子进程）发送信号。
        server_process = subprocess.Popen(
            server_command,
            env=server_env,
            preexec_fn=os.setsid
        )

        main_logger.info("⏳ 等待 30 秒以确保服务器完全启动...")
        time.sleep(30)

        # 检查服务器进程是否在等待期间意外退出
        if server_process.poll() is not None:
            raise RuntimeError("API 服务器在启动期间意外退出，请检查其日志。")

        # --- 运行评估客户端 ---
        # 设置调用API客户端所需的环境变量
        client_env = os.environ.copy()
        client_env.update({
            'HF_HOME': HF_HOME, 'HF_TOKEN': HF_TOKEN,
            # 'OPENAI_API_KEY': "sk-none",  # 本地服务, API Key不重要
            # 'OPENAI_API_BASE': f"http://localhost:{port}/v1"
        })

        # 循环执行所有指定的VQA评测任务
        for task in VIS_UND_TASKS:
            main_logger.info(f"  - 🚀 开始VQA任务: {task}")
            command = [
                sys.executable,
                "evaluation/eval_vqa.py",
                "--task", task,
                "--api_nproc", str(client_api_nproc),
                "--save_name", exp_config.get("name", "Untitled")
            ]
            run_command(command, env=client_env)
            main_logger.info(f"  - ✅ VQA任务 '{task}' 完成")

    finally:
        if server_process and server_process.poll() is None:
            pgid = os.getpgid(server_process.pid)
            main_logger.info(f"🛑 正在终止 API 服务器进程组 (PGID: {pgid})...")
            try:
                # 【关键修改 2】: 使用 os.killpg 向整个进程组发送终止信号 (SIGTERM)。
                os.killpg(pgid, signal.SIGTERM)

                # 等待主进程（启动器）响应信号并退出
                server_process.wait(timeout=20)
                main_logger.info("✅ API 服务器进程组已正常终止。")
            except subprocess.TimeoutExpired:
                main_logger.warning(f"🛑 服务器进程组未能正常终止，强制关闭 (PGID: {pgid})...")
                # 如果 SIGTERM 无效，则发送 SIGKILL 强制杀死整个进程组
                os.killpg(pgid, signal.SIGKILL)
                main_logger.info("✅ API 服务器进程组已强制关闭。")
            except ProcessLookupError:
                # 这种情况可能发生：在我们检查 poll() 和执行 killpg() 之间，进程已经自己退出了
                main_logger.warning("服务器进程组在尝试终止时已不存在。")

    main_logger.info("✅ 视觉理解评估完成。")


def run_text_evaluation(exp_config: Dict[str, Any], gpu_list: List[str]):
    """为单个实验配置执行 lm-eval 文本评估。"""
    if "text" not in exp_config.get("eval_type", []):
        return

    model_path = exp_config['model_path']
    num_processes = len(gpu_list)
    model_type = exp_config.get('model_type', 'qwen3')
    arch = exp_config.get('arch', 'Qwen3ForCausalLM')
    model_args_formatted = MODEL_ARGS.format(ckpt=model_path, model_type=model_type, arch=arch)

    main_logger.info(f"📝 开始为实验 '{exp_config.get('name', 'Untitled')}' 执行文本评测...")
    main_logger.info(f"  - 模型路径: {model_path}")
    main_logger.info(f"  - 评测任务: {TASKS}")
    main_logger.info("-" * 50)

    env = os.environ.copy()
    env.update({'PYTHONPATH': '.', 'HF_HOME': HF_HOME, 'HF_TOKEN': HF_TOKEN})

    command = [
        "accelerate", "launch", "--num_processes", str(num_processes),
        "--main_process_port", "19763", "-m", "lm_eval", "--model", "hf",
        "--tasks", TASKS, "--batch_size", str(TEXT_BATCH_SIZE),
        "--trust_remote_code", "--num_fewshot", str(NUM_FEW_SHOT),
        "--model_args", model_args_formatted,
    ]
    run_command(command, env=env)


def main():
    """主执行函数"""
    main_logger.info("🔄 正在从 Git 仓库拉取最新代码...")
    run_command(["git", "pull"])
    main_logger.info("-" * 70)

    # 校验所有实验配置中的评估任务是否有效
    for exp in EXPERIMENTS:
        if "eval_type" in exp:
            for task in exp["eval_type"]:
                if task not in SUPPORTED_TASKS:
                    main_logger.error(f"❌ 实验 '{exp.get('name', '未命名')}' 中包含不支持的评估任务: '{task}'")
                    main_logger.error(f"   支持的任务列表为: {SUPPORTED_TASKS}")
                    sys.exit(1)
    main_logger.info("✅ 所有实验配置中的评估任务均有效。")

    gpu_list_str = os.environ.get('CUDA_VISIBLE_DEVICES', '0,1,2,3,4,5,6,7')
    gpu_list = gpu_list_str.split(',')
    main_logger.info(f"🖥️ 检测到可用 GPUs: {gpu_list_str} (共 {len(gpu_list)} 个)")
    main_logger.info("-" * 70)

    total_experiments = len(EXPERIMENTS)
    for i, experiment_config in enumerate(EXPERIMENTS):
        exp_name = experiment_config.get('name', f'实验 #{i + 1}')
        main_logger.info(f"🚀🚀🚀 开始执行实验 {i + 1}/{total_experiments}: {exp_name} 🚀🚀🚀")

        run_generation(experiment_config, gpu_list)
        main_logger.info("-" * 50)

        run_gen_img_evaluation(experiment_config, gpu_list)
        main_logger.info("-" * 50)

        run_text_evaluation(experiment_config, gpu_list)
        main_logger.info("-" * 50)

        run_vis_und_evaluation(experiment_config, gpu_list)
        main_logger.info("-" * 50)

    main_logger.info("🎉🎉🎉 所有实验均已执行完毕！🎉🎉🎉")


if __name__ == "__main__":
    main()