import os
import json
import base64
import traceback

import pandas as pd  # 使用 pandas 进行数据处理
import requests
import re
import time
from fire import Fire
from tqdm import tqdm
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Dict, Any, Optional
import io
from PIL import Image
from string import punctuation

# MME.tsv (判断题)
# index   category        image_path      question        answer  image
# POPE.tsv (判断题)
# image   question        answer  category        index
# MMBench_DEV_EN.tsv/MMBench_DEV_CN.tsv (选择题)
# index   question        hint    A       B       C       D       answer  category        image   source  l2-category     comment split
# SEEDBench_IMG.tsv (选择题)
# answer  question        A       B       C       D       image   index   category

"""
API 请求示例 (使用 curl):
curl http://localhost:33218/v1/chat/completions \
  -H "Content-Type": "application/json" \
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


# 对于MME.tsv 每个index为奇数的row，都是和上一行是一对数据，共用一张图片，所以奇数row的image为上一行的index
# 在目前的分数之外，对每个目录，计算 acc, acc+这两个指标 acc+就是对于一对数据，都答对了才算对

# MMBench 数据里面已经处理好了，image可能是base64字符串，也可能是某个row index，如果是index，就需要使用那个index的image
# 我们目前是手动实现了circular的evaluation，现在情况是他们应该已经
# A,B,C,D并不一定都存在，需要判断如果没有该选项，构造prompt时自动跳过
# 而且MMBench的index不连续，不能根据行号来选择。（MME和POPE的都是连续的排序好的）

def _post_request(session: requests.Session, server_address: str, text_prompt: str, image_b64: str, retries: int = 3, backoff_factor: float = 0.5) -> str:
    """向API服务器发送单个请求，支持重试"""
    # 数据验证，如果有错直接崩掉
    _64 = base64.b64decode(image_b64)
    _img = Image.open(io.BytesIO(_64))

    payload = {
        "model": "uni-model",
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": text_prompt},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}}
        ]}]
    }
    for i in range(retries):
        try:
            # 增加请求超时时间
            response = session.post(server_address, json=payload, timeout=120)
            response.raise_for_status()
            return response.json()['choices'][0]['message']['content']
        except requests.exceptions.RequestException as e:
            if i == retries - 1:
                return f"Error: Request failed after {retries} retries: {e}"
            time.sleep(backoff_factor * (2 ** i))  # 指数退避
        except (KeyError, IndexError, json.JSONDecodeError) as e:
            return f"Error: Invalid response format from server: {e}"
    return "Error: Request failed unexpectedly."


def text_to_words(text: str):
    _punc = punctuation + '，。？！“”‘’｜、·～'
    text = str(text).strip().lower()
    for c in _punc:
        text = text.replace(c, ' ')
    words = text.split(' ')
    return words


def _process_yes_no_response(text: str) -> str:
    """解析判断题的回答，归一化为 'yes' 或 'no'"""
    if '(yes)' in text or '[yes]' in text or '<yes>' in text or 'yes)' in text:
        return 'yes'
    if '(no)' in text or '[no]' in text or '<no>' in text or 'no)' in text:
        return 'no'

    words = text_to_words(text)
    if ('yes' in words or '是' in words) and not ('no' in words or '否' in words or '不' in words):
        return 'yes'
    if not ('yes' in words or '是' in words) and ('no' in words or '否' in words or '不' in words):
        return 'no'
    return 'no'


def _process_mmbench_response(text: str) -> str:
    """解析选择题的回答，提取选项字母"""
    text = str(text).strip()
    match = re.search(r'[A-D]', text, re.IGNORECASE)
    return match.group(0).upper() if match else 'unknown'


def calculate_yes_no_metrics(labels: List[str], predictions: List[str], pos_label: str = 'yes') -> Dict[str, float]:
    """
    手动计算判断题的 precision, recall, f1-score.
    """
    tp, fp, fn = 0, 0, 0  # True Positives, False Positives, False Negatives
    for gt, pred in zip(labels, predictions):
        if pred == pos_label and gt == pos_label:
            tp += 1
        elif pred == pos_label and gt != pos_label:
            fp += 1
        elif pred != pos_label and gt == pos_label:
            fn += 1

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    return {"precision": precision, "recall": recall, "f1": f1}


def evaluate_yes_no(samples: List[Dict[str, Any]], server_address: str, api_nproc: int, task: str) -> List[Dict[str, Any]]:
    """处理判断题 (MME, POPE)"""
    results = []
    # samples = [row for _, row in df.iterrows()] # 直接使用传入的samples列表

    with ThreadPoolExecutor(max_workers=api_nproc) as executor:
        with requests.Session() as session:
            future_to_sample = {
                executor.submit(
                    _post_request, session, server_address,
                    row['question'],
                    row['image']
                ): row for row in samples
            }
            # 迭代完成的任务，并显示进度条
            for future in tqdm(as_completed(future_to_sample), total=len(samples), desc=f"Evaluating {task}"):
                row = future_to_sample[future]
                try:
                    # 从 future 对象中获取结果
                    raw_prediction = future.result()
                    prediction = _process_yes_no_response(raw_prediction)
                    gt = str(row['answer']).lower()
                    results.append({
                        'index': row['index'],
                        'question': row['question'],
                        'ground_truth': gt,
                        'prediction': prediction,
                        'raw_prediction': raw_prediction,
                        'correct': gt == prediction
                    })
                except Exception as e:
                    # 记录更详细的错误信息，便于排查
                    traceback.print_exc()
                    results.append({
                        'index': row.get('index', 'N/A'),
                        'question': row.get('question', 'N/A'),
                        'ground_truth': str(row.get('answer', 'N/A')).lower(),
                        'prediction': 'error',
                        'raw_prediction': str(e)
                    })
    return results


def _evaluate_single_mmbench_circular(session: requests.Session, server_address: str, row: Dict[str, Any]) -> Dict[str, Any]:
    """对单个 MMBench 题目进行循环（Circular）评测，4次请求都对才算对"""
    options = ['A', 'B', 'C', 'D']
    # 使用 .get() 获取选项内容，如果不存在则为 None
    original_options_text = [row.get(opt) for opt in options]

    # 如果有效选项少于2个，无法进行有意义的评测，直接返回失败
    if sum(1 for opt in original_options_text if pd.notna(opt)) < 2:
        return {'index': row['index'], 'ground_truth': row['answer'], 'circular_eval_passed': False, 'details': 'Not enough options for circular evaluation'}

    question_base = row['question'] + (" " + str(row.get('hint', '')) if pd.notna(row.get('hint')) else "")
    all_correct = True
    details = []

    for i in range(4):  # 4次循环平移
        shifted_options = original_options_text[i:] + original_options_text[:i]

        # 答案的索引计算逻辑不变
        try:
            original_correct_idx = options.index(row['answer'])
        except ValueError:
            # 如果答案标签（如'A'）本身就有问题，则无法评测
            return {'index': row['index'], 'ground_truth': row['answer'], 'circular_eval_passed': False, 'details': f"Invalid answer key '{row['answer']}'"}

        new_correct_letter = options[(original_correct_idx - i + 4) % 4]

        # 动态构建 prompt，跳过内容为 None 或 NaN 的选项
        option_str = "\n".join([
            f"{opt}. {text}" for opt, text in zip(options, shifted_options) if pd.notna(text)
        ])

        prompt = (f"{question_base}\n" +
                  option_str + "\n" +
                  "Please answer with the letter of the correct option.")

        raw_pred = _post_request(session, server_address, prompt, row['image'])
        pred_letter = _process_mmbench_response(raw_pred)
        details.append({'shift': i, 'ground_truth': new_correct_letter, 'prediction': pred_letter})

        if pred_letter != new_correct_letter:
            all_correct = False
            break  # 提前退出

    return {'index': row['index'], 'ground_truth': row['answer'], 'circular_eval_passed': all_correct, 'details': details}


def _evaluate_single_mmbench_regular(session: requests.Session, server_address: str, row: Dict[str, Any]) -> Dict[str, Any]:
    """对单个 MMBench 题目进行常规评测"""
    options = ['A', 'B', 'C', 'D']
    # 动态构建选项字符串，只包括存在的选项
    option_str = "\n".join([
        f"{opt}. {row[opt]}" for opt in options if opt in row and pd.notna(row[opt])
    ])

    prompt = (f"{row['question']}" +
              (" " + str(row.get('hint', '')) if pd.notna(row.get('hint')) else "") + "\n" +
              option_str + "\n" +  # 使用动态生成的选项字符串
              "Please answer with the letter of the correct option.")

    raw_pred = _post_request(session, server_address, prompt, row['image'])
    pred_letter = _process_mmbench_response(raw_pred)

    return {'index': row['index'], 'ground_truth': row['answer'], 'prediction': pred_letter, 'raw_prediction': raw_pred, 'correct': pred_letter == row['answer']}


def evaluate_mmbench(samples: List[Dict[str, Any]], server_address: str, api_nproc: int, circular: bool) -> List[Dict[str, Any]]:
    """处理选择题 (MMBench)，支持常规和循环两种评测模式"""
    results = []
    eval_func = _evaluate_single_mmbench_circular if circular else _evaluate_single_mmbench_regular

    with ThreadPoolExecutor(max_workers=api_nproc) as executor:
        with requests.Session() as session:
            future_to_sample = {executor.submit(eval_func, session, server_address, s): s for s in samples}
            for future in tqdm(as_completed(future_to_sample), total=len(samples), desc=f"Evaluating MMBench (circular={circular})"):
                sample = future_to_sample[future]
                try:
                    results.append(future.result())
                except Exception as e:
                    traceback.print_exc()
                    results.append({
                        'index': sample.get('index', 'N/A'),
                        'question': sample.get('question', 'N/A'),
                        'ground_truth': sample.get('answer', 'N/A'),
                        'prediction': 'error',
                        'raw_prediction': str(e)
                    })
    return results


def _evaluate_single_seedbench(session: requests.Session, server_address: str, row: Dict[str, Any]) -> Dict[str, Any]:
    """对单个 SEEDBench 题目进行评测"""
    options = ['A', 'B', 'C', 'D']
    # 动态构建选项字符串，只包括存在的选项
    option_str = "\n".join([
        f"{opt}. {row[opt]}" for opt in options if opt in row and pd.notna(row[opt])
    ])

    prompt = (f"{row['question']}\n" +
              option_str + "\n" +
              "Please answer with the letter of the correct option.")

    raw_pred = _post_request(session, server_address, prompt, row['image'])
    pred_letter = _process_mmbench_response(raw_pred)

    return {
        'index': row['index'],
        'category': row['category'],  # SEEDBench 需要 category 来计算分组 acc
        'ground_truth': row['answer'],
        'prediction': pred_letter,
        'raw_prediction': raw_pred,
        'correct': pred_letter == row['answer']
    }


def evaluate_seedbench(samples: List[Dict[str, Any]], server_address: str, api_nproc: int) -> List[Dict[str, Any]]:
    """处理选择题 (SEEDBench)"""
    results = []
    eval_func = _evaluate_single_seedbench

    with ThreadPoolExecutor(max_workers=api_nproc) as executor:
        with requests.Session() as session:
            # 提交所有任务
            future_to_sample = {executor.submit(eval_func, session, server_address, s): s for s in samples}

            # 使用 tqdm 监视进度
            for future in tqdm(as_completed(future_to_sample), total=len(samples), desc="Evaluating SEEDBench_IMG"):
                sample = future_to_sample[future]
                try:
                    results.append(future.result())
                except Exception as e:
                    traceback.print_exc()
                    results.append({
                        'index': sample.get('index', 'N/A'),
                        'category': sample.get('category', 'N/A'),
                        'question': sample.get('question', 'N/A'),
                        'ground_truth': sample.get('answer', 'N/A'),
                        'prediction': 'error',
                        'raw_prediction': str(e)
                    })
    return results


def print_report(results: List[Dict[str, Any]], task: str, df: pd.DataFrame) -> Optional[Dict[str, Any]]:
    """打印最终的评测报告, df 为原始加载的数据, 并返回一个包含主要指标的字典"""
    print("\n" + "=" * 20 + f" REPORT FOR: {task.upper()} " + "=" * 20)
    summary = {}  # 用于保存最终结果的字典
    if not results:
        print("No results to report.")
        return None

    if task in ['MME', 'POPE']:
        labels = [r['ground_truth'] for r in results if 'ground_truth' in r]
        predictions = [r['prediction'] for r in results if 'prediction' in r]
        valid_indices = [i for i, p in enumerate(predictions) if p in ['yes', 'no']]

        if not valid_indices:
            print("No valid ('yes'/'no') predictions found.")
        else:
            valid_labels = [labels[i] for i in valid_indices]
            valid_preds = [predictions[i] for i in valid_indices]
            # 计算准确率
            correct_count = sum(1 for gt, pred in zip(valid_labels, valid_preds) if gt == pred)
            accuracy = correct_count / len(valid_preds) if valid_preds else 0.0
            # 使用自定义函数计算指标
            metrics = calculate_yes_no_metrics(valid_labels, valid_preds, pos_label='yes')
            print(f"Total Samples: {len(results)}, Valid Predictions: {len(valid_preds)}")
            print(f"Accuracy: {accuracy:.4f}")
            print(f"Precision (for 'yes'): {metrics['precision']:.4f}")
            print(f"Recall (for 'yes'): {metrics['recall']:.4f}")
            print(f"F1-Score (for 'yes'): {metrics['f1']:.4f}")
            summary = {
                "task": task,
                "total_samples": len(results),
                "valid_predictions": len(valid_preds),
                "accuracy": accuracy,
                "precision_yes": metrics['precision'],
                "recall_yes": metrics['recall'],
                "f1_score_yes": metrics['f1']
            }

        # --- MME 特有的指标计算 ---
        if task == 'MME':
            print("\n--- MME Specific Metrics (acc, acc+) ---")
            results_df = pd.DataFrame(results)

            # 确保 'index' 列是数值类型以便合并
            results_df['index'] = pd.to_numeric(results_df['index'])
            df['index'] = pd.to_numeric(df['index'])

            merged_df = pd.merge(results_df, df[['index', 'category']], on='index', how='left')

            merged_df.sort_values('index', inplace=True)
            # 判断成对的问题是否都答对了
            is_pair_correct = merged_df['correct'] & merged_df['correct'].shift(-1)
            merged_df['pair_correct'] = False
            merged_df.loc[merged_df['index'] % 2 == 0, 'pair_correct'] = is_pair_correct[merged_df['index'] % 2 == 0]

            # 按类别计算 acc 和 acc+
            category_metrics_df = merged_df.groupby('category').apply(lambda g: pd.Series({
                'acc': g['correct'].mean(),
                'acc+': g['pair_correct'].sum() / (len(g) / 2)
            }))
            category_metrics = category_metrics_df.to_dict('index')

            total_acc, total_acc_plus = 0, 0
            for category, cat_metrics in sorted(category_metrics.items()):
                print(f"  - {category:<20s}: Acc: {cat_metrics['acc']:.4f}, Acc+: {cat_metrics['acc+']:.4f}")
                total_acc += cat_metrics['acc']
                total_acc_plus += cat_metrics['acc+']

            num_categories = len(category_metrics)
            overall_acc = total_acc / num_categories if num_categories > 0 else 0
            overall_acc_plus = total_acc_plus / num_categories if num_categories > 0 else 0
            print("-" * 42)
            print(f"  - {'Overall (Avg of Cats)':<20s}: Acc: {overall_acc:.4f}, Acc+: {overall_acc_plus:.4f}")

            # 更新摘要
            summary['mme_overall_acc'] = overall_acc
            summary['mme_overall_acc_plus'] = overall_acc_plus
            summary['mme_category_metrics'] = category_metrics

            # --- 计算MME-perception指标 ---
            # 定义与Perception相关的类别
            perception_categories = [
                'existence', 'count', 'position', 'color', 'posters',
                'celebrity', 'scene', 'landmark', 'artwork', 'OCR'
            ]
            perception_categories_lower = [cat.lower() for cat in perception_categories]  # 转为小写以实现不区分大小写的匹配

            perception_acc_sum = 0.0
            perception_acc_plus_sum = 0.0
            # 累加所有Perception类别的acc和acc+分数
            for category, cat_metrics in category_metrics.items():
                if category.lower() in perception_categories_lower:
                    perception_acc_sum += cat_metrics.get('acc', 0.0)
                    perception_acc_plus_sum += cat_metrics.get('acc+', 0.0)

            # 计算最终的MME Perception分数
            mme_perception_score = 100 * (perception_acc_sum + perception_acc_plus_sum)
            summary['mme_perception_score'] = mme_perception_score

            print("\n--- MME Perception Score ---")
            print(f"  - Score (SUM(acc) + SUM(acc+)) * 100: {mme_perception_score:.2f}")


    elif 'MMBench' in task:
        is_circular = results and 'circular_eval_passed' in results[0]
        key = 'circular_eval_passed' if is_circular else 'correct'
        correct_count = sum(1 for r in results if r.get(key, False))
        accuracy = correct_count / len(results) if results else 0

        print(f"Mode: {'Circular' if is_circular else 'Regular'} Evaluation")
        print(f"Total Samples: {len(results)}, Correctly Answered: {correct_count}")
        print(f"Accuracy: {accuracy:.4f}")

        summary = {
            "task": task,
            "mode": 'Circular' if is_circular else 'Regular',
            "total_samples": len(results),
            "correctly_answered": correct_count,
            "accuracy": accuracy
        }

    elif task == 'SEEDBench_IMG':
        # 将结果转为 DataFrame，方便按类别分组
        results_df = pd.DataFrame(results)
        category_acc = results_df.groupby('category')['correct'].mean().to_dict()

        print("--- SEEDBench Category Accuracy ---")
        total_acc_sum = 0
        num_categories = len(category_acc)

        # 打印每个类别的准确率
        for category, acc in sorted(category_acc.items()):
            print(f"  - {category:<30s}: Acc: {acc:.4f}")
            total_acc_sum += acc

        # 计算总分，即所有类别准确率的均值
        overall_score = total_acc_sum / num_categories if num_categories > 0 else 0
        print("-" * 42)
        print(f"Overall Score (Average of Categories): {overall_score:.4f}")

        summary = {
            "task": task,
            "total_samples": len(results),
            "overall_score": overall_score,
            "category_accuracy": category_acc
        }

    print("=" * 58 + "\n")
    return summary


def main(task: str, base_path: str = "~/LMUData/",
         server_address: str = "http://localhost:33218/v1/chat/completions",
         log_path: str = './outputs/eval_vqa_res/',
         api_nproc: int = 100,
         circular: bool = False,
         save_name: str = 'untitled'):
    """
    VQA 评测脚本

    Args:
        task (str): 要评测的任务, 可选: 'MME', 'POPE', 'MMBench_DEV_EN', 'MMBench_DEV_CN', 'SEEDBench_IMG'.
        base_path (str): 数据集所在的根目录.
        server_address (str): 模型 API 服务器地址.
        log_path (str): 保存预测结果的目录.
        api_nproc (int): 并发请求数量.
        circular (bool): 是否对 MMBench 进行 Circular 评测.
        save_name (str): 保存文件时使用的自定义名称.
    """
    # 准备路径和日志文件
    os.makedirs(log_path, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(log_path, f"{task}_{timestamp}_{save_name}.jsonl")

    # 加载数据
    file_path = Path(base_path).expanduser() / f"{task}.tsv"
    if not file_path.exists():
        raise FileNotFoundError(f"Data file not found at: {file_path}")
    print(f"Loading data from {file_path}...")

    # 使用 pandas 读取 TSV 文件
    df = pd.read_csv(file_path, sep='\t')

    # 针对 MME 任务的特殊数据预处理：配对图片
    if task == 'MME':
        print("MME task: preprocessing paired image data...")
        # 为偶数行创建 index -> image base64 的映射
        image_map = df[df['index'] % 2 == 0].set_index('index')['image'].to_dict()

        # 定义一个函数来为奇数行查找其配对的图片
        def get_paired_image(row):
            if row['index'] % 2 != 0:
                try:  # 奇数行的 image 列存的是偶数行的 index
                    return image_map.get(int(row['image']))
                except (ValueError, TypeError):
                    return None  # 如果 image 列不是合法的 index，返回 None
            return row['image']  # 偶数行直接返回自己的图片

        # 使用 tqdm.pandas 显示进度
        tqdm.pandas(desc="Pairing images for MME")
        df['image'] = df.progress_apply(get_paired_image, axis=1)

        # 检查并移除无法找到配对图片的行
        original_len = len(df)
        df.dropna(subset=['image'], inplace=True)
        if len(df) < original_len:
            print(f"Warning: Dropped {original_len - len(df)} rows with missing paired images.")

    # 针对 MMBench 任务的特殊数据预处理：解析引用的图片
    elif 'MMBench' in task:
        print("MMBench task: preprocessing image references...")
        # 筛选出 image 列是 base64 字符串的行（通过长度判断，索引通常不会这么长）
        # 并创建 index -> image base64 的映射
        base64_rows = df[df['image'].str.len() > 100]
        image_map = base64_rows.set_index('index')['image'].to_dict()

        # 定义一个函数来查找引用的图片
        def get_referenced_image(row):
            # 尝试将 image 列的内容转为整数，如果成功，说明是索引
            try:
                ref_index = int(row['image'])
                return image_map.get(ref_index)
            # 如果转换失败，说明它本身就是 base64 字符串
            except (ValueError, TypeError):
                return row['image']

        # 使用 tqdm.pandas 显示进度
        tqdm.pandas(desc="Resolving image references for MMBench")
        df['image'] = df.progress_apply(get_referenced_image, axis=1)

        # 检查并移除无法找到引用图片的行
        original_len = len(df)
        df.dropna(subset=['image'], inplace=True)
        if len(df) < original_len:
            print(f"Warning: Dropped {original_len - len(df)} rows with missing referenced images.")


    # 将 DataFrame 转换为字典列表，以兼容后续处理函数
    samples = df.to_dict('records')

    print(f"Loaded {len(samples)} samples.")

    # 根据任务类型执行评测
    results = []
    if task in ['MME', 'POPE']:
        results = evaluate_yes_no(samples, server_address, api_nproc, task)
    elif 'MMBench' in task:
        results = evaluate_mmbench(samples, server_address, api_nproc, circular)
    elif task == 'SEEDBench_IMG':
        results = evaluate_seedbench(samples, server_address, api_nproc)
    else:
        raise ValueError(f"Unknown task: {task}. Supported tasks are 'MME', 'POPE', 'MMBench_DEV_EN', 'MMBench_DEV_CN', 'SEEDBench_IMG'.")

    # 保存中间预测结果
    if results:
        with open(output_file, 'w', encoding='utf-8') as f:
            for res in results:
                f.write(json.dumps(res, ensure_ascii=False) + '\n')
        print(f"Intermediate results saved to {output_file}")

    # 打印评测报告
    eval_summary = print_report(results, task, df)

    # 保存最终的评测结果摘要
    if eval_summary:
        # 基于输出文件名创建摘要文件名
        summary_file = output_file.replace('.jsonl', '_summary.json')
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(eval_summary, f, ensure_ascii=False, indent=4)
        print(f"Evaluation summary saved to {summary_file}")


if __name__ == "__main__":
    Fire(main)