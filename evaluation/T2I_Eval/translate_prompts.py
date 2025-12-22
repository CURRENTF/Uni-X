import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Tuple, Union

from fire import Fire
from tqdm import tqdm

from tools.data_translator import translate_text_api


def translate_task(line_data: Dict, source_language: str, target_language: str) -> Union[Dict, None]:
    """
    单个翻译任务。
    """
    index = line_data.get("Index")
    prompt_to_translate = line_data.get("Prompt")

    if not prompt_to_translate:
        return None

    try:
        # 调用API进行翻译
        translated_text, _ = translate_text_api(
            text=prompt_to_translate,
            source_language=source_language,
            target_language=target_language,
            max_tokens=500
        )
        return {"Index": index, "Prompt": translated_text}
    except Exception as e:
        # 捕获翻译过程中可能出现的任何异常
        print(f"翻译出错，错误为: {e}")
        return None


def main(data_path: str, save_path: str, source_language: str = 'en', target_language: str = 'zh'):
    """
    使用多线程翻译JSONL文件中的 "Prompt" 字段。

    Args:
        data_path (str): 输入的jsonl文件路径。
        save_path (str): 保存翻译结果的jsonl文件路径。
        source_language (str): 源语言代码，默认为 'en'。
        target_language (str): 目标语言代码，默认为 'zh'。
    """
    # 从输入文件中读取所有行
    if data_path.endswith("jsonl"):
        data_mode = 'jsonl'
        with open(data_path, 'r', encoding='utf-8') as f:
            tasks = [json.loads(line) for line in f if line.strip()]
    elif data_path.endswith('txt'):
        data_mode = 'txt'
        with open(data_path, 'r', encoding='utf-8') as f:
            tasks = [{'Index': i, 'Prompt': line} for i, line in enumerate(f) if line.strip()]
    else:
        raise ValueError('data path only in [jsonl, txt]')

    # 使用线程池并发执行翻译任务
    with ThreadPoolExecutor(max_workers=90) as executor, \
         open(save_path, 'w', encoding='utf-8') as f_out:

        # 提交所有任务到线程池
        futures = [executor.submit(translate_task, task, source_language, target_language) for task in tasks]

        # 使用tqdm显示进度条，并在任务完成时处理结果
        txt_results = []
        for future in tqdm(as_completed(futures), total=len(tasks), desc="Translating"):
            result = future.result()
            if result is not None:
                # 将结果以json格式写入输出文件
                if data_mode == 'jsonl':
                    _content = json.dumps(result, ensure_ascii=False)
                    f_out.write(_content + '\n')
                else:
                    txt_results.append(result)

        txt_results = [x['Prompt'] + '\n' for x in sorted(txt_results, key=lambda x: x["Index"])]
        f_out.writelines(txt_results)

    print(f"翻译完成，结果已保存至: {save_path}")


if __name__ == '__main__':
    Fire(main)