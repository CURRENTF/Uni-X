import os
import json
import glob

# 输出文件名
output_filename = 'dpg_prompts_en.jsonl'
# 查找当前目录下所有的 .txt 文件
txt_files = glob.glob('prompts/*.txt')

print(f"找到了 {len(txt_files)} 个 .txt 文件，将开始处理...")

# 使用 'w' 模式打开文件，如果文件已存在则会覆盖
with open(output_filename, 'w', encoding='utf-8') as outfile:
    for filename in txt_files:
        try:
            # 读取每个 txt 文件的内容
            with open(filename, 'r', encoding='utf-8') as infile:
                content = infile.read()

            # 创建符合格式的字典
            data = {
                "Index": filename,
                "Prompt": content
            }

            # 将字典转换为 JSON 字符串（ensure_ascii=False 保证中文等字符正常显示）
            # 然后在末尾添加换行符，以符合 jsonl 格式
            outfile.write(json.dumps(data, ensure_ascii=False) + '\n')

        except Exception as e:
            print(f"处理文件 {filename} 时出错: {e}")

print(f"所有文件处理完毕，结果已保存到 {output_filename}")