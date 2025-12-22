#!/bin/bash

# 检查参数
if [ $# -eq 0 ]; then
    echo "用法: $0 <图片文件夹路径>"
    exit 1
fi

# 获取输入文件夹路径
input_dir="$1"

# 检查输入文件夹是否存在
if [ ! -d "$input_dir" ]; then
    echo "错误: 文件夹 '$input_dir' 不存在"
    exit 1
fi

basename_dir=$(basename "$input_dir")
temp_folder="for_eval_${basename_dir}"
# 创建final文件夹
mkdir -p "${temp_folder}"
# 遍历输入文件夹中的png文件
for img_file in "$input_dir"/*.png; do
    # 检查文件是否存在（避免通配符没有匹配到文件的情况）
    if [ ! -f "$img_file" ]; then
        continue
    fi

    # 获取文件名（不含路径和扩展名）
    filename=$(basename "$img_file" .png)
    cp "$img_file" "${temp_folder}/${filename}.png"
#    for i in {0..3}; do
#        cp "$img_file" "${temp_folder}/${filename}.png"
#    done
done

tar -cf "./${temp_folder}.tar" "${temp_folder}"
rm -rf ${temp_folder}
