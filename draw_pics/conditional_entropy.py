import os

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import io
import re

data = '''
English, n=1 entropy = 10.8332 bits
English, n=2 entropy = 6.9576 bits
English, n=3 entropy = 4.1062 bits
English, n=4 entropy = 1.8410 bits
English, n=5 entropy = 0.7664 bits
English, n=6 entropy = 0.3536 bits
English, n=7 entropy = 0.1824 bits
English, n=8 entropy = 0.0923 bits
English, n=9 entropy = 0.0444 bits
German, n=1 entropy = 10.6777 bits
German, n=2 entropy = 6.3728 bits
German, n=3 entropy = 3.9527 bits
German, n=4 entropy = 2.1524 bits
German, n=5 entropy = 1.0638 bits
German, n=6 entropy = 0.5098 bits
German, n=7 entropy = 0.2360 bits
German, n=8 entropy = 0.1117 bits
German, n=9 entropy = 0.0600 bits
Chinese, n=1 entropy = 11.4892 bits
Chinese, n=2 entropy = 7.9386 bits
Chinese, n=3 entropy = 3.9195 bits
Chinese, n=4 entropy = 1.3671 bits
Chinese, n=5 entropy = 0.5106 bits
Chinese, n=6 entropy = 0.2215 bits
Chinese, n=7 entropy = 0.1112 bits
Chinese, n=8 entropy = 0.0651 bits
Chinese, n=9 entropy = 0.0428 bits
Image, n=1 entropy = 11.0564 bits
Image, n=2 entropy = 10.2860 bits
Image, n=3 entropy = 7.3884 bits

'''

'''backup
Image, n=1 entropy = 11.0564 bits
Image, n=2 entropy = 10.2860 bits
Image, n=3 entropy = 3.4166 bits
Image(num_img=1e6), n=3 entropy = 7.3884 bits
Image, n=4 entropy = 0.2780 bits
Image, n=5 entropy = 0.1498 bits
Image, n=6 entropy = 0.0942 bits
Image, n=7 entropy = 0.0618 bits
Image, n=8 entropy = 0.0427 bits
Image, n=9 entropy = 0.0312 bits
'''

# 使用正则表达式解析数据，更健壮
pattern = re.compile(r'(.+), n=(\d+) entropy = ([\d.]+) bits')
parsed_data = []
for line in data.strip().split('\n'):
    match = pattern.match(line)
    if match:
        source, n, entropy = match.groups()
        if int(n) > 5: continue
        parsed_data.append({
            'source': source.strip(),
            'n': int(n),
            'entropy': float(entropy)
        })

# 创建DataFrame
df = pd.DataFrame(parsed_data)

# 设置绘图风格
sns.set_style("whitegrid")

# 定义调色板以区分不同数据源
palette = {
    'English': 'cornflowerblue',
    'German': 'royalblue',
    'Chinese': 'skyblue',
    'Image': 'crimson'
}

# 创建图表，设置大小
plt.figure(figsize=(3, 4))

# 绘制线图，按 source 对数据进行分组
sns.lineplot(data=df, x='n', y='entropy', hue='source', marker='o', palette=palette)

# 添加标题和标签
# plt.title('Conditional Entropy vs. N-gram Size')
plt.xlabel('N-gram (n)')
plt.ylabel('Conditional Entropy (bits)')
plt.legend(title='Data Source')
# plt.yscale('log')

os.makedirs('draw_pics/pics/', exist_ok=True)
plt.savefig('draw_pics/pics/conditional_entropy.pdf', bbox_inches='tight')

# 显示图形
plt.show()