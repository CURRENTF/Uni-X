import pandas as pd
import matplotlib.pyplot as plt
import os
from tqdm import tqdm


# baseline_file: draw_pics/analysis_data/baseline_full_analysis_results.csv
# uni-x (ours): draw_pics/analysis_data/unix_full_analysis_results.csv

def plot_grad_conflict():
    # define file paths
    baseline_file = 'draw_pics/analysis_data/baseline_full_analysis_results.csv'
    unix_file = 'draw_pics/analysis_data/unix_full_analysis_results.csv'
    output_dir = 'draw_pics/pics/grad_conflict'

    # ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # read data into pandas DataFrames
    df_baseline = pd.read_csv(baseline_file)
    df_unix = pd.read_csv(unix_file)

    # calculate gradient conflict as the negative of similarity
    df_baseline['grad_conflict'] = -df_baseline['similarity']
    df_unix['grad_conflict'] = -df_unix['similarity']

    # group data by checkpoint and group_name
    grouped_baseline = df_baseline.groupby(['checkpoint_id', 'group_name'])
    grouped_unix = df_unix.groupby(['checkpoint_id', 'group_name'])

    # iterate over each group in the unix data to plot
    print("Generating individual plots for each checkpoint and group...")
    for (checkpoint_id, group_name), unix_group in tqdm(grouped_unix, desc="Generating individual plots"):
        # find the corresponding baseline group
        try:
            baseline_group = grouped_baseline.get_group((checkpoint_id, group_name))
        except KeyError:
            continue  # 如果 baseline 中没有对应的组，则跳过

        # sort by layer_idx to ensure correct line plotting
        unix_group = unix_group.sort_values('layer_idx')
        baseline_group = baseline_group.sort_values('layer_idx')

        # create plot
        plt.figure(figsize=(6, 4))
        plt.plot(baseline_group['layer_idx'], baseline_group['grad_conflict'], label='Baseline', marker='o', linestyle='-')
        plt.plot(unix_group['layer_idx'], unix_group['grad_conflict'], label='Uni-X (Ours)', marker='x', linestyle='--')

        # set plot labels
        plt.xlabel('Layer Index')
        plt.ylabel('Gradient Conflict')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)

        # save the figure to a PDF file
        output_filename = f"{checkpoint_id}---{group_name}-grad-conflict.pdf"
        plt.savefig(os.path.join(output_dir, output_filename), bbox_inches='tight')
        plt.close()  # close the plot to free up memory

    print(f"Generated {len(grouped_unix)} plots in '{output_dir}'.")
    # TODO 对每个 group name，画带方差的折线图
    print("\nGenerating aggregated plots with variance for each group name...")
    # 获取所有唯一的 group_name
    all_group_names = df_unix['group_name'].unique()

    # 遍历每个 group_name，聚合所有 checkpoint 的数据进行绘图
    for group_name in tqdm(all_group_names, desc="Generating aggregated plots"):
        # 筛选当前 group_name 的数据
        df_baseline_group = df_baseline[df_baseline['group_name'] == group_name]
        df_unix_group = df_unix[df_unix['group_name'] == group_name]

        # 按层索引(layer_idx)分组，计算梯度冲突的均值和标准差
        baseline_agg = df_baseline_group.groupby('layer_idx')['grad_conflict'].agg(['mean', 'std']).reset_index()
        unix_agg = df_unix_group.groupby('layer_idx')['grad_conflict'].agg(['mean', 'std']).reset_index()

        # 创建新图像
        plt.figure(figsize=(4, 3))

        # 绘制 Baseline 的均值折线和标准差范围
        plt.plot(baseline_agg['layer_idx'], baseline_agg['mean'], label='Baseline', linestyle='-')
        plt.fill_between(baseline_agg['layer_idx'], baseline_agg['mean'] - baseline_agg['std'], baseline_agg['mean'] + baseline_agg['std'], alpha=0.2)

        # 绘制 Uni-X (Ours) 的均值折线和标准差范围
        plt.plot(unix_agg['layer_idx'], unix_agg['mean'], label='Uni-X (Ours)', linestyle='--')
        plt.fill_between(unix_agg['layer_idx'], unix_agg['mean'] - unix_agg['std'], unix_agg['mean'] + unix_agg['std'], alpha=0.2)

        # 设置图像标签和标题
        plt.xlabel('Layer Index')
        plt.ylabel('Gradient Conflict')
        # plt.title(f'Aggregated Gradient Conflict: {group_name}')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)

        # 保存图像到 PDF 文件
        output_filename = f"{group_name}-aggregated-grad-conflict.pdf"
        plt.savefig(os.path.join(output_dir, output_filename), bbox_inches='tight')
        plt.close()  # 关闭图像以释放内存

    print(f"Generated {len(all_group_names)} aggregated plots with variance in '{output_dir}'.")


if __name__ == '__main__':
    plot_grad_conflict()
