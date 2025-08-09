import pandas as pd
import matplotlib.pyplot as plt


def plot_fid_curves(csv1, csv2, labels=("CSV1", "CSV2")):

    df1 = pd.read_csv(csv1)
    df2 = pd.read_csv(csv2)


    plt.figure(figsize=(8,6))
    plt.plot(df1['step'], df1['fid'], label=labels[0], linewidth=2, marker='o', markersize=6)
    plt.plot(df2['step'], df2['fid'], label=labels[1], linewidth=2, marker='s', markersize=6)

    # 图形美化
    plt.xlabel('Training Step', fontsize=16)
    plt.ylabel('FID', fontsize=16)
    plt.title('FID vs Training Step', fontsize=18)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)

    plt.legend(fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()

    # 保存图片
    plt.savefig('fid_curves_lora_vs_nolora.png', dpi=500)
    plt.close()


plot_fid_curves('/scratch/leuven/375/vsc37593/finetune_expr_v2/nolora_with_camera_1e6lr/fid_scores.csv', 
                '/scratch/leuven/375/vsc37593/finetune_expr_v2/lora_with_camera/fid_scores.csv', 
                labels=("Full finetuning", "Attention layers only"))