import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
 
data = np.array(
    [
    [1/554, 1/449], # deepseek v2
    [1/5041,1/2733], # diff attn
    [1/1764, 1/1536], # mamba2
    [1/35,1/27], # diff attn
    ]
)
data = data / data[:,0][:,np.newaxis]
 
# 数据
categories = ['Deepseek-V2-Lite','Diff-Transformer-3B', 'Mamba2-2.7B', 'YOCO-160M']
 
# 条形图宽度
bar_width = 0.3
index = np.arange(len(categories))
 
# 绘制条形图
# fig, ax0 = plt.subplots()

fig = plt.figure(figsize=(6, 3))
gs = gridspec.GridSpec(1, 1, figure=fig, height_ratios=[1], wspace=0.3, hspace=0.9)
ax0 = fig.add_subplot(gs[0])

ax0.set_ylim(0.5, 2.3)  # 上面的图为10到最大值

ax0.set_xticklabels([])
ax0.set_xticks([])

bars0_0 = ax0.bar(index, data[:,0], bar_width, label='PyTorch+AttnForge', color='lightblue', edgecolor='black', hatch='//')
bars0_1 = ax0.bar(index + bar_width, data[:,1], bar_width, label='PyTorch', color='lightyellow', edgecolor='black', hatch='-')
for bs in bars0_0 + bars0_1:
    height = bs.get_height()
    warning_text = "Not supported"
    if height == 0:
        ax0.text(
            bs.get_x() + bs.get_width() / 2,
            height + 0.05,
            warning_text,
            ha="center",
            va="bottom",
            fontsize=9,
            rotation=90,
            color="red",
            weight="bold",
        )
    else:
        ax0.text(
            bs.get_x() + bs.get_width() / 2 - 0.03,
            height + 0.05,
            f'{height:.2f}x',
            ha='center',
            va='bottom',
            fontsize=8
        )
 
# 添加标签和标题
ax0.set_xlabel(' ')
# ax1.set_ylabel('Normalized Speedup\nvs. PyTorch')
fig.text(
    0.02,
    0.5,
    'Normalized Speedup\nvs. PyTorch',
    fontsize=11,
    rotation=90,
    va="center",
    ha="center",
)
# ax0.set_title('batch size=8, seqlen-2048, head dimention=128')
ax0.set_xticks(index + bar_width * 0.5)
ax0.set_xticklabels(categories, fontsize=9)
ax0.legend(loc='upper center', bbox_to_anchor=(0.5, 1.0), ncol=2, fontsize=10)
fig.suptitle("End-to-end Inference Performance", fontsize=14)

# 显示图表
plt.tight_layout()
# plt.show()
# plt.savefig('figure_intro.png')
plt.savefig(
    "e2e_inference.pdf",
    bbox_inches="tight",
)
 
# fig.legend(
#     handles_other,
#     labels_other,
#     loc="upper center",
#     bbox_to_anchor=(0.45, 0.980 - 0.02),
#     ncol=len(labels_other),
#     fontsize=10,
#     frameon=True,
# )