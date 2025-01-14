import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
 
data = np.array(
    [
    [66, 521], # [34,521], # fa3, 2k,dim128, causal
    # [25,355], # sigmoid, 2k,dim128, causal
    [69,0], # torch compile # [37,0], # relu causal
    # [1/4.93,1/0.77] # mamba2 fwd, 8,24,2k
    [1/1.13, 1/0.39] # gated retnet, 1,32,2048,256,512
    ]
)
data = data / data[:,0][:,np.newaxis]
 
# 数据
categories = ['Softmax-Attention','ReLU-Attention','Gated-RetNet']
 
# 条形图宽度
bar_width = 0.2
index = np.arange(len(categories))
 
# 绘制条形图
# fig, ax0 = plt.subplots()

fig = plt.figure(figsize=(4, 2))
gs = gridspec.GridSpec(1, 1, figure=fig, height_ratios=[1], wspace=0.3, hspace=0.6)
gss = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[0, 0:12], hspace=0.25)

ax0 = fig.add_subplot(gss[0])
ax1 = fig.add_subplot(gss[1])

ax0.set_ylim(7.5, 10.5)  # 上面的图为10到最大值
ax1.set_ylim(0, 3)  # 下面的图为0到5

ax0.axhline(y=1, color="black", linestyle="dashed")
ax0.spines["bottom"].set_visible(False)
ax0.set_xticklabels([])
ax0.set_xticks([])

ax1.spines["top"].set_visible(False)

bars0_0 = ax0.bar(index, data[:,0], bar_width, label='PyTorch', color='lightblue', edgecolor='black', hatch='//')
bars0_1 = ax0.bar(index + bar_width, data[:,1], bar_width, label='Library (Flash-Attention, Gated-RetNet)', color='lightyellow', edgecolor='black', hatch='-')
bars1_0 = ax1.bar(index, data[:,0], bar_width, label='PyTorch', color='lightblue', edgecolor='black', hatch='//')
bars1_1 = ax1.bar(index + bar_width, data[:,1], bar_width, label='Library (Flash-Attention, Gated-RetNet)', color='lightyellow', edgecolor='black', hatch='-')
for bs in bars1_0 + bars1_1:
    height = bs.get_height()
    warning_text = "Not supported"
    if height == 0:
        ax1.text(
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
        ax1.text(
            bs.get_x() + bs.get_width() / 2 - 0.03,
            height + 0.05,
            f'{height:.2f}x',
            ha='center',
            va='bottom',
            fontsize=8
        )

for bs in bars0_0 + bars0_1:
    height = bs.get_height()
    print(height)
    warning_text = "Not supported"
    if height > 7.5:
        ax0.text(
            bs.get_x() + bs.get_width() / 2 - 0.03,
            height + 0.05,
            f'{height:.2f}x',
            ha='center',
            va='bottom',
            fontsize=8
        )
 
d = 0.01  # 斜线的长度
kwargs = dict(transform=ax0.transAxes, color="k", clip_on=False)
ax0.plot((1 - d, 1 + d), (-d, +d), **kwargs)  # top-left diagonal
ax0.plot((-d, +d), (-d, +d), **kwargs)  # top-right diagonal


kwargs.update(transform=ax1.transAxes)  # switch to the bottom axes
ax1.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)  # bottom-left diagonal
ax1.plot((-d, +d), (1 - d, 1 + d), **kwargs)  # bottom-right diagonal


# 添加标签和标题
ax1.set_xlabel(' ')
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
ax1.set_xticks(index + bar_width * 0.5)
ax1.set_xticklabels(categories)
ax0.legend(loc='upper center', bbox_to_anchor=(0.5, 1.0), ncol=1, fontsize=8)
 
# 显示图表
plt.tight_layout()
# plt.show()
# plt.savefig('figure_intro.png')
plt.savefig(
    "figure_intro.pdf",
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