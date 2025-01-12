# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import matplotlib.ticker as ticker
import argparse
from matplotlib.ticker import MultipleLocator

colormap = plt.cm.Set2# LinearSegmentedColormap

from results import *
colers_sets = [
    # nilu
    (20 / 255, 54 / 255, 95 / 255),
    (248 / 255, 231 / 255, 210 / 255),
    # (118 / 255, 162 / 255, 185 / 255),
    (191 / 255, 217 / 255, 229 / 255),
    (214 / 255, 79 / 255, 56 / 255),
    (112 / 255, 89 / 255, 146 / 255),
    # dori
    (214 / 255, 130 / 255, 148 / 255),
    (169 / 255, 115 / 255, 153 / 255),
    (248 / 255, 242 / 255, 236 / 255),
    (214 / 255, 130 / 255, 148 / 255),
    (243 / 255, 191 / 255, 202 / 255),
    # (41/ 255, 31/ 255, 39/ 255),
    # coller
    # (72/ 255, 76/ 255, 35/ 255),
    (124 / 255, 134 / 255, 65 / 255),
    (185 / 255, 198 / 255, 122 / 255),
    (248 / 255, 231 / 255, 210 / 255),
    (182 / 255, 110 / 255, 151 / 255),
]

# colers_sets = colormap(np.linspace(0, 1, 12))

# 创建一个figure实例
fig = plt.figure(figsize=(16, 6))

# 获取Torch-Inductor的时间值
_1x_baseline = "AttentionFroge"

# 设置网格布局
gs = gridspec.GridSpec(3, 12, figure=fig, height_ratios=[1, 1, 1], wspace=0.3, hspace=0.6)

hatch_patterns = ["-", "+", "x", "\\", "*", "o", "O", "."]

legend_items = {}

llm_legands = []
other_legands = []


def get_legend_item(label):
    if label not in legend_items:
        idx = len(legend_items)
        legend_items[label] = (
            colers_sets[idx % len(colers_sets)],
            hatch_patterns[idx % len(hatch_patterns)],
        )
    return legend_items[label]


ax0 = fig.add_subplot(gs[0, 0:6])  # ResNet
providers = deepseek_bwd_providers
times_data = deepseek_bwd_times_data
_1x_baseline_times = dict(times_data)[_1x_baseline]
norm_time_data = []
for label, times in times_data:
    # if label != _1x_baseline:
    norm_time = [
        t / p_i if p_i != 0 else 0 for p_i, t in zip(_1x_baseline_times, times)
    ]
    norm_time_data.append((label, norm_time))

# Create an array for x-axis positions
x = np.arange(len(providers))

# Set the width of the bars
bar_width = 0.12

# Draw cublas as a horizontal dashed line
ax0.axhline(y=1, color="black", linestyle="dashed")

# Create bars using a loop
for i, (label, speedup) in enumerate(norm_time_data):
    if label not in other_legands:
        other_legands.append(label)
    ax0.bar(
        x + i * bar_width,
        speedup,
        bar_width,
        label=label,
        linewidth=0.8,
        edgecolor="black",
        hatch=get_legend_item(label)[1],
        color=get_legend_item(label)[0],
    )

# X-axis and labels
ax0.set_xticks(x + len(norm_time_data) * bar_width / 2)
ax0.set_xticklabels(providers)
ax0.grid(False)


ax1 = fig.add_subplot(gs[0, 6:12])  # ResNet
providers = llama_bwd_providers
times_data = llama_bwd_times_data
_1x_baseline_times = dict(times_data)[_1x_baseline]
norm_time_data = []
for label, times in times_data:
    # if label != _1x_baseline:
    norm_time = [
        t / p_i if p_i != 0 else 0 for p_i, t in zip(_1x_baseline_times, times)
    ]
    norm_time_data.append((label, norm_time))

# Create an array for x-axis positions
x = np.arange(len(providers))

# Set the width of the bars
bar_width = 0.12

# Draw cublas as a horizontal dashed line
ax1.axhline(y=1, color="black", linestyle="dashed")

# Create bars using a loop
for i, (label, speedup) in enumerate(norm_time_data):
    if label not in other_legands:
        other_legands.append(label)
    ax1.bar(
        x + i * bar_width,
        speedup,
        bar_width,
        label=label,
        linewidth=0.8,
        edgecolor="black",
        hatch=get_legend_item(label)[1],
        color=get_legend_item(label)[0],
    )

# X-axis and labels
ax1.set_xticks(x + len(norm_time_data) * bar_width / 2)
ax1.set_xticklabels(providers)
ax1.grid(False)

ax2 = fig.add_subplot(gs[1, 0:6])  # ResNet
providers = dit_bwd_providers
times_data = dit_bwd_times_data
_1x_baseline_times = dict(times_data)[_1x_baseline]
norm_time_data = []
for label, times in times_data:
    # if label != _1x_baseline:
    norm_time = [
        t / p_i if p_i != 0 else 0 for p_i, t in zip(_1x_baseline_times, times)
    ]
    norm_time_data.append((label, norm_time))

# Create an array for x-axis positions
x = np.arange(len(providers))

# Set the width of the bars
bar_width = 0.12

# Draw cublas as a horizontal dashed line
ax2.axhline(y=1, color="black", linestyle="dashed")

# Create bars using a loop
for i, (label, speedup) in enumerate(norm_time_data):
    if label not in other_legands:
        other_legands.append(label)
    ax2.bar(
        x + i * bar_width,
        speedup,
        bar_width,
        label=label,
        linewidth=0.8,
        edgecolor="black",
        hatch=get_legend_item(label)[1],
        color=get_legend_item(label)[0],
    )

# X-axis and labels
ax2.set_xticks(x + len(norm_time_data) * bar_width / 2)
ax2.set_xticklabels(providers)
ax2.grid(False)

ax3 = fig.add_subplot(gs[1, 6:10])  # ResNet
providers = sigmoid_bwd_providers
times_data = sigmoid_bwd_times_data
_1x_baseline_times = dict(times_data)[_1x_baseline]
norm_time_data = []
for label, times in times_data:
    # if label != _1x_baseline:
    norm_time = [
        t / p_i if p_i != 0 and t != 0 else 0
        for p_i, t in zip(_1x_baseline_times, times)
    ]
    norm_time_data.append((label, norm_time))

# Create an array for x-axis positions
x = np.arange(len(providers))

# Set the width of the bars
bar_width = 0.22

# Draw cublas as a horizontal dashed line
ax3.axhline(y=1, color="black", linestyle="dashed")

# Create bars using a loop
for i, (label, speedup) in enumerate(norm_time_data):
    if label not in other_legands:
        other_legands.append(label)
    rec = ax3.bar(
        x + i * bar_width,
        speedup,
        bar_width,
        label=label,
        linewidth=0.8,
        edgecolor="black",
        hatch=get_legend_item(label)[1],
        color=get_legend_item(label)[0],
    )
    for rect in rec:
        height = rect.get_height()
        if height == 0:
            warning_text = f"{label} Failed"
            ax3.text(
                rect.get_x() + rect.get_width() / 2 + 0.01,
                height + 0.05,
                warning_text,
                ha="center",
                va="bottom",
                fontsize=8,
                rotation=90,
                color="red",
                weight="bold",
            )

# X-axis and labels
ax3.set_xticks(x + len(norm_time_data) * bar_width / 2)
ax3.set_xticklabels(providers)
ax3.grid(False)

ax4 = fig.add_subplot(gs[1, 10:12])  # ResNet
providers = gla_bwd_providers
times_data = gla_bwd_times_data
_1x_baseline_times = dict(times_data)[_1x_baseline]
norm_time_data = []
for label, times in times_data:
    # if label != _1x_baseline:
    norm_time = [
        t / p_i if p_i != 0 else 0 for p_i, t in zip(_1x_baseline_times, times)
    ]
    norm_time_data.append((label, norm_time))

# Create an array for x-axis positions
x = np.arange(len(providers))

# Set the width of the bars
bar_width = 0.25

# Draw cublas as a horizontal dashed line
ax4.axhline(y=1, color="black", linestyle="dashed")

# Create bars using a loop
for i, (label, speedup) in enumerate(norm_time_data):
    if label not in other_legands:
        other_legands.append(label)
    ax4.bar(
        x + i * bar_width,
        speedup,
        bar_width,
        label=label,
        linewidth=0.8,
        edgecolor="black",
        hatch=get_legend_item(label)[1],
        color=get_legend_item(label)[0],
    )

# X-axis and labels
ax4.set_xticks(x + len(norm_time_data) * bar_width / 2)
ax4.set_xticklabels(providers)
ax4.grid(False)

ax5 = fig.add_subplot(gs[2, 0:3])  # ResNet
providers = relu_bwd_providers
times_data = relu_bwd_times_data
_1x_baseline_times = dict(times_data)[_1x_baseline]
norm_time_data = []
for label, times in times_data:
    # if label != _1x_baseline:
    norm_time = [
        t / p_i if p_i != 0 else 0 for p_i, t in zip(_1x_baseline_times, times)
    ]
    norm_time_data.append((label, norm_time))

# Create an array for x-axis positions
x = np.arange(len(providers))

# Set the width of the bars
bar_width = 0.25

# Draw cublas as a horizontal dashed line
ax5.axhline(y=1, color="black", linestyle="dashed")

# Create bars using a loop
for i, (label, speedup) in enumerate(norm_time_data):
    if label not in other_legands:
        other_legands.append(label)
    ax5.bar(
        x + i * bar_width,
        speedup,
        bar_width,
        label=label,
        linewidth=0.8,
        edgecolor="black",
        hatch=get_legend_item(label)[1],
        color=get_legend_item(label)[0],
    )

# X-axis and labels
ax5.set_xticks(x + len(norm_time_data) * bar_width / 2)
ax5.set_xticklabels(providers)
ax5.grid(False)

ax6 = fig.add_subplot(gs[2, 3:6])  # ResNet
providers = mamba_bwd_providers
times_data = mamba_bwd_times_data
_1x_baseline_times = dict(times_data)[_1x_baseline]
norm_time_data = []
for label, times in times_data:
    # if label != _1x_baseline:
    norm_time = [
        t / p_i if p_i != 0 else 0 for p_i, t in zip(_1x_baseline_times, times)
    ]
    norm_time_data.append((label, norm_time))

# Create an array for x-axis positions
x = np.arange(len(providers))

# Set the width of the bars
bar_width = 0.25

# Draw cublas as a horizontal dashed line
ax6.axhline(y=1, color="black", linestyle="dashed")

# Create bars using a loop
for i, (label, speedup) in enumerate(norm_time_data):
    if label not in other_legands:
        other_legands.append(label)
    ax6.bar(
        x + i * bar_width,
        speedup,
        bar_width,
        label=label,
        linewidth=0.8,
        edgecolor="black",
        hatch=get_legend_item(label)[1],
        color=get_legend_item(label)[0],
    )

# X-axis and labels
ax6.set_xticks(x + len(norm_time_data) * bar_width / 2)
ax6.set_xticklabels(providers)
ax6.grid(False)

# ax7 = fig.add_subplot(gs[2, 6:8])  # ResNet
# providers = retnet_bwd_providers
# times_data = retnet_bwd_times_data
# _1x_baseline_times = dict(times_data)[_1x_baseline]
# norm_time_data = []
# for label, times in times_data:
#     # if label != _1x_baseline:
#     norm_time = [
#         t / p_i if p_i != 0 else 0 for p_i, t in zip(_1x_baseline_times, times)
#     ]
#     norm_time_data.append((label, norm_time))

# # Create an array for x-axis positions
# x = np.arange(len(providers))

# # Set the width of the bars
# bar_width = 0.25

# # Draw cublas as a horizontal dashed line
# ax7.axhline(y=1, color="black", linestyle="dashed")

# # Create bars using a loop
# for i, (label, speedup) in enumerate(norm_time_data):
#     if label not in other_legands:
#         other_legands.append(label)
#     ax7.bar(
#         x + i * bar_width,
#         speedup,
#         bar_width,
#         label=label,
#         linewidth=0.8,
#         edgecolor="black",
#         hatch=get_legend_item(label)[1],
#         color=get_legend_item(label)[0],
#     )

# # X-axis and labels
# ax7.set_xticks(x + len(norm_time_data) * bar_width / 2)
# ax7.set_xticklabels(providers)
# ax7.grid(False)

ax8 = fig.add_subplot(gs[2, 8:10])  # ResNet
providers = gated_retnet_bwd_providers
times_data = gated_retnet_bwd_times_data
_1x_baseline_times = dict(times_data)[_1x_baseline]
norm_time_data = []
for label, times in times_data:
    # if label != _1x_baseline:
    norm_time = [
        t / p_i if p_i != 0 else 0 for p_i, t in zip(_1x_baseline_times, times)
    ]
    norm_time_data.append((label, norm_time))

# Create an array for x-axis positions
x = np.arange(len(providers))

# Set the width of the bars
bar_width = 0.25

# Draw cublas as a horizontal dashed line
ax8.axhline(y=1, color="black", linestyle="dashed")

# Create bars using a loop
for i, (label, speedup) in enumerate(norm_time_data):
    if label not in other_legands:
        other_legands.append(label)
    ax8.bar(
        x + i * bar_width,
        speedup,
        bar_width,
        label=label,
        linewidth=0.8,
        edgecolor="black",
        hatch=get_legend_item(label)[1],
        color=get_legend_item(label)[0],
    )

# X-axis and labels
ax8.set_xticks(x + len(norm_time_data) * bar_width / 2)
ax8.set_xticklabels(providers)
ax8.grid(False)

ax9 = fig.add_subplot(gs[2, 10:12])  # ResNet
providers = retnet_chunk_bwd_providers
times_data = retnet_chunk_bwd_times_data
_1x_baseline_times = dict(times_data)[_1x_baseline]
norm_time_data = []
for label, times in times_data:
    # if label != _1x_baseline:
    norm_time = [
        t / p_i if p_i != 0 else 0 for p_i, t in zip(_1x_baseline_times, times)
    ]
    norm_time_data.append((label, norm_time))

# Create an array for x-axis positions
x = np.arange(len(providers))

# Set the width of the bars
bar_width = 0.25

# Draw cublas as a horizontal dashed line
ax9.axhline(y=1, color="black", linestyle="dashed")

# Create bars using a loop
for i, (label, speedup) in enumerate(norm_time_data):
    if label not in other_legands:
        other_legands.append(label)
    ax9.bar(
        x + i * bar_width,
        speedup,
        bar_width,
        label=label,
        linewidth=0.8,
        edgecolor="black",
        hatch=get_legend_item(label)[1],
        color=get_legend_item(label)[0],
    )

# X-axis and labels
ax9.set_xticks(x + len(norm_time_data) * bar_width / 2)
ax9.set_xticklabels(providers)
ax9.grid(False)

legand_font = 12
ax0.text(
    0.5,
    -0.53,
    "(a) DeepSeek-V2-Lite",
    transform=ax0.transAxes,
    fontsize=legand_font,
    ha="center",
)

ax1.text(
    0.5,
    -0.53,
    "(b) LLAMA-3.1-8B",
    transform=ax1.transAxes,
    fontsize=legand_font,
    ha="center",
)

ax2.text(
    0.5,
    -0.53,
    "(c) Diff-Transformer-3B",
    transform=ax2.transAxes,
    fontsize=legand_font,
    ha="center",
)

ax3.text(
    0.5,
    -0.53,
    "(d) Llama-8B-Sigmoid",
    transform=ax3.transAxes,
    fontsize=legand_font,
    ha="center",
)

ax4.text(
    0.5,
    -0.53,
    "(e) Simple GLA",
    transform=ax4.transAxes,
    fontsize=legand_font,
    ha="center",
)

ax5.text(
    0.5,
    -0.53,
    "(f) Relu Attention",
    transform=ax5.transAxes,
    fontsize=legand_font,
    ha="center",
)

ax6.text(
    0.5,
    -0.53,
    "(g) Mamba2",
    transform=ax6.transAxes,
    fontsize=legand_font,
    ha="center",
)

# ax7.text(
#     0.5,
#     -0.53,
#     "(g) RetNet",
#     transform=ax7.transAxes,
#     fontsize=legand_font,
#     ha="center",
# )

ax8.text(
    0.5,
    -0.53,
    "(h) Gated RetNet",
    transform=ax8.transAxes,
    fontsize=legand_font,
    ha="center",
)

ax9.text(
    0.5,
    -0.53,
    "(i) Retnet Chunk",
    transform=ax9.transAxes,
    fontsize=legand_font,
    ha="center",
)


y_size = 8
ax0.tick_params(axis="y", labelsize=y_size)  # 设置y轴刻度标签的字体大小
ax1.tick_params(axis="y", labelsize=y_size)  # 设置y轴刻度标签的字体大小
ax2.tick_params(axis="y", labelsize=y_size)  # 设置y轴刻度标签的字体大小
ax3.tick_params(axis="y", labelsize=y_size)  # 设置y轴刻度标签的字体大小
ax4.tick_params(axis="y", labelsize=y_size)  # 设置y轴刻度标签的字体大小
ax5.tick_params(axis="y", labelsize=y_size)  # 设置y轴刻度标签的字体大小
ax6.tick_params(axis="y", labelsize=y_size)  # 设置y轴刻度标签的字体大小
# ax7.tick_params(axis="y", labelsize=y_size)  # 设置y轴刻度标签的字体大小
ax8.tick_params(axis="y", labelsize=y_size)  # 设置y轴刻度标签的字体大小
ax9.tick_params(axis="y", labelsize=y_size)  # 设置y轴刻度标签的字体大小

# ax3_2.yaxis.set_major_locator(MultipleLocator(5))
axes = [ax0, ax1, ax2, ax3, ax4, ax5, ax6, ax8, ax9]
for ax in axes:
    ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))

# 为上面六个图添加图例
handles_other = []
labels_other = []
handles_Ladder = []
labels_Ladder = []
for ax in [ax0, ax1, ax2, ax3, ax4, ax5, ax6, ax8, ax9]:
    handles, labels = ax.get_legend_handles_labels()
    for handle, label in zip(handles, labels):
        if label not in (labels_other + labels_Ladder):
            if "Ladder" in label:
                handles_Ladder.append(handle)
                labels_Ladder.append(label)
            else:
                handles_other.append(handle)
                labels_other.append(label)
        else:
            pass
handles_other.extend(handles_Ladder)
labels_other.extend(labels_Ladder)
fig.legend(
    handles_other,
    labels_other,
    loc="upper center",
    bbox_to_anchor=(0.45, 0.980 - 0.02),
    ncol=len(labels_other),
    fontsize=10,
    frameon=True,
)

# 调整布局以避免图例被遮挡
fig.text(
    0.09,
    0.5,
    "Normalized latency Vs. AttentionFroge \n(lower is better)",
    fontsize=11,
    rotation=90,
    va="center",
    ha="center",
)
plt.subplots_adjust(top=0.85, bottom=0.15)
plt.show()
# plt.savefig(
#     "end2end_h100.pdf",
#     bbox_inches="tight",
# )
# plt.savefig(
#     "png/figure8_end2end_a100.png",
#     bbox_inches="tight",
#     transparent=False,
#     dpi=255,
# )