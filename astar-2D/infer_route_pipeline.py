"""
推理 + 走线流水线
=================
1. 生成一份数据样本，保存成图片 (figs/sample_input.png)
2. 用 checkpoints/best_congestion_unet.pt 对该样本推理，保存预测图 (figs/inference_result.png)
3. 用预测的拥塞图调用 astar_with_congestion.py 对所有线网走线，
   保存走线结果图 (figs/routing_result.png)
"""

import os
import random
import importlib.util

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = "/home/wangzhaoqing/test1-ok"
CKPT = os.path.join(BASE, "checkpoints", "best_congestion_unet.pt")
FIG_DIR = os.path.join(BASE, "figs")
# SAMPLE_SEED = random.randint(10**5, 10**9)  # 每次运行随机取种子；日志会打印，便于复现
SAMPLE_SEED = 534567632

def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ----------------------------------------------------------------------
# 0. 加载训练模块与 A* 模块
# ----------------------------------------------------------------------
tc = load_module("train_congestion", os.path.join(BASE, "train_congestion-v1.py"))
astar_mod = load_module("astar", os.path.join(BASE, "astar_with_congestion.py"))

# 与训练时保持同一数据分布（不再覆盖 num_obstacles_max / num_nets_max），
# 避免推理场景比训练更稠密导致分布外(OOD)、预测不可靠。
cfg = tc.Config()
device = "cuda" if torch.cuda.is_available() else "cpu"
os.makedirs(FIG_DIR, exist_ok=True)

# ----------------------------------------------------------------------
# 1. 加载模型
# ----------------------------------------------------------------------
ckpt = torch.load(CKPT, map_location=device)
model = tc.CongestionUNet(cfg.in_channels).to(device)
model.load_state_dict(ckpt["model_state"])
model.eval()
print(f"Loaded checkpoint: {CKPT} (epoch={ckpt.get('epoch')}, "
      f"best_val_mae={ckpt.get('best_val_mae'):.4f})")

# ----------------------------------------------------------------------
# 2. 生成一份样本数据并保存图片
# ----------------------------------------------------------------------
print(f"Sample seed: {SAMPLE_SEED}")   # 记下这个值，需要复现时改回固定种子即可
ds = tc.RoutingDataset(1, cfg, seed=SAMPLE_SEED)
feat, label = ds[0]                     # feat: (6,H,W)  label: (1,H,W)

obstacle_map = feat[0].numpy()          # (H,W)
pin_map = feat[1].numpy()               # (H,W)
gt_map = label[0].numpy()               # (H,W)

fig, axes = plt.subplots(1, 2, figsize=(11, 5))
axes[0].imshow(obstacle_map, cmap="Reds", alpha=0.6)
axes[0].imshow(pin_map, cmap="Blues", alpha=0.8)
axes[0].set_title("Sample input (obstacles + pins)")
im = axes[1].imshow(gt_map, cmap="hot", vmin=0, vmax=1)
axes[1].set_title("Ground-truth congestion (simulated)")
fig.colorbar(im, ax=axes[1])
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "sample_input.png"), dpi=120)
plt.close(fig)
print("Saved sample_input.png")

# ----------------------------------------------------------------------
# 3. 推理并保存预测图
# ----------------------------------------------------------------------
with torch.no_grad():
    pred = model(feat.unsqueeze(0).to(device))          # (1,1,H,W)
pred_map = pred[0, 0].cpu().numpy()
print(f"Prediction stats: min={pred_map.min():.4f} max={pred_map.max():.4f} "
      f"mean={pred_map.mean():.4f}")

fig, axes = plt.subplots(1, 2, figsize=(11, 5))
axes[0].imshow(gt_map, cmap="hot", vmin=0, vmax=1)
axes[0].set_title("Ground truth")
im1 = axes[1].imshow(pred_map, cmap="hot", vmin=0, vmax=1)
axes[1].set_title("Model prediction (input to A*)")
fig.colorbar(im1, ax=axes[1])
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "inference_result.png"), dpi=120)
plt.close(fig)
print("Saved inference_result.png")

# ----------------------------------------------------------------------
# 4. A* 走线（用预测拥塞图作为代价）
# ----------------------------------------------------------------------
obstacle_mask = obstacle_map > 0.5
pin_coords = [tuple(p) for p in np.argwhere(pin_map > 0.5)]   # (row, col)
# 任意随机配对：用与样本一致的种子打乱引脚顺序（便于复现），再两两配对
order = np.random.default_rng(SAMPLE_SEED).permutation(len(pin_coords))
pin_coords = [pin_coords[i] for i in order]
nets = [(pin_coords[i], pin_coords[i + 1])
        for i in range(0, len(pin_coords) - 1, 2)]
print(f"Pins: {len(pin_coords)}, nets: {len(nets)}")

# 预测经 sigmoid 输出、上限为 1.0，hard_thr 即归一化空间的硬约束阈值。
# 若走廊被硬约束堵死则逐级放宽（0.85 -> 0.97 -> 1.0 即只避障），并在日志中标明。
#
# beta 按预测图幅值定标：代价 step = alpha*1 + beta*cong，beta 表示
# "一单位拥塞值相当于多少格走线代价"。取预测图的高分位作为参考拥塞 c_ref，
# 令最拥塞处约等于 BETA_TARGET_CELLS 格，beta 随模型输出幅值自适应，
# 避免像 beta=25 那样拍绝对数值导致路径过度绕行（那是为补偿塌缩的预测幅值）。
BETA_TARGET_CELLS = 5.0
c_ref = float(np.percentile(pred_map, 99.0))
beta = BETA_TARGET_CELLS / max(c_ref, 1e-6)
print(f"beta={beta:.2f} (c_ref=p99={c_ref:.3f}, target={BETA_TARGET_CELLS} cells)")

HARD_THR_STEPS = (0.85, 0.97, 1.0)
paths, failed = [], []
for net_id, (s, g) in enumerate(nets):
    path, used_thr = None, None
    for thr in HARD_THR_STEPS:
        path = astar_mod.astar_with_congestion(
            s, g, pred_map, obstacle_mask, alpha=1.0, beta=beta, hard_thr=thr)
        if path is not None:
            used_thr = thr
            break
    if path is None:
        failed.append(net_id)
        print(f"Net {net_id}: FAILED ({s} -> {g})")
    else:
        paths.append((net_id, path))
        note = "" if used_thr == HARD_THR_STEPS[0] else f" [relaxed hard_thr={used_thr:g}]"
        print(f"Net {net_id}: routed, {len(path)} cells, {s} -> {g}{note}")

# ----------------------------------------------------------------------
# 5. 保存走线结果图
# ----------------------------------------------------------------------
colors = plt.cm.tab10(np.linspace(0, 1, 10))
fig, ax = plt.subplots(figsize=(9.5, 8))
im = ax.imshow(pred_map, cmap="hot", vmin=0, vmax=1, alpha=0.9)  # pred_map 作清晰底图
# 障碍区域只画半透明填充 + 黑色轮廓，不整幅罩灰，保留拥塞底图可见
obstacle_overlay = np.ma.masked_less(obstacle_map, 0.5)
ax.imshow(obstacle_overlay, cmap="gray_r", alpha=0.55)
ax.contour(obstacle_map, levels=[0.5], colors="black", linewidths=1.0)

ys, xs = np.where(pin_map > 0.5)
ax.scatter(xs, ys, s=30, marker="s", c="lime", edgecolors="black",
           linewidths=0.5, zorder=3, label="pins")

for net_id, path in paths:
    arr = np.array(path)
    ax.plot(arr[:, 1], arr[:, 0], color=colors[net_id % 10],
            linewidth=1.8, zorder=2, label=f"net {net_id}")

ax.set_title(f"A* routing on predicted congestion "
             f"({len(paths)}/{len(nets)} nets routed)")
fig.colorbar(im, ax=ax, shrink=0.75, label="predicted congestion")
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.04),
          ncol=7, fontsize=7)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "routing_result.png"), dpi=120,
            bbox_inches="tight")
plt.close(fig)
print("Saved routing_result.png")
print(f"Done. Figures in {FIG_DIR}; failed nets: {failed if failed else 'none'}")
