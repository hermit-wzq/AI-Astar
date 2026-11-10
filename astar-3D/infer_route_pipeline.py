"""
推理 + 走线流水线（多层版）
===========================
1. 生成一份数据样本，保存成图片 (figs/sample_input.png)
2. 用 checkpoints/best_congestion_unet.pt 对该样本推理，保存预测图 (figs/inference_result.png)
3. 用预测的拥塞图调用 astar_with_congestion.py 对所有线网在 3D 网格上走线
   （含层间过孔），保存分层走线结果图 (figs/routing_result.png)
"""

import os
import random
import importlib.util

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

BASE = "/home/wangzhaoqing/test1-3D-ok"
CKPT = os.path.join(BASE, "checkpoints", "best_congestion_unet.pt")
FIG_DIR = os.path.join(BASE, "figs")
SAMPLE_SEED = random.randint(10**5, 10**9)  # 每次运行随机取种子；日志会打印，便于复现
# SAMPLE_SEED = 534567632

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

device = "cuda" if torch.cuda.is_available() else "cpu"
os.makedirs(FIG_DIR, exist_ok=True)

# ----------------------------------------------------------------------
# 1. 加载模型（架构相关配置从 checkpoint 读回，保证与训练一致）
# ----------------------------------------------------------------------
ckpt = torch.load(CKPT, map_location=device)

cfg = tc.Config()
ckpt_cfg = ckpt.get("cfg", {})
# 只回读与"网格/层数/标签归一化/过孔"相关的配置；in_channels 由 Config 提供，
# 避免旧 checkpoint 的通道数污染数据集（导致写通道 6/7 越界）。
for k in ("num_layers", "grid_h", "grid_w", "cong_threshold",
          "via_cost", "via_penalty", "num_obstacles_max", "num_nets_max"):
    if k in ckpt_cfg:
        setattr(cfg, k, ckpt_cfg[k])

model = tc.CongestionUNet(cfg.in_channels).to(device)
model.load_state_dict(ckpt["model_state"])
model.eval()
print(f"Loaded checkpoint: {CKPT} (epoch={ckpt.get('epoch')}, "
      f"best_val_mae={ckpt.get('best_val_mae'):.4f})")
print(f"Config: L={cfg.num_layers}, grid={cfg.grid_h}x{cfg.grid_w}, "
      f"via_cost={cfg.via_cost}")

# ----------------------------------------------------------------------
# 2. 生成一份样本数据并保存图片
# ----------------------------------------------------------------------
print(f"Sample seed: {SAMPLE_SEED}")   # 记下这个值，需要复现时改回固定种子即可
ds = tc.RoutingDataset(1, cfg, seed=SAMPLE_SEED)
feat, label = ds[0]                     # feat: (L, C, H, W)  label: (L, H, W)
L = cfg.num_layers

obstacle_map = feat[:, 0].numpy()       # (L, H, W)
pin_map = feat[:, 1].numpy()            # (L, H, W)
gt_map = label.numpy()                  # (L, H, W)

# 样本输入图：展示第 0 层
fig, axes = plt.subplots(1, 2, figsize=(11, 5))
axes[0].imshow(obstacle_map[0], cmap="Reds", alpha=0.6)
axes[0].imshow(pin_map[0], cmap="Blues", alpha=0.8)
axes[0].set_title("Sample input layer 0 (obstacles + pins)")
im = axes[1].imshow(gt_map[0], cmap="hot", vmin=0, vmax=1)
axes[1].set_title("Ground-truth congestion layer 0 (simulated)")
fig.colorbar(im, ax=axes[1])
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "sample_input.png"), dpi=120)
plt.close(fig)
print("Saved sample_input.png")

# ----------------------------------------------------------------------
# 3. 推理并保存预测图
# ----------------------------------------------------------------------
with torch.no_grad():
    pred = model(feat.unsqueeze(0).to(device))          # (1, L, H, W)
pred_map = pred[0].cpu().numpy()                        # (L, H, W)
print(f"Prediction stats: min={pred_map.min():.4f} max={pred_map.max():.4f} "
      f"mean={pred_map.mean():.4f}")

fig, axes = plt.subplots(1, 2, figsize=(11, 5))
axes[0].imshow(gt_map[0], cmap="hot", vmin=0, vmax=1)
axes[0].set_title("Ground truth layer 0")
im1 = axes[1].imshow(pred_map[0], cmap="hot", vmin=0, vmax=1)
axes[1].set_title("Model prediction layer 0 (input to A*)")
fig.colorbar(im1, ax=axes[1])
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "inference_result.png"), dpi=120)
plt.close(fig)
print("Saved inference_result.png")

# ----------------------------------------------------------------------
# 4. A* 走线（用预测拥塞图作为代价，3D 网格含过孔）
# ----------------------------------------------------------------------
obstacle_mask = obstacle_map > 0.5                      # (L, H, W)
pin_coords = [tuple(int(v) for v in p)                  # (layer, row, col)
              for p in np.argwhere(pin_map > 0.5)]
# 任意随机配对：用与样本一致的种子打乱引脚顺序（便于复现），再两两配对
order = np.random.default_rng(SAMPLE_SEED).permutation(len(pin_coords))
pin_coords = [pin_coords[i] for i in order]
nets = [(pin_coords[i], pin_coords[i + 1])
        for i in range(0, len(pin_coords) - 1, 2)]
print(f"Pins: {len(pin_coords)}, nets: {len(nets)}")

# 预测经 sigmoid 输出、上限为 1.0，hard_thr 即归一化空间的硬约束阈值。
# 若走廊被硬约束堵死则逐级放宽（0.85 -> 0.97 -> 1.0 即只避障），并在日志中标明。
BETA_TARGET_CELLS = 5.0
c_ref = float(np.percentile(pred_map, 99.0))
beta = BETA_TARGET_CELLS / max(c_ref, 1e-6)
print(f"beta={beta:.2f} (c_ref=p99={c_ref:.3f}, target={BETA_TARGET_CELLS} cells)")

HARD_THR_STEPS = (0.85, 0.97, 1.0)

# 顺序布线：按直线距离（含层差）从短到长排序，先给资源占用少的线网让路。
def net_span(s, g):
    return abs(s[1] - g[1]) + abs(s[2] - g[2]) + cfg.via_cost * abs(s[0] - g[0])

route_occupancy = np.zeros_like(obstacle_mask)   # (L, H, W) 已被前序线网占用的格
paths, failed = [], []
for net_id in sorted(range(len(nets)), key=lambda i: net_span(*nets[i])):
    s, g = nets[net_id]
    # 硬禁止区 = 障碍 ∪ 已布走线；除本网两端外，其它线网的引脚也不允许被穿越
    blocked = obstacle_mask | route_occupancy
    for (pl, py, px) in pin_coords:
        if (pl, py, px) != s and (pl, py, px) != g:
            blocked[pl, py, px] = True

    path, used_thr = None, None
    for thr in HARD_THR_STEPS:
        path = astar_mod.astar_with_congestion(
            s, g, pred_map, blocked, alpha=1.0, beta=beta,
            hard_thr=thr, via_cost=cfg.via_cost)
        if path is not None:
            used_thr = thr
            break
    if path is None:
        failed.append(net_id)
        print(f"Net {net_id}: FAILED ({s} -> {g})")
    else:
        paths.append((net_id, path))
        for (pl, py, px) in path:               # 占用本网走线，禁止后续线网共用
            route_occupancy[pl, py, px] = True
        n_via = sum(1 for i in range(len(path) - 1)
                    if path[i][0] != path[i + 1][0])
        note = "" if used_thr == HARD_THR_STEPS[0] else f" [relaxed hard_thr={used_thr:g}]"
        print(f"Net {net_id}: routed, {len(path)} cells, {n_via} vias, "
              f"{s} -> {g}{note}")

# ----------------------------------------------------------------------
# 5. 保存分层走线结果图
# ----------------------------------------------------------------------
colors = plt.cm.tab10(np.linspace(0, 1, 10))
fig, axes = plt.subplots(1, L, figsize=(6.5 * L, 6.5), squeeze=False)
axes = axes[0]
im = None
for l in range(L):
    ax = axes[l]
    im = ax.imshow(pred_map[l], cmap="hot", vmin=0, vmax=1, alpha=0.9)
    obstacle_overlay = np.ma.masked_less(obstacle_map[l], 0.5)
    ax.imshow(obstacle_overlay, cmap="gray_r", alpha=0.55)
    ax.contour(obstacle_map[l], levels=[0.5], colors="black", linewidths=1.0)

    ys, xs = np.where(pin_map[l] > 0.5)
    ax.scatter(xs, ys, s=30, marker="s", c="lime", edgecolors="black",
               linewidths=0.5, zorder=3)

    # 只画该层内的连续路径段：遇到层号变化就断开。
    # 否则同一层的点集若中间曾离开本层(过孔到别层再回来)，会被错误地
    # 用一条直线首尾相连（出现"跨层直线"假象）。
    def draw_layer_run(net_id, run):
        if len(run) >= 2:
            seg = np.array(run)
            ax.plot(seg[:, 1], seg[:, 0], color=colors[net_id % 10],
                    linewidth=1.8, zorder=2)

    for net_id, path in paths:
        run = []
        for (pl, y, x) in path:
            if pl == l:
                run.append((y, x))
            else:
                draw_layer_run(net_id, run)
                run = []
        draw_layer_run(net_id, run)
    # 过孔：本层作为过孔端点(相邻点层号不同)用空心圆标记
    via_pts = []
    for net_id, path in paths:
        for i, (pl, y, x) in enumerate(path):
            if pl != l:
                continue
            prev_diff = i > 0 and path[i - 1][0] != l
            next_diff = i < len(path) - 1 and path[i + 1][0] != l
            if prev_diff or next_diff:
                via_pts.append((x, y))
    if via_pts:
        vp = np.array(via_pts)
        ax.scatter(vp[:, 0], vp[:, 1], s=70, marker="o", facecolors="none",
                   edgecolors="cyan", linewidths=1.6, zorder=4)

    ax.set_title(f"Layer {l}")
    ax.set_xlim(-0.5, cfg.grid_w - 0.5)
    ax.set_ylim(cfg.grid_h - 0.5, -0.5)

total_vias = sum(1 for _, path in paths for i in range(len(path) - 1)
                 if path[i][0] != path[i + 1][0])
fig.colorbar(im, ax=list(axes), shrink=0.75, label="predicted congestion")
legend_handles = [
    Line2D([0], [0], marker="s", color="none", markerfacecolor="lime",
           markeredgecolor="black", markersize=8, label="pins"),
    Line2D([0], [0], marker="o", color="none", markerfacecolor="none",
           markeredgecolor="cyan", markersize=9, label="via"),
]
axes[-1].legend(handles=legend_handles, loc="upper center",
                bbox_to_anchor=(0.5, -0.06), ncol=2, fontsize=8)
fig.suptitle(f"A* routing on predicted congestion "
             f"({len(paths)}/{len(nets)} nets routed, {total_vias} vias)")
plt.savefig(os.path.join(FIG_DIR, "routing_result.png"), dpi=120,
            bbox_inches="tight")
plt.close(fig)
print("Saved routing_result.png")
print(f"Done. Figures in {FIG_DIR}; failed nets: {failed if failed else 'none'}")
