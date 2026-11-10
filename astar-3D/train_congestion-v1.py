"""
布线拥塞预测模型完整训练示例 (PyTorch) - 修复版
=============================================
"""

import os
import math
import random
import argparse
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ----------------------------------------------------------------------
# 1. 配置
# ----------------------------------------------------------------------
class Config:
    grid_h = 200
    grid_w = 200
    num_layers = 3          # 布线金属层数 L
    # 共享 2D U-Net：层被折进 batch 维，故 in_channels 是"单层"通道数。
    # 单层通道布局：[0]obstacle [1]pin [2]dens_near [3]dens_far
    #              [4]x_norm [5]y_norm [6]proj_pin [7]proj_obs
    in_channels = 8
    via_cost = 8.0          # 一次层间过孔的走线代价（单位：格）
    via_penalty = 3         # 拥塞波前跨层传播时的额外距离
    num_obstacles_max = 10
    num_nets_max = 20
    # 与 generate_congestion_map_3d 的实际拥塞峰值匹配（实测峰值约 1.5）。
    # 原值 6.0 会把标签压缩到接近 0（实测 label 全局最大仅 ≈0.28），
    # 阈值 0.3 下正样本为 0，二分类指标退化，模型只学到近零图。
    cong_threshold = 1.5
    # 二分类指标的归一化阈值（对应原始拥塞约 0.45）；必须低于标签最大值。
    metric_thr = 0.3
    train_size = 400
    val_size = 100
    batch_size = 16
    lr = 1e-3
    weight_decay = 1e-4
    epochs = 40
    patience = 15
    huber_beta = 0.1
    grad_clip = 1.0
    seed = None
    device = "cuda" if torch.cuda.is_available() else "cpu"
    save_dir = "./checkpoints"
    fig_dir = "./figs"


# ----------------------------------------------------------------------
# 2. 工具函数
# ----------------------------------------------------------------------
def box_blur_centered(a, k=5):
    """居中二维盒式模糊，k 必须为奇数。"""
    assert k % 2 == 1, "k must be odd"
    a = np.asarray(a, dtype=np.float64)
    r = k // 2

    # 水平
    p = np.pad(a, ((0, 0), (r, r)), mode="edge")
    c = np.cumsum(p, axis=1)
    c0 = np.concatenate([np.zeros((c.shape[0], 1), dtype=c.dtype), c], axis=1)
    out = (c0[:, k:k + a.shape[1]] - c0[:, :a.shape[1]]) / k

    # 垂直
    p = np.pad(out, ((r, r), (0, 0)), mode="edge")
    c = np.cumsum(p, axis=0)
    c0 = np.concatenate([np.zeros((1, c.shape[1]), dtype=c.dtype), c], axis=0)
    out = (c0[k:k + a.shape[0], :] - c0[:a.shape[0], :]) / k
    return out.astype(np.float32)


def generate_congestion_map_3d(h, w, obstacles, pins, num_layers,
                               radius=20, via_penalty=3):
    """
    多层波前传播 (BFS) 近似模拟拥塞分布。
    obstacles: list of (layer, row, col, h, w)
    pins:      list of (layer, row, col)
    同层按 4 邻域传播（步长 1）；跨层过孔传播（步长 via_penalty，
    仅当 (y, x) 在相邻层都可用时）。
    """
    free = np.ones((num_layers, h, w), dtype=bool)
    for (l, oy, ox, oh, ow) in obstacles:
        y0, y1 = max(0, oy), min(h, oy + oh)
        x0, x1 = max(0, ox), min(w, ox + ow)
        if y0 < y1 and x0 < x1:
            free[l, y0:y1, x0:x1] = False

    cong = np.zeros((num_layers, h, w), dtype=np.float32)
    # 复用 visited，用时间戳避免每次分配
    visited = np.zeros((num_layers, h, w), dtype=np.int32)
    stamp = 0

    for (pl, py, px) in pins:
        if not (0 <= pl < num_layers and 0 <= py < h and 0 <= px < w):
            continue
        if not free[pl, py, px]:
            continue

        stamp += 1
        q = deque([(pl, py, px, 0)])
        visited[pl, py, px] = stamp

        while q:
            l, y, x, d = q.popleft()
            if d > radius:
                continue
            cong[l, y, x] += 1.0 / (1.0 + d)
            if d == radius:
                continue
            # 同层 4 邻域
            for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                ny, nx = y + dy, x + dx
                if (0 <= ny < h and 0 <= nx < w
                        and free[l, ny, nx]
                        and visited[l, ny, nx] != stamp):
                    visited[l, ny, nx] = stamp
                    q.append((l, ny, nx, d + 1))
            # 跨层过孔（两层在 (y, x) 都可用时）
            for nl in (l - 1, l + 1):
                if (0 <= nl < num_layers
                        and free[nl, y, x]
                        and visited[nl, y, x] != stamp):
                    visited[nl, y, x] = stamp
                    q.append((nl, y, x, d + via_penalty))
    return cong


# ----------------------------------------------------------------------
# 3. 数据集
# ----------------------------------------------------------------------
class RoutingDataset(Dataset):
    def __init__(self, size, cfg, seed=None):
        self.cfg = cfg
        self.h, self.w = cfg.grid_h, cfg.grid_w
        self.size = size
        self.seed = seed
        # 样本由 (seed, idx) 唯一确定，缓存后无需每个 epoch 重复生成
        self._cache = {}
        assert self.h >= 8 and self.w >= 8, "grid size too small"

    def _make_sample(self, rng, np_rng):
        L, h, w = self.cfg.num_layers, self.h, self.w
        C = self.cfg.in_channels

        # 随机线网数量（先确定，用于约束障碍物）
        num_nets = rng.randint(2, self.cfg.num_nets_max)
        needed = 2 * num_nets

        # 随机障碍物 (layer, row, col, h, w)，可分布在任意层
        # 逐个放置并保证至少留下 needed 个自由体素，避免生成全零的退化样本
        obstacles = []
        free = np.ones((L, h, w), dtype=bool)
        for _ in range(rng.randint(2, self.cfg.num_obstacles_max)):
            l = rng.randrange(L)
            oh = rng.randint(8, min(24, h))
            ow = rng.randint(8, min(24, w))
            oy = rng.randint(0, h - oh)
            ox = rng.randint(0, w - ow)
            trial = free.copy()
            trial[l, oy:oy + oh, ox:ox + ow] = False
            if trial.sum() >= needed:
                obstacles.append((l, oy, ox, oh, ow))
                free = trial

        # 随机引脚：只在 free 体素上采样（free 体素数已保证 >= needed）
        free_coords = np.argwhere(free)  # shape (M, 3), [layer, row, col]
        idx = np_rng.choice(len(free_coords), size=needed, replace=False)
        chosen = free_coords[idx]
        pins = [(int(l), int(y), int(x)) for l, y, x in chosen]

        # 逐层特征图
        obstacle_map = np.zeros((L, h, w), dtype=np.float32)
        for (l, oy, ox, oh, ow) in obstacles:
            obstacle_map[l, oy:oy + oh, ox:ox + ow] = 1.0

        pin_map = np.zeros((L, h, w), dtype=np.float32)
        for (l, py, px) in pins:
            pin_map[l, py, px] = 1.0

        x_norm = np.tile(np.linspace(0, 1, w, dtype=np.float32), (h, 1))
        y_norm = np.tile(np.linspace(0, 1, h, dtype=np.float32)[:, None], (1, w))

        feat = np.zeros((L, C, h, w), dtype=np.float32)
        for l in range(L):
            pin_l = pin_map[l]
            feat[l, 0] = obstacle_map[l]
            feat[l, 1] = pin_l
            feat[l, 2] = box_blur_centered(pin_l, k=5)
            feat[l, 3] = box_blur_centered(pin_l, k=15)
            feat[l, 4] = x_norm
            feat[l, 5] = y_norm
        # 跨层投影：把别层的引脚/障碍聚合进来，弥补共享 2D U-Net 无层间感知
        feat[:, 6] = pin_map.max(axis=0)
        feat[:, 7] = obstacle_map.max(axis=0)

        cong = generate_congestion_map_3d(
            h, w, obstacles, pins, L, via_penalty=self.cfg.via_penalty)
        label = np.clip(cong, 0, self.cfg.cong_threshold) / self.cfg.cong_threshold
        return torch.from_numpy(feat), torch.from_numpy(label)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        sample = self._cache.get(idx)
        if sample is None:
            if self.seed is not None:
                rng = random.Random(self.seed + idx)
                np_rng = np.random.RandomState(self.seed + idx)
            else:
                rng = random
                np_rng = np.random
            sample = self._make_sample(rng, np_rng)
            self._cache[idx] = sample
        return sample


# ----------------------------------------------------------------------
# 4. UNet 模型
# ----------------------------------------------------------------------
class DoubleConv(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class Down(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.mp = nn.MaxPool2d(2)
        self.conv = DoubleConv(in_c, out_c)

    def forward(self, x):
        return self.conv(self.mp(x))


class Up(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_c, out_c, 2, stride=2)
        self.conv = DoubleConv(out_c * 2, out_c)

    def forward(self, x, skip):
        x = self.up(x)
        # 尺寸保险：若因奇数尺寸导致不匹配，裁剪 skip
        if x.shape[-2:] != skip.shape[-2:]:
            skip = skip[..., :x.shape[-2], :x.shape[-1]]
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class CongestionUNet(nn.Module):
    def __init__(self, in_channels=6, base=32, use_sigmoid=True):
        super().__init__()
        self.use_sigmoid = use_sigmoid
        self.inc = DoubleConv(in_channels, base)
        self.down1 = Down(base, base * 2)
        self.down2 = Down(base * 2, base * 4)
        self.down3 = Down(base * 4, base * 8)
        self.up1 = Up(base * 8, base * 4)
        self.up2 = Up(base * 4, base * 2)
        self.up3 = Up(base * 2, base)
        self.outc = nn.Conv2d(base, 1, 1)

    def _unet(self, x):
        """标准 2D U-Net 前向：x (N, C, H, W) -> (N, 1, H, W)。"""
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        x = self.outc(x)
        if self.use_sigmoid:
            x = torch.sigmoid(x)
        return x

    def forward(self, x):
        # 共享 2D U-Net：把层维折进 batch 维，权重共享。
        # 5D 输入 (B, L, C, H, W) -> 输出 (B, L, H, W)，与标签对齐。
        if x.dim() == 5:
            B, L = x.shape[0], x.shape[1]
            y = self._unet(x.reshape(B * L, *x.shape[2:]))
            return y.reshape(B, L, *y.shape[2:])
        # 4D 输入 (N, C, H, W) -> (N, 1, H, W)
        return self._unet(x)


# ----------------------------------------------------------------------
# 5. 评估指标
# ----------------------------------------------------------------------
def compute_metrics(pred, target, thr=0.3):
    """pred/target: (N, 1, H, W) 张量；thr 为归一化空间的二值化阈值。"""
    pred = pred.detach().float().flatten()
    target = target.detach().float().flatten()
    suf = f"@{thr:g}"

    if torch.isnan(pred).any() or torch.isnan(target).any():
        nan = float("nan")
        return {"MAE": nan, "RMSE": nan, "Corr": nan,
                f"Acc{suf}": nan, f"Precision{suf}": nan,
                f"Recall{suf}": nan, f"F1{suf}": nan, f"IoU{suf}": nan}

    mae = F.l1_loss(pred, target).item()
    mse = F.mse_loss(pred, target).item()
    rmse = math.sqrt(mse)

    # Pearson 相关系数
    p_mean, t_mean = pred.mean(), target.mean()
    num = ((pred - p_mean) * (target - t_mean)).sum()
    den = torch.sqrt(((pred - p_mean) ** 2).sum() * ((target - t_mean) ** 2).sum())
    corr = (num / (den + 1e-8)).item()

    # 二分类指标
    pred_bin = (pred > thr).float()
    tgt_bin = (target > thr).float()

    tp = (pred_bin * tgt_bin).sum().item()
    fp = (pred_bin * (1 - tgt_bin)).sum().item()
    fn = ((1 - pred_bin) * tgt_bin).sum().item()
    tn = ((1 - pred_bin) * (1 - tgt_bin)).sum().item()

    eps = 1e-8
    acc = (tp + tn) / (tp + fp + fn + tn + eps)
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)
    iou = tp / (tp + fp + fn + eps)

    return {
        "MAE": mae,
        "RMSE": rmse,
        "Corr": corr,
        f"Acc{suf}": acc,
        f"Precision{suf}": precision,
        f"Recall{suf}": recall,
        f"F1{suf}": f1,
        f"IoU{suf}": iou,
    }


# ----------------------------------------------------------------------
# 6. 训练 / 验证
# ----------------------------------------------------------------------
def train_one_epoch(model, loader, opt, cfg):
    model.train()
    total_loss = 0.0
    total_samples = 0
    all_preds, all_targets = [], []

    for x, y in loader:
        x, y = x.to(cfg.device), y.to(cfg.device)
        opt.zero_grad()
        pred = model(x)
        loss = F.smooth_l1_loss(pred, y, beta=cfg.huber_beta)
        loss.backward()
        if cfg.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        opt.step()

        total_loss += loss.item() * len(x)
        total_samples += len(x)
        all_preds.append(pred.detach().cpu())
        all_targets.append(y.detach().cpu())

    avg_loss = total_loss / max(total_samples, 1)
    pred = torch.cat(all_preds, dim=0)
    target = torch.cat(all_targets, dim=0)
    metrics = compute_metrics(pred, target, cfg.metric_thr)
    metrics["loss"] = avg_loss
    return metrics


@torch.no_grad()
def evaluate(model, loader, cfg):
    model.eval()
    all_preds, all_targets = [], []
    for x, y in loader:
        x, y = x.to(cfg.device), y.to(cfg.device)
        pred = model(x)
        all_preds.append(pred.cpu())
        all_targets.append(y.cpu())

    pred = torch.cat(all_preds, dim=0)
    target = torch.cat(all_targets, dim=0)
    return compute_metrics(pred, target, cfg.metric_thr)


# ----------------------------------------------------------------------
# 7. 可视化
# ----------------------------------------------------------------------
def visualize_sample(model, loader, cfg, path):
    model.eval()
    x, y = next(iter(loader))
    x, y = x[:1].to(cfg.device), y[:1].to(cfg.device)
    with torch.no_grad():
        pred = model(x)                  # (1, L, H, W)
    # 取第 0 层可视化（多层共享同一套权重，逐层形态一致）
    obstacle = x[0, 0, 0].cpu().numpy()
    pin = x[0, 0, 1].cpu().numpy()
    y_np = y[0, 0].cpu().numpy()
    p_np = pred[0, 0].cpu().numpy()

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(obstacle, cmap="Reds", alpha=0.5)
    axes[0].imshow(pin, cmap="Blues", alpha=0.7)
    axes[0].set_title("Input (obstacles + pins)")
    axes[1].imshow(y_np, cmap="hot", vmin=0, vmax=1)
    axes[1].set_title("Ground Truth")
    im2 = axes[2].imshow(p_np, cmap="hot", vmin=0, vmax=1)
    axes[2].set_title("Prediction")
    fig.colorbar(im2, ax=axes[2])
    plt.tight_layout()
    plt.savefig(path, dpi=100)
    plt.close(fig)


def plot_curves(train_losses, val_maes, path):
    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label="train loss")
    plt.plot(val_maes, label="val MAE")
    plt.xlabel("epoch")
    plt.ylabel("value")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(path, dpi=100)
    plt.close()


# ----------------------------------------------------------------------
# 8. 主流程
# ----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="布线拥塞预测模型训练")
    parser.add_argument("--epochs", type=int, default=Config.epochs)
    parser.add_argument("--batch_size", type=int, default=Config.batch_size)
    parser.add_argument("--lr", type=float, default=Config.lr)
    parser.add_argument("--grid", type=int, default=Config.grid_h, help="网格边长")
    parser.add_argument("--num_layers", type=int, default=Config.num_layers,
                        help="布线金属层数")
    parser.add_argument("--device", type=str, default=Config.device)
    args = parser.parse_args()

    cfg = Config()
    cfg.epochs = args.epochs
    cfg.batch_size = args.batch_size
    cfg.lr = args.lr
    cfg.grid_h = cfg.grid_w = args.grid
    cfg.num_layers = args.num_layers
    cfg.device = args.device
    if cfg.device == "cuda" and not torch.cuda.is_available():
        cfg.device = "cpu"

    # 尺寸检查
    if cfg.grid_h < 32:
        raise ValueError("grid size must be >= 32")
    if cfg.grid_h % 8 != 0:
        raise ValueError("grid size must be divisible by 8")

    # 未指定种子时随机生成一个，并打印以便复现
    if cfg.seed is None:
        cfg.seed = random.randint(0, 2**31 - 1)
    print(f"Seed: {cfg.seed}")

    # 给 checkpoint 用的完整配置：Config 类属性 + 运行时覆盖
    full_cfg = {k: v for k, v in vars(Config).items() if not k.startswith("_")}
    full_cfg.update(vars(cfg))

    os.makedirs(cfg.save_dir, exist_ok=True)
    os.makedirs(cfg.fig_dir, exist_ok=True)
    print(f"Device: {cfg.device}")
    print(f"Grid: {cfg.grid_h}x{cfg.grid_w}, Layers: {cfg.num_layers}")

    # 数据（训练/验证用不同 seed 区间，保证样本不重叠）
    train_ds = RoutingDataset(cfg.train_size, cfg, seed=cfg.seed)
    val_ds = RoutingDataset(cfg.val_size, cfg, seed=cfg.seed + 10000)

    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True,
        num_workers=0, pin_memory=(cfg.device == "cuda"),
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.batch_size, shuffle=False,
        num_workers=0, pin_memory=(cfg.device == "cuda"),
    )
    print(f"Train: {len(train_ds)} samples, Val: {len(val_ds)} samples")

    # 模型 / 优化器
    model = CongestionUNet(cfg.in_channels).to(cfg.device)
    opt = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs)

    # 训练
    best_val_mae = float("inf")
    best_epoch = -1
    patience_counter = 0
    train_losses, val_maes = [], []
    best_path = os.path.join(cfg.save_dir, "best_congestion_unet.pt")

    for epoch in range(1, cfg.epochs + 1):
        train_metrics = train_one_epoch(model, train_loader, opt, cfg)
        scheduler.step()
        val_metrics = evaluate(model, val_loader, cfg)

        tl = train_metrics["loss"]
        train_losses.append(tl)
        val_maes.append(val_metrics["MAE"])

        log = " | ".join(f"{k}={v:.4f}" for k, v in val_metrics.items())
        print(f"[Epoch {epoch:3d}/{cfg.epochs}] "
              f"train_loss={tl:.4f} train_MAE={train_metrics['MAE']:.4f} | "
              f"val {log}")

        # 早停 + 保存最优模型
        if val_metrics["MAE"] < best_val_mae:
            best_val_mae = val_metrics["MAE"]
            best_epoch = epoch
            patience_counter = 0
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": opt.state_dict(),
                "best_val_mae": best_val_mae,
                "cfg": full_cfg,
            }, best_path)
        else:
            patience_counter += 1
            if patience_counter >= cfg.patience:
                print(f"Early stopping at epoch {epoch}. "
                      f"Best epoch: {best_epoch}, "
                      f"Best val MAE: {best_val_mae:.4f}")
                break

    print(f"Training done. Best epoch: {best_epoch}, "
          f"Best val MAE: {best_val_mae:.4f}")

    # 回载最优权重，保证可视化与后续推理使用 best 模型（与 best_val_mae 一致）
    if os.path.exists(best_path):
        checkpoint = torch.load(best_path, map_location=cfg.device)
        model.load_state_dict(checkpoint["model_state"])
        print(f"Loaded best model from {best_path} "
              f"(epoch {checkpoint.get('epoch')}, "
              f"best_val_mae {checkpoint.get('best_val_mae'):.4f})")

    plot_curves(train_losses, val_maes,
                os.path.join(cfg.fig_dir, "loss_curves.png"))
    visualize_sample(model, val_loader, cfg,
                     os.path.join(cfg.fig_dir, "prediction_sample.png"))
    print(f"模型与图片已保存至 {cfg.save_dir} 和 {cfg.fig_dir}")


if __name__ == "__main__":
    main()