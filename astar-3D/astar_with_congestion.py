import heapq
import numpy as np


def reconstruct(parent, start, goal):
    """由 parent 字典回溯路径，返回 [start, ..., goal]。"""
    path = [goal]
    cur = goal
    while cur != start:
        cur = parent[cur]
        path.append(cur)
    path.reverse()
    return path


def astar_with_congestion(start, goal, cong_map, obstacle_mask,
                          alpha=1.0, beta=4.0, hard_thr=0.85, via_cost=8.0):
    """
    多层 (3D) 网格上的 A* 布线。

    坐标均为 (layer, row, col) 三元组；cong_map / obstacle_mask 形状 (L, H, W)。
    移动分两类：
      - 同层 4 邻域，步长代价 alpha + beta * cong；
      - 跨层过孔 (layer±1, 同一 row/col)，额外代价 via_cost。
    启发式 h = |Δrow| + |Δcol| + via_cost * |Δlayer| 可采纳
    （过孔真实代价 >= via_cost）。
    """
    L, H, W = cong_map.shape
    g_score = np.full((L, H, W), np.inf, dtype=np.float64)
    g_score[start] = 0.0
    parent = {}

    def h(a, b):                       # 曼哈顿距离 + 过孔下界
        return (abs(a[1] - b[1]) + abs(a[2] - b[2])
                + via_cost * abs(a[0] - b[0]))

    # 关键：把 β*cong 提前烘进启发式不可行，只能烘进 g。
    pq = [(h(start, goal), 0.0, start)]
    while pq:
        _, g, cur = heapq.heappop(pq)
        if g > g_score[cur]:           # 陈旧条目：已找到更优 g，跳过
            continue
        if cur == goal:
            return reconstruct(parent, start, goal)

        l, y, x = cur
        # 同层 4 邻域
        for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            ny, nx = y + dy, x + dx
            if not (0 <= ny < H and 0 <= nx < W):
                continue
            if obstacle_mask[l, ny, nx]:
                continue
            if cong_map[l, ny, nx] > hard_thr:   # 硬约束：极度拥塞当作障碍
                continue

            step = alpha * 1.0 + beta * float(cong_map[l, ny, nx])
            ng = g + step
            if ng < g_score[l, ny, nx]:
                g_score[l, ny, nx] = ng
                parent[(l, ny, nx)] = (l, y, x)
                heapq.heappush(pq, (ng + h((l, ny, nx), goal), ng, (l, ny, nx)))

        # 跨层过孔：同一 (y, x) 打到相邻层
        for nl in (l - 1, l + 1):
            if not (0 <= nl < L):
                continue
            if obstacle_mask[nl, y, x]:
                continue
            if cong_map[nl, y, x] > hard_thr:
                continue

            ng = g + via_cost + alpha * 1.0 + beta * float(cong_map[nl, y, x])
            if ng < g_score[nl, y, x]:
                g_score[nl, y, x] = ng
                parent[(nl, y, x)] = (l, y, x)
                heapq.heappush(pq, (ng + h((nl, y, x), goal), ng, (nl, y, x)))
    return None
