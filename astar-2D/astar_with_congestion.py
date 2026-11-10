import heapq, numpy as np

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
                          alpha=1.0, beta=4.0, hard_thr=0.85):
    H, W = cong_map.shape
    g_score = np.full((H, W), np.inf, dtype=np.float64)
    g_score[start] = 0.0
    parent = {}

    def h(a, b):                       # 曼哈顿距离
        return abs(a[0]-b[0]) + abs(a[1]-b[1])

    # 关键：把 β*cong 提前烘进启发式不可行，只能烘进 g。
    pq = [(h(start, goal), 0.0, start)]
    while pq:
        _, g, cur = heapq.heappop(pq)
        if g > g_score[cur]:           # 陈旧条目：已找到更优 g，跳过
            continue
        if cur == goal:
            return reconstruct(parent, start, goal)

        y, x = cur
        for dy, dx in ((1,0),(-1,0),(0,1),(0,-1)):
            ny, nx = y+dy, x+dx
            if not (0 <= ny < H and 0 <= nx < W):
                continue
            if obstacle_mask[ny, nx]:
                continue
            if cong_map[ny, nx] > hard_thr:   # 硬约束：极度拥塞当作障碍
                continue

            step = alpha * 1.0 + beta * float(cong_map[ny, nx])
            ng = g + step
            if ng < g_score[ny, nx]:
                g_score[ny, nx] = ng
                parent[(ny, nx)] = (y, x)
                heapq.heappush(pq, (ng + h((ny, nx), goal), ng, (ny, nx)))
    return None