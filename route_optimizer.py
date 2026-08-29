"""
MODULE: Safe-route recommendation.

Input:  risk grid (from collision_risk.py) + start/end lon-lat
Output: a list of (lon, lat) waypoints for the safest path, plus total distance

Uses A* search over the grid, where each cell's traversal cost is driven by
its risk value (impassable cells are excluded, risky-but-passable cells cost more).
"""
import heapq
import numpy as np
from region import REGION, lonlat_to_grid, grid_to_lonlat

IMPASSABLE_THRESHOLD = 0.9   # risk value at/above this = cannot enter cell


def _neighbors(row, col, n):
    for dr, dc in [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]:
        r, c = row+dr, col+dc
        if 0 <= r < n and 0 <= c < n:
            yield r, c, np.hypot(dr, dc)


def find_safe_route(risk_grid, start_lonlat, end_lonlat):
    n = REGION["grid_size"]
    start = lonlat_to_grid(*start_lonlat)
    goal = lonlat_to_grid(*end_lonlat)

    def _search(threshold):
        def cost(row, col):
            if (row, col) == start or (row, col) == goal:
                return 1.0
            r = risk_grid[row, col]
            if r >= threshold:
                return None
            return 1.0 + r * 8.0

        def heuristic(a, b):
            return np.hypot(a[0]-b[0], a[1]-b[1])

        open_set = [(0, start)]
        came_from = {}
        g_score = {start: 0.0}

        while open_set:
            _, current = heapq.heappop(open_set)
            if current == goal:
                break
            for r, c, step_dist in _neighbors(*current, n):
                cell_cost = cost(r, c)
                if cell_cost is None:
                    continue
                tentative_g = g_score[current] + step_dist * cell_cost
                if (r, c) not in g_score or tentative_g < g_score[(r, c)]:
                    g_score[(r, c)] = tentative_g
                    came_from[(r, c)] = current
                    f = tentative_g + heuristic((r, c), goal)
                    heapq.heappush(open_set, (f, (r, c)))

        if goal not in came_from and goal != start:
            return None

        path = [goal]
        while path[-1] != start:
            path.append(came_from[path[-1]])
        path.reverse()

        waypoints = [grid_to_lonlat(r, c) for r, c in path]
        total_cost = g_score[goal]

        return {
            "success": True,
            "waypoints": waypoints,
            "n_waypoints": len(waypoints),
            "routing_cost": round(float(total_cost), 2),
        }

    # Try strict threshold first, then progressively relax
    for threshold in [IMPASSABLE_THRESHOLD, 0.95, 1.01]:
        result = _search(threshold)
        if result is not None:
            if threshold > IMPASSABLE_THRESHOLD:
                result["warning"] = f"Path required through high-risk zone (threshold relaxed to {threshold})"
            return result

    return {"success": False, "reason": "No safe path found - region fully blocked", "waypoints": []}


if __name__ == "__main__":
    from region import PORT_A, PORT_B, generate_sic_history
    from predict_sea_ice import predict_sea_ice
    from predict_iceberg import predict_iceberg_trajectories
    from collision_risk import compute_risk_grid

    history = generate_sic_history()
    sic_forecast = predict_sea_ice(history)
    trajectories = predict_iceberg_trajectories()
    risk = compute_risk_grid(sic_forecast[0], trajectories, day_index=0)

    route = find_safe_route(risk, (PORT_A["lon"], PORT_A["lat"]), (PORT_B["lon"], PORT_B["lat"]))
    print("Route success:", route["success"])
    print("Waypoint count:", route["n_waypoints"])
    print("Routing cost:", route["routing_cost"])
    print("First 3 waypoints:", route["waypoints"][:3])
    print("Last 3 waypoints:", route["waypoints"][-3:])
