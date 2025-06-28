import heapq
import argparse
import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import Polygon, Point

class Node:
    def __init__(self, x, y, cost=0):
        self.x = x
        self.y = y
        self.cost = cost
        self.heuristic = 0
        self.total_cost = 0
        self.parent = None
    
    def __lt__(self, other):
        return self.total_cost < other.total_cost

def heuristic(a, b):
    return abs(a.x - b.x) + abs(a.y - b.y)

def get_neighbors(node, grid):
    neighbors = []
    for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        x, y = node.x + dx, node.y + dy
        if 0 <= x < len(grid) and 0 <= y < len(grid[0]) and grid[int(x)][int(y)] == 0:
            neighbors.append(Node(x, y))
    return neighbors

def a_star(start, goal, grid, heuristic_modifier=1.0, prune_threshold=None, prune_goal=None):
    open_list = []
    heapq.heappush(open_list, start)
    closed_list = set()
    while open_list:
        current_node = heapq.heappop(open_list)
        if (current_node.x, current_node.y) == (goal.x, goal.y):
            path = []
            while current_node:
                path.append((current_node.x, current_node.y))
                current_node = current_node.parent
            return path[::-1]
        closed_list.add((current_node.x, current_node.y))
        for neighbor in get_neighbors(current_node, grid):
            if (neighbor.x, neighbor.y) in closed_list:
                continue
            if prune_threshold and prune_goal and heuristic(neighbor, prune_goal) < prune_threshold:
                continue
            neighbor.cost = current_node.cost + 1
            neighbor.heuristic = heuristic(neighbor, goal) * heuristic_modifier
            neighbor.total_cost = neighbor.cost + neighbor.heuristic
            neighbor.parent = current_node
            heapq.heappush(open_list, neighbor)
    return None

def convert_obstacle_to_vertices(obstacle):
    length, height, center = obstacle
    x, y = center
    half_length = length / 2
    half_height = height / 2
    vertices = np.array([
        [x - half_length, y - half_height],
        [x + half_length, y - half_height],
        [x + half_length, y + half_height],
        [x - half_length, y + half_height],
    ])
    return vertices

def add_obstacles_to_grid(grid, obstacles, resolution, offset):
    for obstacle in obstacles:
        vertices = convert_obstacle_to_vertices(obstacle)
        polygon = Polygon(vertices)
        minx, miny, maxx, maxy = [int((coord + offset) * resolution) for coord in polygon.bounds]
        for x in range(minx, maxx + 1):
            for y in range(miny, maxy + 1):
                point = Point(x / resolution - offset, y / resolution - offset)
                if polygon.contains(point):
                    grid[x][y] = 1
    return grid

def plot_grid(grid, start, real_goal, decoy_goal, combined_path, resolution=10):
    fig, ax = plt.subplots()
    ax.grid(True)

    for x in range(len(grid)):
        for y in range(len(grid[0])):
            if grid[x][y] == 1:
                ax.add_patch(plt.Rectangle(((y - 100)/resolution  -
0.5/resolution , (x - 100)/resolution
- 0.5/resolution ), 1/resolution, 1/resolution, color="black"))

    ax.plot(start.y , start.x , "go", label="Start")
    ax.plot(real_goal.y , real_goal.x , "ro", label="Real Goal")
    ax.plot(decoy_goal.y , decoy_goal.x , "bo", label="Decoy Goal")

    if combined_path:
        combined_path_x, combined_path_y = zip(*[(node[1] / float(resolution), node[0] / float(resolution)) for node in combined_path])
        ax.plot(combined_path_x, combined_path_y, "g-", label="Combined Path")

    ax.legend()
    plt.show()

def get_ambiguous_path(start, real_goal, decoy_goal, obstacles, resolution=10,
grid_size=200, alpha=1.5, return_standard = True):
    grid = np.zeros((grid_size, grid_size))
    offset = grid_size // (2 * resolution)
    print(f"offset is {offset } ")
    grid = add_obstacles_to_grid(grid, obstacles, resolution, offset)

    start_node = Node((start[0] + offset) * resolution, (start[1] + offset) * resolution)
    real_goal_node = Node((real_goal[0] + offset) * resolution, (real_goal[1] + offset) * resolution)
    decoy_goal_node = Node((decoy_goal[0] + offset) * resolution, (decoy_goal[1] + offset) * resolution)

    def modified_heuristic(node, goal, bogus_goal):
        if heuristic(node, goal) < heuristic(node, bogus_goal):
            return heuristic(node, goal) * alpha
        return heuristic(node, goal)

    open_list = []
    heapq.heappush(open_list, (0, start_node))
    closed_list = set()
    while open_list:
        current_cost, current_node = heapq.heappop(open_list)
        if (current_node.x, current_node.y) == (real_goal_node.x, real_goal_node.y):
            path = []
            while current_node:
                path.append((current_node.x, current_node.y))
                current_node = current_node.parent
            if not return_standard:return ((np.array(path[::-1]) - offset *
resolution) / float(resolution)).tolist(), grid
            else: return (np.array(path[::-1])- offset * resolution).tolist(), grid
        closed_list.add((current_node.x, current_node.y))
        for neighbor in get_neighbors(current_node, grid):
            if (neighbor.x, neighbor.y) in closed_list:
                continue
            neighbor.cost = current_node.cost + 1
            neighbor.heuristic = modified_heuristic(neighbor, real_goal_node, decoy_goal_node)
            neighbor.total_cost = neighbor.cost + neighbor.heuristic
            neighbor.parent = current_node
            heapq.heappush(open_list, (neighbor.total_cost, neighbor))
    return None, grid

def get_final_path(start, real_goal, decoy_goal, obstacles, strategy,
resolution=10, grid_size=200,return_standard = True):
    if strategy == 1:  # Full Dissimulation
        return get_ambiguous_path(start, real_goal, decoy_goal, obstacles, resolution, grid_size, alpha=1.0)

    if strategy == 2:  # Simulation
        return get_ambiguous_path(start, real_goal, decoy_goal, obstacles, resolution, grid_size, alpha=1.5)

    if strategy == 3:  # Ambiguous Pathfinding
        return get_ambiguous_path(start, real_goal, decoy_goal, obstacles,
resolution, grid_size, alpha=1.5, return_standard = return_standard)

    return []

def main():
    parser = argparse.ArgumentParser(description='Select path-planning strategy.')
    parser.add_argument('--strategy', type=int, choices=[1, 2, 3], default=3, help='Choose strategy 1, 2, or 3.')
    args = parser.parse_args()

    resolution = 10  # Define the resolution for the grid
    grid_size = 200  # Define the size of the grid
    offset = grid_size // (2 * resolution)  # Define the offset for negative coordinates

    # Define obstacles
    obstacle_1 = (3, 2, np.array([4, 2]))  # (length, height, center_coordinates)
    obstacles = [obstacle_1]

    start = [0, 2]
    real_goal = [4, 4]
    decoy_goal = [4, 0]

    combined_path, grid = get_final_path(start, real_goal, decoy_goal, obstacles, strategy=args.strategy, resolution=resolution, grid_size=grid_size)

    if combined_path:
        print("Combined Path:", combined_path)
    else:
        print("No combined path found!")

    plot_grid(grid=grid, start=Node(start[0], start[1]), real_goal=Node(real_goal[0], real_goal[1]), decoy_goal=Node(decoy_goal[0], decoy_goal[1]), combined_path=combined_path, resolution=resolution)

if __name__ == "__main__":
    main()


