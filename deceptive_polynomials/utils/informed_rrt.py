"""
Informed RRT* path planning

author: Karan Chawla
        Atsushi Sakai(@Atsushi_twi)

Reference: Informed RRT*: Optimal Sampling-based Path planning Focused via
Direct Sampling of an Admissible Ellipsoidal Heuristic
https://arxiv.org/pdf/1404.2334.pdf

"""
import sys
from pdb import set_trace as trace
import pathlib

import time
from utils.obs_full import convert_obstacle_to_vertices
import shapely.geometry as geom
import shapely.ops as ops

sys.path.append(str(pathlib.Path(__file__).parent.parent.parent))

from scipy.spatial import ConvexHull
from matplotlib.path import Path

import matplotlib.patches as patches

import copy
import math
import random

import matplotlib.pyplot as plt
import numpy as np

from utils.angle import rot_mat_2d

show_animation = True
def distance_squared_point_to_segment(v, w, p):

    """
    Return minimum squared distance between line segment vw and point p.
    
    Args:
    v (numpy.ndarray): The starting point of the line segment.
    w (numpy.ndarray): The ending point of the line segment.
    p (numpy.ndarray): The point for which the distance is calculated.
    
    Returns:
    float: The minimum squared distance between the line segment vw and point p.
    """
    # Return minimum distance between line segment vw and point p
    if np.array_equal(v, w):
        return (p - v).dot(p - v)  # v == w case
    l2 = (w - v).dot(w - v)  # i.e. |w-v|^2 -  avoid a sqrt
    # Consider the line extending the segment,
    # parameterized as v + t (w - v).
    # We find projection of point p onto the line.
    # It falls where t = [(p-v) . (w-v)] / |w-v|^2
    # We clamp t from [0,1] to handle points outside the segment vw.
    t = max(0, min(1, (p - v).dot(w - v) / l2))
    projection = v + t * (w - v)  # Projection falls on the segment
    return (p - projection).dot(p - projection)


def point_to_line_distance(p, a, b):

    """
    Calculate the distance from a point to a line segment.
    
    Args:
    p (numpy.ndarray): The point for which the distance is calculated.
    a (numpy.ndarray): The starting point of the line segment.
    b (numpy.ndarray): The ending point of the line segment.
    
    Returns:
    float: The distance from point p to the line segment ab.
    """
    ab = b - a
    ap = p - a
    t = np.dot(ap, ab) / np.dot(ab, ab)
    t = np.clip(t, 0, 1)
    closest_point = a + t * ab
    return np.linalg.norm(p - closest_point)

def line_segment_intersection(a1, a2, b1, b2):

    """
    Check if two line segments intersect.
    
    Args:
    a1 (numpy.ndarray): The starting point of the first line segment.
    a2 (numpy.ndarray): The ending point of the first line segment.
    b1 (numpy.ndarray): The starting point of the second line segment.
    b2 (numpy.ndarray): The ending point of the second line segment.
    
    Returns:
    bool: True if the line segments intersect, False otherwise.
    """
    a = a2 - a1
    b = b2 - b1
    c = b1 - a1
    det = np.linalg.det(np.array([a, -b]).T)
    if np.abs(det) < 1e-6:
        return False
    t = np.linalg.det(np.array([c, -b]).T) / det
    u = np.linalg.det(np.array([a, c]).T) / det
    return 0 <= t <= 1 and 0 <= u <= 1

def polygon_contains_point(vertices, point):
    """
    Check if a point is inside a polygon.
    
    Args:
    vertices (list): A list of vertices that define the polygon.
    point (numpy.ndarray): The point to be checked.
    
    Returns:
    bool: True if the point is inside the polygon, False otherwise.
    """
    j = len(vertices) - 1
    odd_nodes = False
    for i in range(len(vertices)):
        if ((vertices[i][1] < point[1] and vertices[j][1] >= point[1]) or
            (vertices[j][1] < point[1] and vertices[i][1] >= point[1])):
            if (vertices[i][0] + (point[1] - vertices[i][1]) / (vertices[j][1] - vertices[i][1]) *
                (vertices[j][0] - vertices[i][0]) < point[0]):
                odd_nodes = not odd_nodes
        j = i
    return odd_nodes
def expand_polygon(polygon, min_radius):
    """
    Expand a polygon by a specified distance.
    
    Args:
    polygon (list): A list of vertices that define the polygon.
    min_radius (float): The distance by which to expand the polygon.
    
    Returns:
    numpy.ndarray: The vertices of the expanded polygon.
    """
    polygon_geom = geom.Polygon(polygon)
    expanded_polygon_geom = polygon_geom.buffer(min_radius)
    expanded_polygon = np.array(expanded_polygon_geom.exterior.coords)[:-1]
    return expanded_polygon

def segment_intersects_obstacles(x1, y1, x2, y2, obstacles, min_radius=0.5):
    """
    Check if a line segment intersects any obstacles.
    
    Args:
    x1 (float): The x-coordinate of the starting point of the line segment.
    y1 (float): The y-coordinate of the starting point of the line segment.
    x2 (float): The x-coordinate of the ending point of the line segment.
    y2 (float): The y-coordinate of the ending point of the line segment.
    obstacles (list): A list of obstacle vertices.
    min_radius (float, optional): The minimum distance between the line segment and the obstacles. Defaults to 0.5.
    
    Returns:
    bool: True if the line segment intersects any obstacles, False otherwise.
    """
    p1 = np.array([x1, y1])
    p2 = np.array([x2, y2])
    for obstacle in obstacles:
        if polygon_contains_point(obstacle, p1) or polygon_contains_point(obstacle, p2):
            return False
        for i in range(len(obstacle)):
            next_idx = (i + 1) % len(obstacle)
            if line_segment_intersection(p1, p2, obstacle[i], obstacle[next_idx]):
                return False
            elif min_radius > 0:
                dist_squared = distance_squared_point_to_segment(obstacle[i], obstacle[next_idx], p1)
                if dist_squared <= min_radius ** 2:
                    return False
    return True


class InformedRRTStar:

    def __init__(self, start, goal, obstacle_list, rand_area, expand_dis=1,
                 goal_sample_rate=10, max_iter=50):

        self.start = Node(start[0], start[1])
        self.goal = Node(goal[0], goal[1])
        self.min_rand = rand_area[0]
        self.max_rand = rand_area[1]
        self.expand_dis = expand_dis
        self.goal_sample_rate = goal_sample_rate
        self.max_iter = max_iter
        self.obstacle_list = obstacle_list
        self.node_list = None
    def set_goal(self, goal): 
        self.goal = Node(goal[0], goal[1])
    def informed_rrt_star_search(self, animation=True):

        self.node_list = [self.start]
        # max length we expect to find in our 'informed' sample space,
        # starts as infinite
        c_best = float('inf')
        solution_set = set()
        path = None

        # Computing the sampling space
        c_min = math.hypot(self.start.x - self.goal.x,
                           self.start.y - self.goal.y)
        x_center = np.array([[(self.start.x + self.goal.x) / 2.0],
                             [(self.start.y + self.goal.y) / 2.0], [0]])
        a1 = np.array([[(self.goal.x - self.start.x) / c_min],
                       [(self.goal.y - self.start.y) / c_min], [0]])

        e_theta = math.atan2(a1[1], a1[0])
        # first column of identity matrix transposed
        id1_t = np.array([1.0, 0.0, 0.0]).reshape(1, 3)
        m = a1 @ id1_t
        u, s, vh = np.linalg.svd(m, True, True)
        c = u @ np.diag(
            [1.0, 1.0,
             np.linalg.det(u) * np.linalg.det(np.transpose(vh))]) @ vh

        for i in range(self.max_iter):
            # Sample space is defined by c_best
            # c_min is the minimum distance between the start point and
            # the goal x_center is the midpoint between the start and the
            # goal c_best changes when a new path is found

            rnd = self.informed_sample(c_best, c_min, x_center, c)
            n_ind = self.get_nearest_list_index(self.node_list, rnd)
            nearest_node = self.node_list[n_ind]
            # steer
            theta = math.atan2(rnd[1] - nearest_node.y,
                               rnd[0] - nearest_node.x)
            new_node = self.get_new_node(theta, n_ind, nearest_node)
            d = self.line_cost(nearest_node, new_node)

            no_collision = self.check_collision(nearest_node, theta, d)

            if no_collision:
                near_inds = self.find_near_nodes(new_node)
                new_node = self.choose_parent(new_node, near_inds)

                self.node_list.append(new_node)
                self.rewire(new_node, near_inds)

                if self.is_near_goal(new_node):
                    if segment_intersects_obstacles(new_node.x, new_node.y, self.goal.x, self.goal.y, self.obstacle_list):
                        solution_set.add(new_node)
                        last_index = len(self.node_list) - 1
                        temp_path = self.get_final_course(last_index)
                        temp_path_len = self.get_path_len(temp_path)
                        if temp_path_len < c_best:
                            path = temp_path
                            c_best = temp_path_len
            if animation:
                self.draw_graph(x_center=x_center, c_best=c_best, c_min=c_min,
                                e_theta=e_theta, rnd=rnd)

        return path

    def choose_parent(self, new_node, near_inds):
        if len(near_inds) == 0:
            return new_node

        d_list = []
        for i in near_inds:
            dx = new_node.x - self.node_list[i].x
            dy = new_node.y - self.node_list[i].y
            d = math.hypot(dx, dy)
            theta = math.atan2(dy, dx)
            if self.check_collision(self.node_list[i], theta, d):
                d_list.append(self.node_list[i].cost + d)
            else:
                d_list.append(float('inf'))

        min_cost = min(d_list)
        min_ind = near_inds[d_list.index(min_cost)]

        if min_cost == float('inf'):
            print("min cost is inf")
            return new_node

        new_node.cost = min_cost
        new_node.parent = min_ind

        return new_node

    def find_near_nodes(self, new_node):
        n_node = len(self.node_list)
        r = 50.0 * math.sqrt((math.log(n_node) / n_node))
        d_list = [(node.x - new_node.x) ** 2 + (node.y - new_node.y) ** 2 for
                  node in self.node_list]
        near_inds = [d_list.index(i) for i in d_list if i <= r ** 2]
        return near_inds

    def informed_sample(self, c_max, c_min, x_center, c):
        if c_max < float('inf'):
            r = [c_max / 2.0, 0.0, 0.0]
            if c_max ** 2 - c_min ** 2 >= 0:
                r = [c_max / 2.0, math.sqrt(c_max ** 2 - c_min ** 2) / 2.0,
                     math.sqrt(c_max ** 2 - c_min ** 2) / 2.0]
            rl = np.diag(r)
            x_ball = self.sample_unit_ball()
            rnd = np.dot(np.dot(c, rl), x_ball) + x_center
            rnd = [rnd[(0, 0)], rnd[(1, 0)]]
        else:
            rnd = self.sample_free_space()

        return rnd

    @staticmethod
    def sample_unit_ball():
        a = random.random()
        b = random.random()

        if b < a:
            a, b = b, a

        sample = (b * math.cos(2 * math.pi * a / b),
                  b * math.sin(2 * math.pi * a / b))
        return np.array([[sample[0]], [sample[1]], [0]])

    def sample_free_space(self):
        if random.randint(0, 100) > self.goal_sample_rate:
            rnd = [random.uniform(self.min_rand, self.max_rand),
                   random.uniform(self.min_rand, self.max_rand)]
        else:
            rnd = [self.goal.x, self.goal.y]

        return rnd

    @staticmethod
    def get_path_len(path):
        path_len = 0
        for i in range(1, len(path)):
            node1_x = path[i][0]
            node1_y = path[i][1]
            node2_x = path[i - 1][0]
            node2_y = path[i - 1][1]
            path_len += math.hypot(node1_x - node2_x, node1_y - node2_y)

        return path_len

    @staticmethod
    def line_cost(node1, node2):
        return math.hypot(node1.x - node2.x, node1.y - node2.y)

    @staticmethod
    def get_nearest_list_index(nodes, rnd):
        d_list = [(node.x - rnd[0]) ** 2 + (node.y - rnd[1]) ** 2 for node in
                  nodes]
        min_index = d_list.index(min(d_list))
        return min_index

    def get_new_node(self, theta, n_ind, nearest_node):
        new_node = copy.deepcopy(nearest_node)

        new_node.x += self.expand_dis * math.cos(theta)
        new_node.y += self.expand_dis * math.sin(theta)

        new_node.cost += self.expand_dis
        new_node.parent = n_ind
        return new_node

    def is_near_goal(self, node):
        d = self.line_cost(node, self.goal)
        if d < self.expand_dis:
            return True
        return False

    def rewire(self, new_node, near_inds):
        n_node = len(self.node_list)
        for i in near_inds:
            near_node = self.node_list[i]

            d = math.hypot(near_node.x - new_node.x, near_node.y - new_node.y)

            s_cost = new_node.cost + d

            if near_node.cost > s_cost:
                theta = math.atan2(new_node.y - near_node.y,
                                   new_node.x - near_node.x)
                if self.check_collision(near_node, theta, d):
                    near_node.parent = n_node - 1
                    near_node.cost = s_cost

    @staticmethod
    def distance_squared_point_to_segment(v, w, p):
        # Return minimum distance between line segment vw and point p
        if np.array_equal(v, w):
            return (p - v).dot(p - v)  # v == w case
        l2 = (w - v).dot(w - v)  # i.e. |w-v|^2 -  avoid a sqrt
        # Consider the line extending the segment,
        # parameterized as v + t (w - v).
        # We find projection of point p onto the line.
        # It falls where t = [(p-v) . (w-v)] / |w-v|^2
        # We clamp t from [0,1] to handle points outside the segment vw.
        t = max(0, min(1, (p - v).dot(w - v) / l2))
        projection = v + t * (w - v)  # Projection falls on the segment
        return (p - projection).dot(p - projection)

    def check_segment_collision(self, x1, y1, x2, y2):
        for (ox, oy, size) in self.obstacle_list:
            dd = self.distance_squared_point_to_segment(
                np.array([x1, y1]), np.array([x2, y2]), np.array([ox, oy]))
            if dd <= size ** 2:
                return False  # collision
        return True

    def check_collision(self, near_node, theta, d):
        tmp_node = copy.deepcopy(near_node)
        end_x = tmp_node.x + math.cos(theta) * d
        end_y = tmp_node.y + math.sin(theta) * d
#        return segment_intersects_obstacles(
        return segment_intersects_obstacles(tmp_node.x, tmp_node.y, end_x,
end_y, self.obstacle_list)

    def get_final_course(self, last_index):
        path = [[self.goal.x, self.goal.y]]
        while self.node_list[last_index].parent is not None:
            node = self.node_list[last_index]
            path.append([node.x, node.y])
            last_index = node.parent
        path.append([self.start.x, self.start.y])
        return path

    def draw_graph(self, x_center=None, c_best=None, c_min=None, e_theta=None,
                   rnd=None):
        plt.clf()
        # for stopping simulation with the esc key.
        plt.gcf().canvas.mpl_connect(
            'key_release_event', lambda event:
            [exit(0) if event.key == 'escape' else None])
        if rnd is not None:
            plt.plot(rnd[0], rnd[1], "^k")
            if c_best != float('inf'):
                self.plot_ellipse(x_center, c_best, c_min, e_theta)

        for node in self.node_list:
            if node.parent is not None:
                if node.x or node.y is not None:
                    plt.plot([node.x, self.node_list[node.parent].x],
                             [node.y, self.node_list[node.parent].y], "-g")

#        for (ox, oy, size) in self.obstacle_list:
#            plt.plot(ox, oy, "ok", ms=30 * size)
    # Plot obstacles
        ax = plt.gca()
        for obstacle in self.obstacle_list:
            hull = ConvexHull(obstacle)
            polygon = patches.Polygon(obstacle[hull.vertices], fill=True, alpha=0.5)
            ax.add_patch(polygon)
        plt.plot(self.start.x, self.start.y, "xr")
        plt.plot(self.goal.x, self.goal.y, "xr")
        plt.axis([-2, 15, -2, 15])
        plt.grid(True)
        plt.pause(0.01)

    @staticmethod
    def plot_ellipse(x_center, c_best, c_min, e_theta):  # pragma: no cover

        a = math.sqrt(c_best ** 2 - c_min ** 2) / 2.0
        b = c_best / 2.0
        angle = math.pi / 2.0 - e_theta
        cx = x_center[0]
        cy = x_center[1]
        t = np.arange(0, 2 * math.pi + 0.1, 0.1)
        x = [a * math.cos(it) for it in t]
        y = [b * math.sin(it) for it in t]
        fx = rot_mat_2d(-angle) @ np.array([x, y])
        px = np.array(fx[0, :] + cx).flatten()
        py = np.array(fx[1, :] + cy).flatten()
        plt.plot(cx, cy, "xc")
        plt.plot(px, py, "--c")


class Node:

    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.cost = 0.0
        self.parent = None


def main():
    print("Start informed rrt star planning")

    # create obstacles
    obstacle_list = [(5, 5, 0.5), (9, 6, 1), (7, 5, 1), (1, 5, 1), (3, 6, 1),
                     (7, 9, 1)]
    obstacle_list = [
        np.array([[1, 1], [2, 1], [2, 2], [1, 2]]),
        np.array([[4, 4], [5, 4], [5, 5], [4, 5]]),
    ]
    x_start=0
    y_start=1
    goal = (9,8)
    obstacle_1 = (3, 2, np.array([5.5,7])) # (length, high, center_coordinates)
    obstacles = [obstacle_1]
    obs_points = [convert_obstacle_to_vertices(obstacle) for obstacle in obstacles]
    # Set params
    start_time = time.time()
    rrt = InformedRRTStar(start=[x_start,y_start], goal=goal, max_iter=1000, rand_area=[-10, 20], obstacle_list=obs_points)
#    rrt.max_iter = 100
    path = rrt.informed_rrt_star_search(animation=True)
    end_time = time.time()
    elapsed_time = end_time - start_time

    print(f"Execution time: {elapsed_time:.3f} seconds")
    print("Done!!")

    # Plot path
    if show_animation:
        rrt.draw_graph()
        plt.plot([x for (x, y) in path], [y for (x, y) in path], '-r')
        plt.grid(True)
        plt.pause(0.01)
        plt.show()


if __name__ == '__main__':
    main()
