import numpy as np

from shapely.geometry import LineString, Polygon
import os
from pdb import set_trace as trace
import sys

from scipy.spatial import distance

def convert_obstacle_to_vertices(obstacle):
    """
    Convert an obstacle representation to a list of vertices.
    
    Args:
    obstacle (tuple): A tuple containing the length, height, and center coordinates of the obstacle.
    
    Returns:
    numpy.ndarray: An array of vertices representing the obstacle.
    """
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


def point_to_segment_dist(point, segment, radius=0):
    """
    Calculate the distance from a point to a line segment.
    
    Args:
    point (numpy.ndarray): The point for which the distance is calculated.
    segment (tuple): A tuple containing the starting and ending points of the line segment.
    radius (float, optional): A buffer radius to be subtracted from the distance. Defaults to 0.
    
    Returns:
    float: The distance from the point to the line segment, accounting for the radius.
    """
    start, end = segment
    d = distance.euclidean(start, end)
    t = np.dot(point - start, end - start) / (d * d)
    t = np.clip(t, 0, 1)
    proj = start + t * (end - start)
    dist = distance.euclidean(point, proj) - radius
    return max(dist, 0)

def compute_velocities(f_data):
    """
    Calculate the velocities between consecutive points in a trajectory.
    
    Args:
    f_data (numpy.ndarray): An array of points representing the trajectory.
    
    Returns:
    numpy.ndarray: An array of velocities between consecutive points in the trajectory.
    """
    velocities = np.diff(f_data, axis=0)
    return velocities
def obstacle_term(obstacle_list, t_train, f_data, radius=0):
    """
    Compute penalty terms for a trajectory based on the distances to obstacles and intersections.
    
    Args:
    obstacle_list (list): A list of obstacles represented as arrays of vertices.
    t_train (numpy.ndarray): An array of time values corresponding to the trajectory points.
    f_data (numpy.ndarray): An array of points representing the trajectory.
    radius (float, optional): A buffer radius for the obstacles. Defaults to 0.
    
    Returns:
    numpy.ndarray: An array of penalty terms for the trajectory points based on distances to obstacles and intersections.
    """
    num_points = t_train.shape[0]
    distances_to_obs = np.zeros((num_points, 1))

    for i in range(num_points - 1):
        trajectory_segment = LineString([f_data[i], f_data[i + 1]]).buffer(radius)
        intersection_found = False

        for obstacle in obstacle_list:
            obstacle_polygon = Polygon(obstacle).buffer(radius)
            if trajectory_segment.intersects(obstacle_polygon):
                intersection_found = True
                break


        if intersection_found:
            penalty_term + 1e8  # Large penalty for trajectory segments intersecting with obstacles
        else:
            point_location = f_data[i]
            min_distance = np.inf
            for obstacle in obstacle_list:
                num_vertices = obstacle.shape[0]
                for j in range(num_vertices):
                    edge = obstacle[j], obstacle[(j + 1) % num_vertices]
                    dist = point_to_segment_dist(point_location, edge, radius)
                    min_distance = min(min_distance, dist)
            penalty_term = 1 / (min_distance + 0.001)
        distances_to_obs[i] = penalty_term
    return distances_to_obs
def obstacle_term_no_radius(obstacle_list, t_train, f_data, radius=0):
    """
    Compute penalty terms for a trajectory based on the distances to obstacles and intersections, without considering a buffer radius.
    
    Args:
    obstacle_list (list): A list of obstacles represented as arrays of vertices.
    t_train (numpy.ndarray): An array of time values corresponding to the trajectory points.
    f_data (numpy.ndarray): An array of points representing the trajectory.
    radius (float, optional): A buffer radius for the obstacles. Defaults to 0.
    
    Returns:
    numpy.ndarray: An array of penalty terms for the trajectory points based on distances to obstacles and intersections, without considering a buffer radius.
    """
    num_points = t_train.shape[0]
    distances_to_obs = np.zeros((num_points, 1))

    for i in range(num_points - 1):
        trajectory_segment = LineString([f_data[i], f_data[i + 1]])
        intersection_found = False

        for obstacle in obstacle_list:
            obstacle_polygon = Polygon(obstacle)
            if trajectory_segment.intersects(obstacle_polygon):
                intersection_found = True
                break

        if intersection_found:
            penalty_term = 1e7  # Large penalty for trajectory segments intersecting with obstacles
        else:
            point_location = f_data[i]
            min_distance = np.inf
            for obstacle in obstacle_list:
                num_vertices = obstacle.shape[0]
                for j in range(num_vertices):
                    edge = obstacle[j], obstacle[(j + 1) % num_vertices]
                    dist = point_to_segment_dist(point_location, edge, radius)
                    min_distance = min(min_distance, dist)
            penalty_term = 1 / (min_distance + 0.001)

        distances_to_obs[i] = penalty_term
    return distances_to_obs

def obstacle_term_interior_point_only(obstacle_list, t_train, f_data, radius=0):
    """
    Compute penalty terms for a trajectory based on the distances to obstacles, considering only points inside the obstacles.

    Args:
    obstacle_list (list): A list of obstacles represented as arrays of vertices.
    t_train (numpy.ndarray): An array of time values corresponding to the trajectory points.
    f_data (numpy.ndarray): An array of points representing the trajectory.
    radius (float, optional): A buffer radius for the obstacles. Defaults to 0.

    Returns:
    numpy.ndarray: An array of penalty terms for the trajectory points based on distances to obstacles, considering only points inside the obstacles.
    """
    num_points = t_train.shape[0]
    distances_to_obs = np.zeros((num_points, 1))  # Change shape to (num_points, 1)
    for i, point_location in enumerate(f_data):
        min_distance = np.inf
        inside_obstacle = False
        for obstacle in obstacle_list:
            # Check if point is inside the obstacle
            if polygon_contains_point(obstacle, point_location):
                inside_obstacle = True
                break

            num_vertices = obstacle.shape[0]
            for j in range(num_vertices):
                edge = obstacle[j], obstacle[(j + 1) % num_vertices]
                dist = point_to_segment_dist(point_location, edge, radius)
                min_distance = min(min_distance, dist)

        if inside_obstacle:
            penalty_term = 1e8  # Large penalty for points inside the obstacle
        else:
            penalty_term = 1 / (min_distance + 0.001)  # Add small constant to avoid division by zero

        distances_to_obs[i] = penalty_term
    return distances_to_obs
def obstacle_term_no_interior(obstacle_list, t_train, f_data, radius=0):
    """
    Compute penalty terms for a trajectory based on the distances to obstacles, without considering points inside the obstacles.

    Args:
    obstacle_list (list): A list of obstacles represented as arrays of vertices.
    t_train (numpy.ndarray): An array of time values corresponding to the trajectory points.
    f_data (numpy.ndarray): An array of points representing the trajectory.
    radius (float, optional): A buffer radius for the obstacles. Defaults to 0.

    Returns:
    numpy.ndarray: An array of penalty terms for the trajectory points based on distances to obstacles, without considering points inside the obstacles.
    """
    num_points = t_train.shape[0]
    distances_to_obs = np.zeros((num_points, 1))  # Change shape to (num_points, 1)
    for i, point_location in enumerate(f_data):
        min_distance = np.inf
        for obstacle in obstacle_list:
            num_vertices = obstacle.shape[0]
            for j in range(num_vertices):
                edge = obstacle[j], obstacle[(j + 1) % num_vertices]
                dist = point_to_segment_dist(point_location, edge, radius)
                min_distance = min(min_distance, dist)
        penalty_term = 1 / (min_distance + 0.001)  # Add small constant to avoid division by zero
        distances_to_obs[i] = penalty_term
    return distances_to_obs

def obstacle_term_velocity(obstacle_list, t_train, f_data, radius=0):
    """
    Compute penalty terms for a trajectory based on the distances to obstacles, taking the velocity of the trajectory into account.

    Args:
    obstacle_list (list): A list of obstacles represented as arrays of vertices.
    t_train (numpy.ndarray): An array of time values corresponding to the trajectory points.
    f_data (numpy.ndarray): An array of points representing the trajectory.
    radius (float, optional): A buffer radius for the obstacles. Defaults to 0.

    Returns:
    numpy.ndarray: An array of penalty terms for the trajectory points based on distances to obstacles and the velocity of the trajectory.
    """
    velocities = compute_velocities(f_data)
    num_points = t_train.shape[0]
    distances_to_obs = np.zeros((num_points, 2))
    for i, (point_location, velocity) in enumerate(zip(f_data, velocities)):
        min_distance = np.inf
        for obstacle in obstacle_list:
            num_vertices = obstacle.shape[0]
            for j in range(num_vertices):
                edge = obstacle[j], obstacle[(j + 1) % num_vertices]
                dist = point_to_segment_dist(point_location, edge, radius)
                min_distance = min(min_distance, dist)
        distances_to_obs[i] = min_distance * velocity
    return distances_to_obs

