
from numpy.random import default_rng
from utils.point_resampler import resample_points

from utils.informed_rrt import InformedRRTStar
from functools import partial

from matplotlib.lines import Line2D

import matplotlib.pyplot as plt
import math
from pdb import set_trace as trace

from scipy.optimize import minimize, NonlinearConstraint, SR1
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
from matplotlib.path import Path
def cost_function(trajectory, distance_weight=0.1):
    """
    Calculate the cost of a given trajectory based on the integral over squared velocities.
    Optionally add a distance regularization term if distance_weight is greater than 0.

    :param trajectory: numpy array, a sequence of points representing the trajectory
    :param distance_weight: float, the weight of the distance regularization term, default=0.1
    :return: float, the cost of the trajectory
    """
    # Here, we use the integral over squared velocities plus a distance regularization term as the cost function
    vel_cost = np.sum(np.diff(trajectory, axis=0)**2)
#    if distance_weight > 0:
#        dist_cost = calculate_distance(trajectory)
#        vel_cost += distance_weight * dist_cost
    return .1 * vel_cost





def simple_objective_function(trajectory_flat, start, goal):

    """
    Objective function for trajectory optimization, which calculates the cost of a given trajectory.

    :param trajectory_flat: numpy array, a flattened sequence of points representing the trajectory
    :param start: numpy array or list, the starting point of the trajectory
    :param goal: numpy array or list, the goal point of the trajectory
    :return: float, the cost of the trajectory
    """
    trajectory = trajectory_flat.reshape(-1, 2)
    trajectory = np.vstack([start,trajectory, goal])
    return cost_function(trajectory)

def plot_convex_hull(points, point = None, title='Convex Hull'):

    """
    Plot the convex hull of a set of points, and optionally a single point.

    :param points: numpy array, a set of points
    :param point: numpy array or list, a single point, default=None
    :param title: string, the title of the plot, default='Convex Hull'
    """
    hull = ConvexHull(points)
    if point is not None: plt.plot(point[0], point[1], 'o')
    plt.plot(points[:, 0], points[:, 1], '.', color='k')
    plt.title(title)
    for simplex in hull.simplices:
        plt.plot(points[simplex, 0], points[simplex, 1], 'c')
    plt.plot(points[hull.vertices, 0], points[hull.vertices, 1], 'o', mec='r', color='none', lw=1, markersize=10)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.grid()
    plt.show()


def check_collision(trajectory, obstacles):
    """
    Check if a trajectory collides with any obstacles.

    :param trajectory: numpy array, a sequence of points representing the trajectory
    :param obstacles: list of numpy arrays, each array representing the vertices of an obstacle
    :return: bool, True if there is a collision, False otherwise
    """
    for obstacle in obstacles:
        hull = ConvexHull(obstacle)

        hull_path = Path( obstacle[hull.vertices] )
        for point in trajectory:
            if hull_path.contains_point(point):
                return True
    return False


def collision_constraint(trajectory_flat, obstacles):

    """
    Constraint function for trajectory optimization that ensures the trajectory is collision-free.

    :param trajectory_flat: numpy array, a flattened sequence of points representing the trajectory
    :param obstacles: list of numpy arrays, each array representing the vertices of an obstacle
    :return: int, 1 if the trajectory is collision-free, -1 otherwise
    """
    trajectory = trajectory_flat.reshape(-1, 2)
    if check_collision(trajectory, obstacles):
        return -1
    return 1


def simple_trajectory_optimization(start, goal, obstacles, num_steps=10):

    """
    Optimize a simple trajectory from start to goal while avoiding obstacles.

    :param start: numpy array or list, the starting point of the trajectory
    :param goal: numpy array or list, the goal point of the trajectory
    :param obstacles: list of numpy arrays, each array representing the vertices of an obstacle
    :param num_steps: int, the number of steps in the trajectory, default=10
    :return: numpy array, the optimized trajectory
    """
    # Initialize the trajectory
    trajectory = np.linspace(start, goal, num_steps)[1:-1]

    rng = default_rng(None)
    error = .1 * rng.standard_normal(trajectory.shape)
    trajectory = error
    # Optimize the trajectory
    res = minimize(
        simple_objective_function,
        trajectory,
        args=(start,goal,),
        method='SLSQP',
        constraints={'type': 'ineq', 'fun': collision_constraint, 'args': (obstacles,)},
#        constraints= nonlin_con
    )
    opt_traj = np.vstack([start,res.x.reshape(-1, 2), goal])
    collided = collision_constraint(opt_traj, obstacles)
    print(f"collided is {collided } ")
    

    return np.vstack([start,res.x.reshape(-1, 2), goal])

def calculate_distance(trajectory):
    """
    Calculate the total distance of a given trajectory.

    :param trajectory: numpy array, a sequence of points representing the trajectory
    :return: float, the total distance of the trajectory
    """
    return np.sum(np.sqrt(np.sum(np.diff(trajectory, axis=0) ** 2, axis=1)))

def calculate_angle2(v1, v2):
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    cosine_theta = np.dot(v1, v2) / (norm_v1 * norm_v2)
    angle = np.arccos(np.clip(cosine_theta, -1.0, 1.0))
    return np.degrees(angle)
def calculate_angle(v1, v2):
    """
    Calculate the angle between two vectors.

    :param v1: numpy array or list, the first vector
    :param v2: numpy array or list, the second vector
    :return: float, the angle between the two vectors in radians
    """
    dot_product = v1[0] * v2[0] + v1[1] * v2[1]
    norm_v1 = math.sqrt(v1[0] ** 2 + v1[1] ** 2)
    norm_v2 = math.sqrt(v2[0] ** 2 + v2[1] ** 2)
    cosine_theta = dot_product / (norm_v1 * norm_v2)
    angle = math.acos(np.clip(cosine_theta, -1, 1))
    return angle

def angle_between_vectors(goal, prev_pos, current_pos):
    v1 = np.array(goal) - np.array(prev_pos)
    v2 = np.array(current_pos) - np.array(prev_pos)
    angle = calculate_angle(v1,v2)
    return angle
def calculate_angles(trajectory, goal):
    """
    Calculate the angle between each point in the trajectory and the goal.

    Parameters:
    trajectory (np.ndarray): A numpy array with shape (n, 2) representing the 2D trajectory.
    goal (np.ndarray or list): A 1D numpy array or list with shape (2,) representing the 2D goal position.

    Returns:
    list: A list of angles in radians between each point in the trajectory and the goal.
    """
    angles = []
    for i in range(len(trajectory)):
        current_pos = trajectory[i]
        
        if i == 0:
            next_pos = trajectory[i + 1]
            angle = angle_between_vectors(goal, current_pos, next_pos)
        else:
            prev_pos = trajectory[i - 1]
            angle = angle_between_vectors(goal, prev_pos, current_pos)

        angles.append(abs(angle))
    return angles

def compare_angles(trajectory, actual_goal, decoy_goal):
    """
    Compare the angles between the trajectory points and the actual and decoy goals.

    Parameters:
    trajectory (np.ndarray): A numpy array with shape (n, 2) representing the 2D trajectory.
    actual_goal (np.ndarray or list): A 1D numpy array or list with shape (2,) representing the 2D actual goal position.
    decoy_goal (np.ndarray or list): A 1D numpy array or list with shape (2,) representing the 2D decoy goal position.

    Returns:
    float: The percentage of points in the trajectory where the angle to the actual goal is less than the angle to the decoy goal.
    """
    actual_angles = calculate_angles(trajectory, actual_goal)
    decoy_angles = calculate_angles(trajectory, decoy_goal)
    num_favorable_angles = sum(a < d for a, d in zip(actual_angles, decoy_angles))
    return (num_favorable_angles / len(trajectory)) * 100


def distance_to_goals(trajectory, actual_goal, decoy_goal):
    """
    Calculate the distance from each point in the trajectory to the actual and decoy goals.

    Parameters:
    trajectory (np.ndarray): A numpy array with shape (n, 2) representing the 2D trajectory.
    actual_goal (np.ndarray or list): A 1D numpy array or list with shape (2,) representing the 2D actual goal position.
    decoy_goal (np.ndarray or list): A 1D numpy array or list with shape (2,) representing the 2D decoy goal position.

    Returns:
    float: The percentage of points in the trajectory where the distance to the actual goal is less than the distance to the decoy goal.
    """
    actual_distances = np.sqrt(np.sum((trajectory - actual_goal) ** 2, axis=1))
    decoy_distances = np.sqrt(np.sum((trajectory - decoy_goal) ** 2, axis=1))
    num_favorable_distances = sum(a < d for a, d in zip(actual_distances, decoy_distances))
    return (num_favorable_distances / len(trajectory)) * 100
    
   
def plot_trajectory(trajectory, start=None, goal=None, decoy_goal = None,ax = None):

    """
    Plot the trajectory, start, actual goal, and decoy goal on a 2D plane.

    Parameters:
    trajectory (np.ndarray): A numpy array with shape (n, 2) representing the 2D trajectory.
    start (np.ndarray or list, optional): A 1D numpy array or list with shape (2,) representing the 2D start position. Defaults to None.
    goal (np.ndarray or list, optional): A 1D numpy array or list with shape (2,) representing the 2D actual goal position. Defaults to None.
    decoy_goal (np.ndarray or list, optional): A 1D numpy array or list with shape (2,) representing the 2D decoy goal position. Defaults to None.
    ax (matplotlib.pyplot.axis, optional): A matplotlib axis object to plot the trajectory on. Defaults to None.
    """
#    fig, ax = plt.subplots()
    ax.plot(trajectory[:, 0], trajectory[:, 1], 'o', label = 'Deceptive Path')
#    print(len(trajectory.shape[0]))
    # Set up the plot style
    plt.grid(True, linestyle='--', alpha=0.7)
    if start is not None:
        ax.plot(start[0], start[1], 'o', label='Start', markersize = 12)
    if goal is not None:
        ax.plot(goal[0], goal[1], '*', label='Actual Goal', markersize = 12)
    if decoy_goal is not None:
        ax.plot(decoy_goal[0], decoy_goal[1], '*', label='Decoy Goal', markersize = 12)
def search_path_with_min_length(rrt, min_length=1, max_iter_increment=50):
    """
    Search for a path with a minimum length using informed RRT*.

    Parameters:
    rrt (InformedRRTStar): An InformedRRTStar object.
    min_length (int, optional): The minimum length of the path. Defaults to 1.
    max_iter_increment (int, optional): The maximum number of iterations to increment in each search attempt. Defaults to 50.

    Returns:
    np.ndarray: A numpy array with shape (n, 2) representing the 2D path found.
    """
    y_short = rrt.informed_rrt_star_search(animation=False)

    while y_short is None or len(y_short) < 1:
        rrt.max_iter += max_iter_increment
        y_short = rrt.informed_rrt_star_search(animation=False)
    
    return np.array(y_short)
#    plt.show()
def compare_distances(trajectory, start, actual_goal, decoy_goal, obstacles = None,ax = None):
    """
    Compares the distances of the informed RRT* paths from each point in the trajectory to the actual and decoy goals.
    
    Args:
    trajectory (numpy.ndarray): A 2D array of trajectory points.
    start (list): The starting position [x, y].
    actual_goal (list): The actual goal position [x, y].
    decoy_goal (list): The decoy goal position [x, y].
    obstacles (list, optional): A list of obstacle vertices. Defaults to None.
    ax (matplotlib.axes.Axes, optional): The axes on which to plot the trajectory. Defaults to None.
    
    Returns:
    float: The percentage of times the actual goal is closer to the points in the trajectory than the decoy goal.
    """
    actual_distances = []
    decoy_distances = []
    print(len(trajectory))
    tolerance = 1e-2
    if True:
        for idx, point in enumerate(trajectory):
            if idx % 4 == 0:
                if idx % 2 == 0: print("computing distance")
                if not np.allclose(point, actual_goal, tolerance):
                    rrt = InformedRRTStar(start=point.tolist(), goal=actual_goal.tolist(), rand_area=[-8, 8], obstacle_list=obstacles)
#                    actual_goal_traj = np.array(rrt.informed_rrt_star_search(animation=False))
                    actual_goal_traj = search_path_with_min_length(rrt)
                    rrt2 = InformedRRTStar(start=point.tolist(), goal=decoy_goal.tolist(), rand_area=[-8, 8], obstacle_list=obstacles)
#                    decoy_goal_traj = np.array(rrt2.informed_rrt_star_search(animation=False))
                    decoy_goal_traj = search_path_with_min_length(rrt2)
                    if np.all(np.not_equal(actual_goal_traj, None)) and np.all(np.not_equal(decoy_goal_traj, None)):
                        actual_distances.append(calculate_distance(actual_goal_traj))
                        plt.plot([x for (x, y) in actual_goal_traj], [y for (x,
y) in actual_goal_traj], '-r', linewidth = 2)
                        decoy_distances.append(calculate_distance(decoy_goal_traj))
                        plt.plot([x for (x, y) in decoy_goal_traj], [y for (x,
y) in decoy_goal_traj], '-b', linewidth = 2)
        plt.plot([],[],'-r', linewidth = 2, label = 'Informed RRT* Path to Actual Goal') 
        plt.plot([],[],'-b', linewidth = 2, label = 'Informed RRT* Path to Decoy Goal') 


        correct_count = 0
        for actual_distance, decoy_distance in zip(actual_distances, decoy_distances):
            if actual_distance < decoy_distance:
                correct_count += 1

        return correct_count / len(actual_distances) * 100
    else: return 0


def evaluate_trajectory(trajectory, actual_goal, decoy_goal, obstacles,
sampled = True):
    """
    Evaluates the given trajectory by comparing angles, distances, and calculating ADII and DDII.

    Args:
    trajectory (list): A list of 2D points representing the trajectory.
    actual_goal (list): The actual goal position [x, y].
    decoy_goal (list): The decoy goal position [x, y].
    obstacles (list): A list of obstacle vertices.

    Returns:
    tuple: A tuple containing the following metrics:
        - distance_traveled (float): The total distance traveled along the trajectory.
        - angle_comparison_percentage (float): The percentage of times the angle favors the actual goal.
        - distance_comparison_percentage (float): The percentage of times the distance favors the actual goal.
        - ADII (float): The Alignment Deception Impact Index.
        - DDII (float): The Distance Deception Impact Index.
    """
    # Calculate distance traveled
    unsampled_traj = trajectory
    trajectory = np.array(resample_points(trajectory))
    distance_traveled = calculate_distance(trajectory)
    
    # Compare angles
    if sampled == True:angle_comparison_percentage = compare_angles(trajectory, actual_goal, decoy_goal)
    else:angle_comparison_percentage = compare_angles(unsampled_traj, actual_goal, decoy_goal)
    

    plt.style.use('seaborn-darkgrid')
    # Compare distances to goals
    fig, ax = plt.subplots(figsize=(10, 8))
    start = trajectory[0]
    if sampled == True:    distance_comparison_percentage = compare_distances(trajectory, start, actual_goal, decoy_goal, obstacles,ax)
    else: distance_comparison_percentage = compare_distances(unsampled_traj, start, actual_goal, decoy_goal, obstacles,ax)
    if sampled == True: plot_trajectory(trajectory, start = trajectory[0], goal = actual_goal, decoy_goal = decoy_goal, ax = ax)
    else: plot_trajectory(unsampled_traj, start = trajectory[0], goal = actual_goal, decoy_goal = decoy_goal, ax = ax)
    for obstacle in obstacles:
        plt.plot(*obstacle.T, 'k-', linewidth=1.5)
        plt.fill(*obstacle.T, color='#2c3e50', alpha=0.7)
    # Create custom legend entries
    plt.xlabel('X', fontsize=14, fontweight='bold')
    plt.ylabel('Y', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11, loc='lower right', prop={'weight': 'bold'})
    plt.show()
        
    # Calculate ADII and DDII
    ADII = (angle_comparison_percentage / 100) * distance_traveled
    DDII = (distance_comparison_percentage / 100) * distance_traveled

    # Print the results
    print(f'number of points: {trajectory.shape[0]}')
    print("Distance traveled: {:.2f}".format(distance_traveled))
    print("Percentage of times angle favors actual goal: {:.2f}%".format(angle_comparison_percentage))
    print("Percentage of times distance favors actual goal: {:.2f}%".format(distance_comparison_percentage))
    print("Alignment Deception Impact Index (ADII): {:.2f}".format(ADII))
    print("Distance Deception Impact Index (DDII): {:.2f}".format(DDII))


    return distance_traveled, angle_comparison_percentage, distance_comparison_percentage, ADII, DDII
