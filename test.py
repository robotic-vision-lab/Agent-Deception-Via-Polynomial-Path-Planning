import numpy as np
from utils.point_resampler import resample_points
from utils.deception_evaluation import *
from utils.obs_full import *
import math
import os
from pdb import set_trace as trace
import sys
import numpy as np
from numpy.random import default_rng
from numpy import ones,vstack
from numpy.linalg import lstsq

import matplotlib.pyplot as plt
from utils.obstacle_utils import *
from utils.point_resampler import add_noise
from matplotlib.animation import PillowWriter

from matplotlib.animation import FuncAnimation
from scipy.optimize import least_squares
import argparse

from scipy.interpolate import interp1d
rng = default_rng()
"""
https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html
https://www.osti.gov/servlets/purl/4102888
 
"""
parser = argparse.ArgumentParser()
parser.add_argument('--t_max', type=float, default = 10)
parser.add_argument('--n_points', type=int, default = 50)
###############################################################################
### ON/OFFs
###############################################################################
parser.add_argument('--circle_on', default=False, action='store_true')
parser.add_argument('--point_on', default=False, action='store_true')
parser.add_argument('--curvature_on', default=False, action='store_true')
parser.add_argument('--ambiguity_on', default=False, action='store_true')
parser.add_argument('--eval_on', default=False, action='store_true')
parser.add_argument('--animate_on', default=False, action='store_true')
parser.add_argument('--beta_test', default=False, action='store_true')
parser.add_argument('--poly_test', default=False, action='store_true')
parser.add_argument('--ambiguity_test', default=False, action='store_true')
parser.add_argument('--obs_beta_test', default=False, action='store_true')
parser.add_argument('--noise_test', default=False, action='store_true')
parser.add_argument('--curve_test', default=False, action='store_true')
parser.add_argument('--reg_test', default=False, action='store_true')
parser.add_argument('--comb_test', default=False, action='store_true')
parser.add_argument('--short_on', default=False, action='store_true')
parser.add_argument('--obs_on', default=False, action='store_true')
parser.add_argument('--plot_show', default=False, action='store_true')
parser.add_argument('--constraints_off', default=False, action='store_true')

###############################################################################
### BETAS
###############################################################################
parser.add_argument('--reg_on', default=False, action='store_true')
parser.add_argument('--circle_beta', type=float, default = 20)
parser.add_argument('--beta', type=float, default = 1)
parser.add_argument('--point_beta', type = float, default = 10)
parser.add_argument('--obs_beta', type = float, default = 1)
parser.add_argument('--reg_beta', type = float, default = 100000)
parser.add_argument('--goal_angle_beta', type = float, default = 1)
parser.add_argument('--alt_angle_beta', type = float, default = 1)

###############################################################################
### Other Hyperparameters
###############################################################################
parser.add_argument('--noise', type=float, default=0)
parser.add_argument('--degree', type=float, default = 4)
parser.add_argument('--min_distance', type=float, default = .2)
parser.add_argument('--number_alternatives', type=float, default = 1)
parser.add_argument('--circle_location', type=str, default='(4,2)')
parser.add_argument('--point_location', type = str, default = '(6,3)')
parser.add_argument('--start_location', type=str, default='(0,1)')
parser.add_argument('--alternative_goals', type=str, default='([5,5],)')
parser.add_argument('--goal', type=str, default='[10,2]')
parser.add_argument('--points', type=str, default='([6,3],)')
parser.add_argument('--obs_points', type=str, default='( [5,1],)')
parser.add_argument('--traj_folder_prefix', type = str, default = '')
parser.add_argument('--title', type=str, default='Ambiguity for Different Beta Values')
parser.add_argument('--use_strategy_4', action='store_true', help='Use strategy 4 for final path planning.')
args = parser.parse_args()
iterations = 0
def interpolate_points(points, num_points=40):
    """
    Interpolate a list of xy coordinates to obtain a specified number of points.

    Args:
    points (list of tuples): A list of (x, y) coordinates.
    num_points (int): The desired number of points in the output.

    Returns:
    np.ndarray: A numpy array of interpolated (x, y) coordinates.
    """
    points = np.array(points)
    n_segments = len(points) - 1
    points_per_segment = num_points // n_segments

    # Parameterize the points using the cumulative distance along the path
    distances = np.sqrt(np.sum(np.diff(points, axis=0)**2, axis=1))
    cumulative_distances = np.hstack(([0], np.cumsum(distances)))
    total_distance = cumulative_distances[-1]

    # Create linear interpolators for x and y coordinates
    x_interpolator = interp1d(cumulative_distances, points[:, 0])
    y_interpolator = interp1d(cumulative_distances, points[:, 1])

    # Generate the new cumulative distances for the desired number of points
    new_cumulative_distances = np.linspace(0, total_distance, num_points)

    # Interpolate the x and y coordinates for the new cumulative distances
    new_x_coordinates = x_interpolator(new_cumulative_distances)
    new_y_coordinates = y_interpolator(new_cumulative_distances)

    # Combine the interpolated x and y coordinates into a single array
    interpolated_points = np.vstack((new_x_coordinates, new_y_coordinates)).T

    return interpolated_points

def save_trajectory(trajectory, name, separator='_'):
    if not os.path.exists(args.traj_folder_prefix + 'trajectories'):
        os.makedirs( args.traj_folder_prefix + 'trajectories')

    file_name = args.traj_folder_prefix + f"trajectories/{name}{separator}.npy"
    np.save(file_name, trajectory)

    # Generate
def calculate_angle(v1, v2):
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

def unit_vector(vector):
    """Returns the unit vector of the input vector.

    Args:
        vector (numpy.ndarray): A vector represented as a NumPy array.

    Returns:
        numpy.ndarray: A unit vector representing the direction of the input vector.

    """
    return vector / np.linalg.norm(vector)

def multi_unit_vector(vector):
    """Returns an array of unit vectors corresponding to a given array of vectors.

    Args:
        vector (numpy.ndarray): An array of vectors represented as a NumPy array.

    Returns:
        numpy.ndarray: An array of unit vectors representing the direction of each vector in the input array.

    """
    new_vector = vector.copy()
    for idx, vec in enumerate(vector):
        new_vector[idx] = unit_vector(vec)
    return new_vector


def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::
        https://stackoverflow.com/questions/2827393/angles-between-two-n-dimensional-vectors-in-python

            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    """
    v1_u = multi_unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))
def get_polynomial_data(t, c, polynomial_type, cstart):
    """
    Computes the polynomial function for the given type and coefficients at each given value of t.
    
    Args:
        t (float): A single value or an array of values to compute the polynomial at.
        c (list): A list of coefficients for the polynomial.
        polynomial_type (int): An integer representing the degree of the polynomial.
        cstart (float): A constant value added to the computed polynomial.
    
    Returns:
        float or numpy.ndarray: The computed polynomial value(s) for the given t value(s) and coefficients.
    """

    
    if polynomial_type == 1: return cstart+ c[1] * t
    elif polynomial_type == 2: return cstart  + c[1] * t + c[2]* t**2
    elif polynomial_type == 3: return cstart + c[1] *t  + c[2] * t**2+  c[3] * t**3
    elif polynomial_type == 4: return cstart + c[1] * t + c[2]* t**2 + c[3] * t**3 + c[4] * t**4
    elif polynomial_type == 5: return cstart + c[1] * t + c[2]* t**2 + c[3] * t**3 + c[4] * t**4 + c[5] * t**5
    elif polynomial_type == 6: return cstart + c[1] * t + c[2]* t**2 + c[3] * t**3 + c[4] * t**4 + c[5] * t**5 + c[6] * t**6
    elif polynomial_type == 7: return cstart + c[1] * t + c[2]* t**2 + c[3] * t**3 + c[4] * t**4 + c[5] * t**5 + c[6] * t**6 + c[7] * t**7
    elif polynomial_type == 8: return cstart + c[1] * t + c[2]* t**2 + c[3] * t**3 + c[4] * t**4 + c[5] * t**5 + c[6] * t**6 + c[7] * t**7 + c[8] * t**8
    elif polynomial_type == 9: return cstart + c[1] * t + c[2]* t**2 + c[3] * t**3 + c[4] * t**4 + c[5] * t**5 + c[6] * t**6 + c[7] * t**7 + c[8] * t**8 + c[9] * t**9
    elif polynomial_type == 10: return cstart + c[1] * t + c[2]* t**2 + c[3] * t**3 + c[4] * t**4 + c[5] * t**5 + c[6] * t**6 + c[7] * t**7 + c[8] * t**8 + c[9] * t ** 9 + c[10] * t**10
    elif polynomial_type == 11: return cstart + c[1] * t + c[2]* t**2 + c[3] * t**3 + c[4] * t**4 + c[5] * t**5 + c[6] * t**6 + c[7] * t**7 + c[8] * t**8 + c[9] * t ** 9 + c[10] * t**10 + c[11] * t **11
def get_distance_along_curve(points): 

    """
    Calculates the total length of a path given a list of points.

    Parameters:
    points (list): A list of 2D points representing a path.

    Returns:
    float: The total length of the path.
    """
    total_distance = 0
    for i in range(1, len(points)):
        x1, y1 = points[i-1]
        x2, y2 = points[i]
        segment_distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        total_distance += segment_distance
    return total_distance
def get_distance_residual(path): 
    """
    Calculates the Euclidean distance between consecutive points in a path.

    Args:
    path (list of tuples): A list of (x, y) coordinates representing a path.

    Returns:
    np.ndarray: A 2D numpy array of distances between consecutive points in the path. Each row represents a single distance.
    """
    displacements = np.diff(path, axis=0)
    distances = np.sqrt(np.sum(displacements**2, axis=1))
    distance_array = np.stack((distances, distances), axis=-1)
    zero_row = np.zeros((1, 2))
    return np.concatenate((zero_row, distance_array), axis=0)
def get_polynomial_derivative(t,c,polynomial_type, cstart):
     
    """
    :param t: The value of x at which to evaluate the derivative
    :type t: float
    :param c: Coefficients of the polynomial, in the form [c0, c1, c2, ..., cn], where c0 is the constant term and cn is the coefficient of the highest degree term.
    :type c: list
    :param polynomial_type: The degree of the polynomial function. Must be an integer between 1 and 11.
    :type polynomial_type: int
    :type cstart: int

    :returns: The value of the derivative of the polynomial function at x=t.
    :rtype: float
    """
    
    if polynomial_type == 1: return  c[1] 
    elif polynomial_type == 2: return c[1] + c[2]* t
    elif polynomial_type == 3: return c[1] + c[2]* t  +  c[3] * t**2
    elif polynomial_type == 4: return c[1] + c[2]* t  +  c[3] * t**2 + c[4] * t**3
    elif polynomial_type == 5: return c[1] + c[2]* t  +  c[3] * t**2 + c[4] * t**3 + c[5] * t**4
    elif polynomial_type == 6: return c[1] + c[2]* t  +  c[3] * t**2 + c[4] * t**3 + c[5] * t**4 + c[6] * t**5
    elif polynomial_type == 7: return c[1] + c[2]* t  +  c[3] * t**2 + c[4] * t**3 + c[5] * t**4 + c[6] * t**5 + c[7] * t**6
    elif polynomial_type == 8: return c[1] + c[2]* t  +  c[3] * t**2 + c[4] * t**3 + c[5] * t**4 + c[6] * t**5 + c[7] * t**6 + c[8] * t**7
    elif polynomial_type == 9: return c[1] + c[2]* t  +  c[3] * t**2 + c[4] * t**3 + c[5] * t**4 + c[6] * t**5 + c[7] * t**6 + c[8] * t**7 + c[9] * t**8
    elif polynomial_type == 10: return c[1] + c[2]* t  +  c[3] * t**2 + c[4] * t**3 + c[5] * t**4 + c[6] * t**5 + c[7] * t**6 + c[8] * t**7 + c[9] * t**8 + c[10] * t**9
    elif polynomial_type == 11: return c[1] + c[2]* t  +  c[3] * t**2 + c[4] * t**3 + c[5] * t**4 + c[6] * t**5 + c[7] * t**6 + c[8] * t**7 + c[9] * t**8 + c[10] * t**9 + c[11] * t **10

def get_line_to_point(x1, y1, x2, y2, t):
    """
    Returns the slope and y-intercept of the line defined by two points.

    Args:
    x1 (float): x-coordinate of the first point
    y1 (float): y-coordinate of the first point
    x2 (float): x-coordinate of the second point
    y2 (float): y-coordinate of the second point
    t (float): input value of t

    Returns:
    tuple: slope and y-intercept of the line
    """
    points = [(x1, y1), (x2, y2)]
    x_coords, y_coords = zip(*points)
    A = vstack([x_coords, ones(len(x_coords))]).T
    m, c = lstsq(A, y_coords)[0]
    return m, c

def gen_data(t, x1, y1, x2, y2, noise=0., n_outliers=0, seed=None):
    """
    Generates noisy data points along a line defined by two points.

    Args:
    t (ndarray): array of input values of t
    x1 (float): x-coordinate of the first point
    y1 (float): y-coordinate of the first point
    x2 (float): x-coordinate of the second point
    y2 (float): y-coordinate of the second point
    noise (float): standard deviation of the noise
    n_outliers (int): number of outlier points to add
    seed (int): seed for the random number generator

    Returns:
    ndarray: noisy data points along the line defined by the two points
    """
    rng = default_rng(seed)

    m, b = get_line_to_point(x1, y1, x2, y2, t)
    y = m * t + b

    error = noise * rng.standard_normal(t.size)
    outliers = rng.integers(0, t.size, n_outliers)
    error[outliers] *= 10

    return y + error

def search_path_with_min_length(rrt, min_length=1, max_iter_increment=50):
    y_short = rrt.informed_rrt_star_search(animation=False)

    while y_short is None or len(y_short) < 1:
        rrt.max_iter += max_iter_increment
        y_short = rrt.informed_rrt_star_search(animation=False)
    
    return np.array(y_short)

def gen_data_for_t(t, x1, y1, x2, y2, noise=0., n_outliers=0, seed=None):
    """
    Generates noisy data points for a moving object along a straight line path.

    Args:
    t (ndarray): array of input values of t
    x1 (float): x-coordinate of the start point
    y1 (float): y-coordinate of the start point
    x2 (float): x-coordinate of the end point
    y2 (float): y-coordinate of the end point
    noise (float): standard deviation of the noise
    n_outliers (int): number of outlier points to add
    seed (int): seed for the random number generator

    Returns:
    ndarray: noisy
    """

    rng = default_rng(seed)
    x0 = np.array([[x1,y1]])
    v = np.array([[x2 - x1, y2-y1]])
    speed = t[-1] - t[0]

    x = x0 + t @ v/speed

    error = noise * rng.standard_normal(x.shape)
    outliers = rng.integers(0, x.shape[0], n_outliers)

    xerror = x + error
    
    return xerror
def gen_data_polynomial(t, c, noise=0.,n_outliers=0, seed=None,xstart=1, polynomial_type = 2):
    """
    Generates polynomial data based on the input arguments.
    
    Args:
    - t: 1D array of floats representing the input values.
    - c: 1D array of floats representing the polynomial coefficients.
    - noise: float representing the standard deviation of the noise.
    - n_outliers: integer representing the number of outliers to add.
    - seed: integer representing the seed value for the random number generator.
    - xstart: float representing the initial value of the x-coordinate.
    - polynomial_type: integer representing the degree of the polynomial function.
    
    Returns:
    - y: 1D array of floats representing the output values.
    """
    rng = default_rng(seed)
    y = get_polynomial_data(t, c, polynomial_type, xstart)

    error = noise * rng.standard_normal(y.shape)
    outliers = rng.integers(0, t.size, n_outliers)

    error[outliers] *= 10

    return y + error

def get_residual(x,t,y, xstart, polynomial_type): 
    """
    Computes the residual between the actual and predicted values of a polynomial function.
    
    Args:
    - x: 1D array of floats representing the polynomial coefficients.
    - t: 1D array of floats representing the input values.
    - y: 1D array of floats representing the output values.
    - xstart: float representing the initial value of the x-coordinate.
    - polynomial_type: integer representing the degree of the polynomial function.
    
    Returns:
    - residual: 1D array of floats representing the difference between the predicted and actual values.
    """
    return get_polynomial_data(t, x, polynomial_type, xstart) - y

if __name__ == "__main__":
    n_points = int(args.n_points)
    n_amb_points = 10
    x0 = np.random.normal(size = (13,))
    y0 = np.random.normal(size = (13,))
    xy0 = np.hstack([x0,y0])
    ###########################################################################
    ### Storing Parsed Arguments
    ###########################################################################
    polynomial_type = args.degree
    beta = args.beta
    print(f"beta is {beta } ")
    noise = args.noise
    circle_on = args.circle_on
    constraints_off = args.constraints_off
    point_on = args.point_on
    circle_beta = args.circle_beta
    point_beta = args.point_beta
    obs_beta = args.obs_beta
    goal_angle_beta = args.goal_angle_beta
    alt_angle_beta = args.alt_angle_beta
    reg_beta = args.reg_beta
    circle_location = eval(args.circle_location)
    point_location = eval(args.point_location)
    start_location = eval(args.start_location)
    points = eval(args.points)
    obstacle_1 = (3, 2, np.array([5.5,7])) # (length, high, center_coordinates)
    obstacles = []
    obs_points = [convert_obstacle_to_vertices(obstacle) for obstacle in obstacles]
    title = args.title
    plot_show = args.plot_show
    xstart = start_location[0]
    ystart = start_location[1]
    t_min = 0
    t_max = args.t_max
    goals = eval(args.alternative_goals)
   ############################################################################
    
    realgoal = eval(args.goal)

    print(f"args.ambiguity_on is {args.ambiguity_on } ")
    print(f"args.curvature_on is {args.curvature_on } ")
    print(f"args.reg_on is {args.reg_on } ")
    paths = []

    ####################################################################
    ###  Replace opt code with scipy opt code
    ####################################################################
        
    def objective(xy, t, xstart, ystart, real_goal, goals, stop_points,
               circle_point, polynomial_type, curvature_on=args.curvature_on,
               reg_on=args.reg_on, obs_on=args.obs_on, ambiguity_on=args.ambiguity_on,
               circle_on=args.circle_on, point_on=args.point_on, short_on=args.short_on,
               beta=1, obs_beta=10, goal_angle_beta=1, alt_angle_beta=1, noise=0.1,
               reg_beta=1, point_beta=args.point_beta, circle_beta=1):
        """
        Computes the objective function for polynomial fitting and path planning.

        Args:
        - xy (numpy array): A 1D array of size 26 containing the x and y coordinates of all the points in the trajectory.
        - t (numpy array): A 1D array of size 100 representing the time points.
        - xstart (float): A float representing the initial x-coordinate.
        - ystart (float): A float representing the initial y-coordinate.
        - real_goal (tuple): A tuple of the form (x, y) representing the actual goal location.
        - goals (list): A list of tuples of the form (x, y) representing alternative goal locations.
        - stop_points (list): A list of tuples of the form (x, y) representing stop points.
        - circle_point (list): A list of tuples of the form (x, y) representing circle points.
        - polynomial_type (int): An integer representing the degree of the polynomial function.
        - curvature_on (bool): A boolean representing whether or not to include the curvature term in the objective function.
        - reg_on (bool): A boolean representing whether or not to include the regularization term in the objective function.
        - obs_on (bool): A boolean representing whether or not to include obstacle avoidance in the objective function.
        - ambiguity_on (bool): A boolean representing whether or not to include ambiguity resolution in the objective function.
        - circle_on (bool): A boolean representing whether or not to include circle constraints in the objective function.
        - point_on (bool): A boolean representing whether or not to include point constraints in the objective function.
        - short_on (bool): A boolean representing whether or not to include the shortest path constraint in the objective function.
        - beta (float): A float representing the weight for ambiguity resolution and constraint penalties.
        - obs_beta (float): A float representing the weight for obstacle avoidance constraint penalties.
        - goal_angle_beta (float): A float representing the weight for goal angle constraint penalties.
        - alt_angle_beta (float): A float representing the weight for alternative angle constraint penalties.
        - noise (float): A float representing the amount of noise to add to the data.
        - reg_beta (float): A float representing the weight for the regularization term.
        - point_beta (float): A float representing the weight for point constraint penalties.
        - circle_beta (float): A float representing the weight for circle constraint penalties.

        Returns:
        - total_loss (float): A float representing the total loss of the objective function.
        """
        residuals = []
        #######################################################################
        ### Generating F(x)
        #######################################################################
        x_data =  get_polynomial_data(t,xy[:13], polynomial_type,xstart)
        y_data = get_polynomial_data(t,xy[13:],polynomial_type, ystart)
        f_data = np.hstack([x_data, y_data])
        #######################################################################
        ### Generating F'(x)
        #######################################################################
        x_prime_data =  get_polynomial_derivative(t,xy[:13], polynomial_type,xstart)
        y_prime_data = get_polynomial_derivative(t,xy[13:],polynomial_type, ystart)
        f_prime_data = np.hstack([x_data, y_data])
        f_angles = np.arctan(f_prime_data)
        #######################################################################
        ### Goal_Angles and Alternative Angles
        #######################################################################
        goal_angle =np.array(calculate_angles(f_data, realgoal)) 
        for goal in goals: 
            alt_angle = np.array(calculate_angles( f_data, goal))
        #######################################################################
        ###  Curvature Residuals
        #######################################################################
            if curvature_on:
                # Create an array of shape (goal_angle.shape[0], 2) with zeros
                angle_residual = np.zeros((goal_angle.shape[0], 2))
                # Calculate the difference between alt_angle and goal_angle when alt_angle > goal_angle
                angle_diff = np.where(alt_angle - goal_angle >0, alt_angle - goal_angle, alt_angle)
                angle_residual[:, 0] = angle_diff
                angle_residual[:, 1] = 0
                residuals.append(alt_angle_beta * 1 * angle_residual)
            
        #######################################################################
        ### Adding Regularization Term
        #######################################################################
        if reg_on:
            regularization = reg_beta *   np.ones((args.n_points,2)) * np.sum(xy**2)
            residuals.append(regularization)
        total_loss  = 0
        ###################################################################
        ### short path residual
        ###################################################################
        loss_sum = np.zeros((args.n_points,2))
        global iterations
        if short_on:
            if not ambiguity_on and not circle_on:
                if iterations < 1000: loss_sum = 1e9 *  (f_data - y_short) 
                else: loss_sum = 1 *  (f_data - y_short) 
            else: 
                loss_sum = .1*  (f_data - y_short) 
            loss_sum[-1] = (f_data[-1] - y_short[-1]) * 1e10
            residuals.append(loss_sum)
            total_loss += loss_sum
        for idx,goal in enumerate(goals): 
         
            y_train = y_trains[idx]
            if iterations < 40:
                if idx == 0: ambiguity_loss = ( 1e9 * (f_data - y_train))
                else: ambiguity_loss += (1e9 *(f_data - y_train))
            else: 
                if idx == 0: ambiguity_loss = ( beta * (f_data - y_train))
                else: ambiguity_loss += (beta *(f_data - y_train))
            if not short_on: 
                ambiguity_loss[-1] = ambiguity_loss[-1]  * 1e6
        ### ambiguity residual
        #########################0##########################################
        if ambiguity_on: residuals.append((ambiguity_loss))
       
        if circle_on is True:
            circle_function = np.concatenate([circle_location[0] + -np.cos(t_train), circle_location[1] + -np.sin(t_train)], axis = 1)
            circle_loss =0
            circle_residual = circle_beta * (f_data - circle_function)
            residuals.append(circle_residual)
        if point_on is True: 
            for point_location in points:
                point_x = np.expand_dims(np.array([point_location[0] for i in range(t_train.shape[0])]), axis = 1)
                point_y = np.expand_dims(np.array([point_location[1] for i in range(t_train.shape[0])]), axis = 1)
                point_function = np.concatenate([point_x, point_y], axis = 1)
                point_indicator = np.zeros((args.n_points,2))
                point_indicator =  point_beta * (f_data - point_function)
                
                residuals.append(point_indicator)
        if obs_on is True: 
            ###############################################################
            ### Obstacles
            ###############################################################
            if iterations > 50: distances_to_obs = 1 * obstacle_term(obs_points, t_train, f_data, radius = .2)
            else: distances_to_obs = 0 * obstacle_term(obs_points, t_train,
f_data, radius = 2)
            residuals.append(distances_to_obs)

        dis_res =get_distance_residual(f_data)
        dis_res1 = dis_res - np.mean(dis_res, axis = 0) 
        residuals.append( 4000 *(dis_res))
        if iterations > 20: residuals.append(500*dis_res1)
        else: residuals.append(1 * dis_res1)
        residual_array = np.hstack(residuals).ravel()
        iterations += 1
        return residual_array
    
    ###########################################################################
    ### Finding solution
    ###########################################################################
    if not args.poly_test: poly_iter = [args.degree] 
    elif args.poly_test: poly_iter = range(1, int(args.degree))
    
    if not args.beta_test: beta_iter = [args.beta]
    elif args.beta_test: beta_iter = [ 0.01, 0.1, 0.5, 1, 10, 100]
    
    if not args.noise_test: noise_iter = [args.noise] 
    elif args.noise_test: noise_iter = [0, 0.01, 0.1, 0.2, 0.4, 0.8, 1]

       
   
    for polynomial_type in poly_iter:
       for beta_value  in beta_iter:  
           for noise in noise_iter: 
                t_train = np.expand_dims(np.linspace(t_min, t_max, n_points), axis = 1).astype(np.float64)
                y_short = gen_data_for_t(t_train, xstart,ystart,realgoal[0],realgoal[1], noise=noise, n_outliers=2).astype(np.float64)


                rrt = InformedRRTStar(start=[xstart, ystart], goal=realgoal, rand_area=[-8, 8], obstacle_list=obs_points)
                y_short = search_path_with_min_length(rrt)
                y_short = np.flip(interpolate_points(y_short, num_points = t_train.shape[0]), axis=0)
                y_short = add_noise(y_short, noise=noise) 
                y_trains = []
                for idx, goal in enumerate(goals):
                    if args.use_strategy_4:
                        y_data, grid = get_final_path([xstart, ystart], realgoal, goal, obstacles, strategy = 3, resolution=10, grid_size=200, return_standard=False)
                        y_data = interpolate_points(np.array(y_data), num_points = t_train.shape[0])  
                    else:
                        rrt = InformedRRTStar(start=[xstart, ystart], goal=goal, max_iter=100, rand_area=[-10, 10], obstacle_list=obs_points)
                        y_data = search_path_with_min_length(rrt)
                        y_data = np.flip(interpolate_points(y_data, num_points = t_train.shape[0]), axis=0)
                        y_data = add_noise(y_data, noise=noise)
                    y_trains.append(y_data)
                if circle_on: 
                    circle_function = np.concatenate([circle_location[0] + -np.cos(t_train), circle_location[1] + -np.sin(t_train)], axis = 1)
                    y_trains.append(circle_function)
                plot_string = ""
                if args.poly_test: plot_string = f"degree = {polynomial_type}"
                if args.noise_test: plot_string= f"noise = {noise}"
                if args.ambiguity_test: 
                    beta = beta_value
                    plot_string = f"beta = {beta_value}"
                if args.obs_beta_test: 
                    obs_beta = beta_value 
                    plot_string = f"obs factor = {beta_value}"
                if args.curve_test: 
                    goal_angle_beta = beta_value 
                    alt_angle_beta = beta_value
                    plot_string = f"angle factor = {beta_value}"
                if args.reg_test: 
                    reg_beta = beta_value
                    plot_string = f"reg factor = {beta_value}"
                if args.comb_test: 
                    plot_string = f"degree = {polynomial_type} noise = {noise} angle factor = {alt_angle_beta} beta = {beta}"
                      
                    
                res_lsq  = least_squares(objective, xy0, loss =
'linear',ftol=1e-8, xtol=1e-8, args = (t_train,xstart,ystart, realgoal, goals, points, circle_location,polynomial_type), kwargs = {"beta": beta, "obs_beta": obs_beta, "obs_beta": obs_beta, "goal_angle_beta": goal_angle_beta, "alt_angle_beta": alt_angle_beta, "noise": noise, "reg_beta": reg_beta, "circle_beta": circle_beta, "point_beta": point_beta})
                
                xresult = res_lsq.x[:13]
                yresult = res_lsq.x[13:]
                
                point_x = np.expand_dims(np.array([points[0][0] for i in range(t_train.shape[0])]), axis = 1)
                point_y = np.expand_dims(np.array([points[0][1] for i in range(t_train.shape[0])]), axis = 1)
                point_function = np.concatenate([point_x, point_y], axis = 1)
                
                
              
                t_test = np.expand_dims(np.linspace(t_min, t_max, args.n_points ) , axis =1)
                y_lsq = gen_data_polynomial(t_test, yresult,polynomial_type=polynomial_type, xstart=ystart)
                x_lsq = gen_data_polynomial(t_test, xresult,polynomial_type=polynomial_type, xstart=xstart)
                ###############################################################################
                ### Evaluation of Paths
                ###############################################################################
                path = np.concatenate([x_lsq,y_lsq], axis =1 ) 
                da, ac, dc, ADEI,DDEI = evaluate_trajectory(path, np.array(realgoal), np.array(goals[0]),obs_points, sampled = False)
                da, ac, dc, ADEI,DDEI = evaluate_trajectory(path, np.array(realgoal), np.array(goals[0]),obs_points, sampled = True)
                DT = "_DT: {:.2f}_".format(da)
                AP = "_ADI: {:.2f}%_".format(ac)
                DP = "_DDI: {:.2f}%_".format(dc)
                NADDEI = "ADII: {:.2f}_".format(ADEI)
                NDDDEI = "DDII: {:.2f}_".format(DDEI)
                paths.append((path, plot_string, DT + AP + DP+ NADDEI + NDDDEI))
              
    for idx,goal in enumerate(goals): 
        y_train = y_trains[idx]
        plt.plot(y_train[:,0], y_train[:,1],'o', label = 'Alternative Path')

    circle_function = np.concatenate([circle_location[0] + -np.cos(t_train), circle_location[1] + -np.sin(t_train)], axis = 1)
    ###########################################################################
    ### Plots
    ###########################################################################
    if point_on: plt.plot(point_function[:,0], point_function[:,1],'o', label = 'Stopping Point')
    plt.plot(y_short[:,0], y_short[:,1],'o', label = 'Shortest Path Function')

    for path, plot_string, metric_string in paths: 
        plt.plot(path[:,0], path[:,-1], label = plot_string)
    for path, plot_string, metric_string in paths:
        save_trajectory(path, args.title + " " + plot_string +  metric_string)
    save_trajectory(y_data, "altpath Alternative Path Function")
    if circle_on is True: plt.plot(circle_function[:,0], circle_function[:,1], 'o', label = 'Circle Path')
    ###############################################################################
    ### Plotting Obstacles
    ###############################################################################
    for obstacle in obs_points:
        plt.plot(*obstacle.T, 'k-')
        plt.fill(*obstacle.T, 'gray', alpha=0.5)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.grid()
    plt.plot(realgoal[0],realgoal[1],'o', label="Actual Goal", markersize=10)
    plt.plot(xstart,ystart,'o', label="Start",markersize=10)
    plt.legend(ncol=2,prop={'size': 8})
    if circle_on: title = 'Circle ' + title
    if point_on: title = 'Point ' + title
    if args.curvature_on: title = 'Curvature ' + title
    if args.obs_on: title = 'Obstacle Avoidance ' + title
    if args.reg_on: title = 'Reg ' + title
    test_list = []
    if args.poly_test: test_list.append('poly test ') 
    if args.noise_test: test_list.append('noise test ') 
    if args.ambiguity_test: 
        test_list.append('ambiguity test ') 
    if args.obs_beta_test: 
        test_list.append('obstacle avoidance test ') 
    if args.curve_test: 
        test_list.append('curvature test ') 
    if args.reg_test: 
        test_list.append('regularization test ') 
    if args.comb_test:
        test_list.append('combination test ') 
    titlename = title
    plt.title(args.title)
    titlestring = test_list + [ str(args.curvature_on), 'curvature_bool', str(reg_beta), 'reg_beta', str(polynomial_type), 'degrees', str(noise), 'noise',
str(beta), 'beta', str(circle_on), 'circle_on', str(xstart),'xstart',
str(ystart), 'ystart', str(realgoal), 'Goal', str(t_max), 't_max.png' ]
    if plot_show is True: plt.show()
    plt.savefig('images/' + '_'.join(titlestring).replace(' ', '_'), bbox_inches='tight')
                 
