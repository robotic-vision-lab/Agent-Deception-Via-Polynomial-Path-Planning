import numpy as np
from pdb import set_trace as trace
import sys

from scipy.interpolate import interp1d

import matplotlib.pyplot as plt
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
def add_noise(trajectory, noise=0.0):
    """
    Adds noise to a given trajectory while keeping the start and goal points unchanged.

    Parameters:
    trajectory (numpy array): A numpy array with shape (N, 2) representing the trajectory.
    noise_std (float): The standard deviation of the Gaussian noise to be added.

    Returns:
    numpy array: A numpy array with shape (N, 2) representing the trajectory with added noise.
    """
    noise_std = noise
    noise = np.random.normal(0, noise_std, trajectory.shape)
    
    # Keep the start and goal points unchanged
    noise[0] = 0
    noise[-1] = 0

    noisy_trajectory = trajectory + noise

    return noisy_trajectory



def resample_points(points, min_distance = 2, num_points = 10):
    """
    Resamples a set of points by adding new points in between if the distance between two
    consecutive points is larger than the specified minimum distance.

    Parameters:
    points (list): A list of tuples containing the (x, y) coordinates of the points.
    min_distance (float): The minimum distance between two consecutive points.

    Returns:
    list: A list of tuples containing the (x, y) coordinates of the resampled points.
    """
    new_points = [points[0]]
    for i in range(1, len(points)):
        p1 = points[i-1]
        p2 = points[i]
        distance = np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
        if distance > min_distance:
            n_points = int(distance // min_distance) + 1
            xs = np.linspace(p1[0], p2[0], n_points)
            ys = np.linspace(p1[1], p2[1], n_points)
            new_points.extend(list(zip(xs[1:], ys[1:])))
        else:
            new_points.append(p2)

    resampled_points = []
    total_distance = 0.0

    for i in range(1, len(new_points)):
        total_distance += np.sqrt((new_points[i][0] - new_points[i-1][0])**2 + (new_points[i][1] - new_points[i-1][1])**2)

    mean_distance = total_distance / (num_points - 1)

    resampled_points.append(new_points[0])
    current_point = new_points[0]
    for point in new_points:
        distance = np.sqrt((point[0] - current_point[0])**2 + (point[1] - current_point[1])**2)
        while distance > mean_distance:
            t = mean_distance / distance
            x = (1 - t) * current_point[0] + t * point[0]
            y = (1 - t) * current_point[1] + t * point[1]
            resampled_points.append((x, y))
            current_point = (x, y)
            distance = np.sqrt((point[0] - current_point[0])**2 + (point[1] - current_point[1])**2)
    
    resampled_points.append(points[-1])  # Make sure to include the last point

    return resampled_points

def exact_resample(points, num_points=10):
    """
    Resample a set of points to produce exactly `num_points` points equally spaced
    along the curve defined by the input points.

    Parameters:
    points (list of tuples): Input (x, y) points defining a path.
    num_points (int): Number of points in the output.

    Returns:
    list of tuples: Resampled (x, y) points.
    """
    points = np.array(points)
    # Compute distances between points and cumulative arc length
    distances = np.sqrt(np.sum(np.diff(points, axis=0)**2, axis=1))
    cumulative = np.insert(np.cumsum(distances), 0, 0)
    total_length = cumulative[-1]

    # Target distances at which to sample
    target_distances = np.linspace(0, total_length, num_points)

    resampled_points = []
    j = 0  # Index in original points
    for d in target_distances:
        # Find segment containing target distance
        while j < len(cumulative) - 1 and cumulative[j+1] < d:
            j += 1

        # Linearly interpolate between points[j] and points[j+1]
        t = (d - cumulative[j]) / (cumulative[j+1] - cumulative[j])
        pt = (1 - t) * points[j] + t * points[j + 1]
        resampled_points.append(tuple(pt))

    return resampled_points


def generate_parabola_points(a, b, c, num_points=100):
    """
    Generates points on a parabola defined by the equation y = ax^2 + bx + c.

    Parameters:
    a (float): The coefficient of the x^2 term in the parabola equation.
    b (float): The coefficient of the x term in the parabola equation.
    c (float): The constant term in the parabola equation.
    num_points (int): The number of points to generate on the parabola.

    Returns:
    list: A list of tuples containing the (x, y) coordinates of the generated points.
    """
    x = np.linspace(-2, 2, num_points)
    y = a * x**2 + b * x + c
    points = list(zip(x, y))
    return points

def main():
    # Generate parabola points
    a, b, c = 1, 0, 0
    num_points = 100
    parabola_points = generate_parabola_points(a, b, c, int(num_points))
    print(f"parabola_points is {parabola_points } ")

    # Resample the points
    old_resample = exact_resample(parabola_points, 10)
    print(f"old_resample is {old_resample } ")
    new_resample = interpolate_points(parabola_points, 10)
    print(f"new_resample is {new_resample } ")

    # Plot the original points and the resampled points side by side
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].plot([x for x, y in parabola_points], [y for x, y in parabola_points], 'bo-')
    axes[0].set_title('Original Points')

    axes[1].plot([x for x, y in old_resample], [y for x, y in old_resample], 'ro-')
    axes[1].set_title('Resampled Points')
    axes[2].plot([x for x, y in new_resample], [y for x, y in new_resample], 'ro-')
    axes[2].set_title('Resampled Points Using interpolation library')

    plt.show()

if __name__ == "__main__":
    main()

