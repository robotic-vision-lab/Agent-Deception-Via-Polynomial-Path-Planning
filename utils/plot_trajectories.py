import argparse

import re
import matplotlib.pyplot as plt
import numpy as np
import os
from obstacle_utils import convert_obstacle_to_vertices

def load_trajectories_unsorted(folder, separator='_'):
    trajectories = []
    names = []

    for file in os.listdir(folder):
        if file.endswith('.npy'):
            trajectory = np.load(os.path.join(folder, file))
            trajectories.append(trajectory)
            name = ' '.join(file.split(separator)).replace('.npy', '')
            names.append(name)

    return trajectories, names


def load_trajectories_sorted_letters_included(folder, separator='_'):
    trajectories = []
    names = []

    for file in os.listdir(folder):
        if file.endswith('.npy'):
            trajectory = np.load(os.path.join(folder, file))
            name = ' '.join(file.split(separator)).replace('.npy', '')
            trajectories.append((trajectory, name))

    # Sort trajectories based on names
    trajectories.sort(key=lambda x: x[1])

    # Separate the sorted trajectories and names into separate lists
    sorted_trajectories = [t[0] for t in trajectories]
    sorted_names = [t[1] for t in trajectories]

    return sorted_trajectories, sorted_names



def load_trajectories(folder, separator='_', extract_numbers=True):
    trajectories = []
    names = []

    for file in os.listdir(folder):

        if file.startswith('altpath'):
            trajectory = np.load(os.path.join(folder, file))
            name = 'Alternative Path Function'
            trajectories.append((trajectory, name, 0))
        elif file.endswith('.npy'):
            trajectory = np.load(os.path.join(folder, file))
            name = ' '.join(file.split(separator)).replace('.npy', '')
            
            # Extract the first number from the name if extract_numbers is True
            if extract_numbers:
                sort_key = int(re.search(r'\d+', name).group())
            else:
                sort_key = name
                
            trajectories.append((trajectory, name, sort_key))

    # Sort trajectories based on sort_key
    trajectories.sort(key=lambda x: x[2])

    # Separate the sorted trajectories and names into separate lists
    sorted_trajectories = [t[0] for t in trajectories]
    sorted_names = [t[1] for t in trajectories]

    return sorted_trajectories, sorted_names



def plot_trajectories(folder, obstacles, goals, separator='_'):
    trajectories, names = load_trajectories(folder, separator)

    # Set up the plot style
    plt.style.use('seaborn-darkgrid')
    plt.figure(figsize=(16, 8))

    # Plot trajectories
    for trajectory, name in zip(trajectories, names):
        plt.plot(trajectory[:, 0], trajectory[:, 1], label=name, linewidth=2)

    # Plot obstacles
    for obstacle in obstacles:
        plt.plot(*obstacle.T, 'k-', linewidth=1.5)
        plt.fill(*obstacle.T, color='#2c3e50', alpha=0.7)

    # Plot goals
    goal_labels = ['Actual Goal', 'Decoy Goal']
    goal_colors = ['green', 'red']
    for goal, label, color in zip(goals, goal_labels, goal_colors):
        plt.plot(goal[0], goal[1], marker='*', markersize=12, label=label, color=color)

    # Add grid, labels, and legend
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xlabel('X', fontsize=14, fontweight='bold')
    plt.ylabel('Y', fontsize=14, fontweight='bold')

    # Set legend properties
    plt.legend(loc='lower left', prop={'size': 12, 'weight': 'bold'})

    plt.savefig(folder + '.png', dpi=800, bbox_inches="tight", pad_inches=0)
    # Show the plot
    plt.show()



def main():
    parser = argparse.ArgumentParser(description="Plot trajectories from a specified folder")
    parser.add_argument("--folder", default="trajectories", help="Name of the folder to load trajectories from")
    parser.add_argument("--goals", default="[[1.5, 9.5], [9.5, 9.5]]", help="List of goal coordinates as strings")
    args = parser.parse_args()

    # Example usage
    obstacle_1 = (3, 2, np.array([5.5,7])) # (length, high, center_coordinates)
#    obstacles = [obstacle_1]
    obstacles = []
    obstacle_list = [convert_obstacle_to_vertices(obstacle) for obstacle in obstacles]

    # Parse goals from the command line
    goals = eval(args.goals)

    plot_trajectories(args.folder, obstacle_list, goals)

if __name__ == "__main__":
    main()

