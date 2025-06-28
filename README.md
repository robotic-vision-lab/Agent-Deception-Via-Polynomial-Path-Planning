# Agent Deception Via Polynomial Path Planning
 
## Overview
Deceptive path planning involves an intelligent agent creating a plan that hides
its true intentions while appearing to pursue an alternative goal.
Deception is a crucial tool for misleading and confusing adversaries, especially
in sectors such as security, transportation, and surveillance, where the ability
to conceal true intentions may lead to significant advantages.
 




![alt text 1](docs/overview_figure/overview_deception.png) 
*Figure 1. Illustration of the deceptive polynomial path planner pipeline. Our
approach is designed to address factors of deception including distance from
each goal (top left and right), movement direction (bottom left), and the
apparent goal of the agent, as inferred from its movement trend.*

This repository provides the source code that implements our deceptive
polynomial path planner shown in Fig. 1. The planner generates deceptive
behaviors that guide an agent towards a target destination while
simultaneously
misleading an adversarial observer into predicting a false goal. 

## Table of Contents
- [Overview](#overview)
- [Installation](#installation)
- [Usage](#usage)
- [Citation](#citation)
- [License](#license)
- [References](#references)
- [Acknowledgements](#acknowledgements)

### Authors

- Nolan B. Gutierrez [<img src="./docs/google_scholar_logo/google_scholar_logo.svg" width=14px>](https://scholar.google.com/citations?user=2KSNiPQAAAAJ&hl=en)
- Brian M. Sadler [<img src="./docs/google_scholar_logo/google_scholar_logo.svg" width=14px>](https://scholar.google.com/citations?user=s9eCQn4AAAAJ&hl=en)
- William J. Beksi [<img src="./docs/google_scholar_logo/google_scholar_logo.svg" width=14px>](https://scholar.google.com/citations?user=lU2Z7MMAAAAJ&hl=en)

## Citation

If you find this project useful, please consider citing it using the following entry:

```bibtex
@misc{gutierrez2025agent,
  author       = {Gutierrez, Nolan and Beksi, William J. and Sadler, Brian M.},
  title        = {Agent Deception Via Polynomial Path Planning},
  year         = 2025,
  howpublished = {\url{https://github.com/robotic-vision-lab/Agent-Deception-Via-Polynomial-Path-Planning.git}},
}
```

## Installation

#### Requirements

- Ubuntu 22
- Miniconda3

#### Setting up your environment

First, install [Miniconda3](https://docs.conda.io/en/latest/miniconda.html) for
your platform, then run the following:


```bash
# Create a new conda environment with Python 3.10
conda create -n deception_env python=3.10 -y
conda activate deception_env

# Clone the repository
git clone https://github.com/robotic-vision-lab/Agent-Deception-Via-Polynomial-Path-Planning.git
cd Agent-Deception-Via-Polynomial-Path-Planning

# Install required Python packages
pip install -r requirements.txt
```


## Usage

To run the deceptive polynomial path planning algorithm, follow these instructions:

1. Ensure that you have activated your conda environment and installed the required dependencies from the `requirements.txt` file.

2. Use the following command to execute the `test.py` script with the desired input parameters:

```bash
for i in {1..10}; do python deceptive_polynomials/test.py --degree 5 --beta 0  --alternative_goals "[[9.5,9.5]]" --points "[[9.5,9.5]]" --circle_location "(7,-8)" --start_location [5.5,1.5] --goal [1.5,9.5] --circle_beta 100   --short_on --obs_on  --title "Exg_Align_Coeff_Dist_Smooth_\$i"  --ambiguity_on --alt_angle_beta 1000  --curvature_on  --reg_beta 100000 ; done
```
The \$i ensures that the loop index is correctly interpreted in the shell. If you're not using a shell loop, remove the backslash.
The command above runs the algorithm 10 times with the specified input parameters. You can modify the input parameters as needed to suit your specific use case.

Here's a brief explanation of the input parameters:
```
- --degree: Degree of the polynomial to be fitted
- --beta: Regularization parameter
- --alternative_goals: Alternative goal locations
- --points: Points on the trajectory
- --circle_location: Location of the circle used in the path planning
- --start_location: Start location of the robot/vehicle
- --goal: The true goal location
- --circle_beta: Circle constraint regularization parameter
- --short_on: Enable/disable the shortest path constraint
- --obs_on: Enable/disable the obstacle constraint
- --title: Title for the output files
- --ambiguity_on: Enable/disable the ambiguity constraint
- --alt_angle_beta: Alternative angle regularization parameter
- --curvature_on: Enable/disable the curvature constraint
- --reg_beta: Regularization parameter for the trajectory fitting
- --point_beta: Point constraint regularization parameter
- --traj_folder_prefix: Prefix for the trajectory folder name
```

After each run, the resulting trajectories are stored in the
`trajectories/` folder by default. To visualize all of the trajectories using matplotlib,
execute the following command:
```bash
python deceptive_polynomials/utils/plot_trajectories.py --folder trajectories
``` 

## License


[![license](https://img.shields.io/github/license/mashape/apistatus.svg?maxAge=2592000)](https://github.com/robotic-vision-lab/Agent-Deception-Via-Polynomial-Path-Planning/blob/main/LICENSE)

## References

This project utilizes code from the following project:  

* PythonRobotics https://github.com/AtsushiSakai/PythonRobotics

## Acknowledgements

This research was supported by the Graduate Assistance in Areas of National Need (GAANN) Fellowship, funded by the U.S. Department of Education. We thank the GAANN program for providing financial support that enabled the continued development of this project.

Initial development and early evaluation of the deceptive polynomial path planner were conducted during an internship at the U.S. Army Combat Capabilities Development Command (DEVCOM) Army Research Laboratory, as part of the DoD’s HBCU/MI Summer Research Internship Program. We are grateful for the opportunity and resources provided by this program.

