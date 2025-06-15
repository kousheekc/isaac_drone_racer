# Isaac Drone Racer

[![IsaacSim](https://img.shields.io/badge/IsaacSim-4.5.0-silver.svg)](https://docs.isaacsim.omniverse.nvidia.com/latest/index.html)
[![Python](https://img.shields.io/badge/python-3.10-blue.svg)](https://docs.python.org/3/whatsnew/3.10.html)
[![pre-commit](https://img.shields.io/github/actions/workflow/status/isaac-sim/IsaacLab/pre-commit.yaml?logo=pre-commit&logoColor=white&label=pre-commit&color=brightgreen)](https://github.com/kousheekc/isaac_drone_racer/blob/master/.github/workflows/pre-commit.yaml)
[![License](https://img.shields.io/badge/license-BSD--3-yellow.svg)](https://opensource.org/licenses/BSD-3-Clause)

**Isaac Drone Racer** is an open-source simulation framework for autonomous drone racing, developed on top of [IsaacLab](https://github.com/isaac-sim/IsaacLab). It is designed for training reinforcement learning policies in realistic racing environments, with a focus on accurate physics and modular design.

Autonomous drone racing is an active area of research. This project builds on insights from that body of work, combining them with massively parallel simulation to train racing policies within minutes — offering a fast and flexible platform for experimentation and benchmarking.

## Features

Key highlights of the Isaac Drone Racer project:

1. **Accurate Physics Modeling** — Simulates rotor dynamics, aerodynamic drag, and power consumption to closely match real-world quadrotor behavior.
2. **Low-Level Flight Controller** — Built-in attitude and rate controllers.
3. **Manager-Based Design** — Modular architecture using IsaacLab's [manager based architecture](https://isaac-sim.github.io/IsaacLab/main/source/refs/reference_architecture/index.html#manager-based).
4. **Onboard Sensor Suite** — Includes simulated fisheye camera, IMU and collision detection.
5. **Track Generator** — Dynamically generate custom race tracks.
6. **Logger and Plotter** — Integrated tools for monitoring and visualizing flight behavior.

## Requirements
This framework has been tested on x64 based Linux systems, specifically Ubuntu 22.04. But it should also work on Windows 10/11.

### Prerequisites
- Workstation capable of running Isaac Sim (see [link](https://github.com/isaac-sim/IsaacSim?tab=readme-ov-file#prerequisites-and-environment-setup))
- [Git](https://git-scm.com/downloads) & [Git LFS](https://git-lfs.com)
- [Conda](https://www.anaconda.com/docs/getting-started/miniconda/install) for local installation or [Docker](https://docs.docker.com/engine/install/ubuntu/) with [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)

## Setup
1. Follow the [Isaac Lab pip installation instructions](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/pip_installation.html), with the following modifications:
- After cloning the Isaac Lab repository:
```bash
git clone git@github.com:isaac-sim/IsaacLab.git
```

- Checkout the `v2.1.0` release tag:
```bash
cd IsaacLab
git checkout v2.1.0
```

2. Clone Isaac Drone Racer:
```bash
git clone https://github.com/kousheekc/isaac_drone_racer.git
```

3. Install the modules in editable mode
```bash
cd isaac_drone_racer
pip3 install -e .
```

## Usage


## Next Steps

- [ ] **Data-driven aerodynamic model pipeline** - integrate tools for data driven system identification, calibration and include the learned aerodynamic forces into the simulation environment.
- [ ] **Power consumption model**  - incorporate a detailed power model that accounts for battery disharge based on current draw.
- [ ] **Policy learning using onboard sensors** - explore and implement methods to transition away from full-state observations by instead using only onboard sensor data (e.g camera + IMU).

## License
This project is licensed under the BSD 3-Clause License - see the [LICENSE](https://github.com/kousheekc/isaac_drone_racer/blob/master/LICENSE) file for details.

## Contact
Kousheek Chakraborty - kousheekc@gmail.com

Project Link: [https://github.com/kousheekc/isaac_drone_racer](https://github.com/kousheekc/isaac_drone_racer)

If you encounter any difficulties, feel free to reach out through the Issues section. If you find any bugs or have improvements to suggest, don't hesitate to make a pull request.
