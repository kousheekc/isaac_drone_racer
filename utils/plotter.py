# Copyright (c) 2025, Kousheek Chakraborty
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# This project uses the IsaacLab framework (https://github.com/isaac-sim/IsaacLab),
# which is licensed under the BSD-3-Clause License.

import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scienceplots  # noqa: F401
from scipy.spatial.transform import Rotation as R


def generate_plots(log_directory: str):  # noqa: C901
    """
    Generate plots from the log file.
    Args:
        log_directory (str): Path to the csv log file.
    """

    plot_position = False
    plot_orientation_from_quat = False
    plot_orientation_from_rot_mat = False
    plot_velocity = False
    plot_angular_velocity = False
    plot_rotors_ang_vel = False
    plot_actions = False

    plt.style.use(["science", "ieee", "bright", "no-latex"])
    matplotlib.rcParams.update({"font.size": 6})

    # check if the log directory exists
    if not os.path.exists(log_directory):
        raise FileNotFoundError(f"The log directory {log_directory} does not exist.")

    # check if the log file exists
    if not os.path.isfile(log_directory):
        raise FileNotFoundError(f"The log file {log_directory} does not exist.")

    # check if the log file is a csv file
    if not log_directory.endswith(".csv"):
        raise ValueError(f"The log file {log_directory} is not a csv file.")

    # check if the log file is empty
    if os.path.getsize(log_directory) == 0:
        raise ValueError(f"The log file {log_directory} is empty.")

    # check if the log file is a csv file
    if not log_directory.endswith(".csv"):
        raise ValueError(f"The log file {log_directory} is not a csv file.")

    # extract the name of the log file without the extension
    log_file_name = os.path.splitext(os.path.basename(log_directory))[0]
    # create a new folder in the log directory named log_file_name_plots
    plot_directory = os.path.join(os.path.dirname(log_directory), log_file_name + "_plots")
    if not os.path.exists(plot_directory):
        os.makedirs(plot_directory)
        print(f"Created directory: {plot_directory}")
    # read the log file
    log_data = pd.read_csv(log_directory)

    # check if the time column is present, otherwise create a column with values starting from zero with increments of 0.005
    if "time" not in log_data.columns:
        log_data["time"] = np.arange(0, len(log_data) * 0.005, 0.005)
        print("time column not found, creating a new one")

    # check if px, py, pz, pxd, pyd, pzd columns are present
    if all(col in log_data.columns for col in ["px", "py", "pz"]):
        plot_position = True

    # check if qw, qx, qy, qz columns are present
    if all(col in log_data.columns for col in ["qw", "qx", "qy", "qz"]):
        plot_orientation_from_quat = True

    # check if r11, r12, r13, r21, r22, r23, r31, r32, r33 columns are present
    if all(col in log_data.columns for col in ["r11", "r12", "r13", "r21", "r22", "r23", "r31", "r32", "r33"]):
        plot_orientation_from_rot_mat = True

    # check if vx, vy, vz columns are present
    if all(col in log_data.columns for col in ["vx", "vy", "vz"]):
        plot_velocity = True

    # check if wx, wy, wz columns are present
    if all(col in log_data.columns for col in ["wx", "wy", "wz"]):
        plot_angular_velocity = True

    # check if the t1, t2, t3, t4, t5, t6 columns are present
    if all(col in log_data.columns for col in ["w1", "w2", "w3", "w4"]):
        plot_rotors_ang_vel = True

    if all(col in log_data.columns for col in ["a1", "a2", "a3", "a4"]):
        plot_actions = True

    if plot_position:
        # plot the position
        fig, ax = plt.subplots(3, 1)
        fig.suptitle("Position")
        ax[0].plot(log_data["time"], log_data["px"], label=r"$\mathbf{p}_{x}$")
        ax[0].set_ylabel(r"$\mathbf{p}_x$ [m]")
        # place the legend outside the plot in the center top of the plot
        ax[0].legend(title="", frameon=False, loc="best", ncol=2)
        ax[0].grid()
        ax[0].set_xlim(0, log_data["time"].max())

        ax[1].plot(log_data["time"], log_data["py"], label=r"$\mathbf{p}_{y}$")
        ax[1].set_ylabel(r"$\mathbf{p}_y$ [m]")
        # ax[1].legend(title='',frameon=True, loc='upper left', ncol=2)
        ax[1].grid()
        ax[1].set_xlim(0, log_data["time"].max())

        ax[2].plot(log_data["time"], log_data["pz"], label=r"$\mathbf{p}_{z}$")
        ax[2].set_ylabel(r"$\mathbf{p}_z$ [m]")
        # ax[2].legend(title='',frameon=True, loc='upper left', ncol=2)
        ax[2].grid()
        ax[2].set_xlim(0, log_data["time"].max())

        plt.xlabel("Time [s]")
        plt.tight_layout()
        plt.savefig(os.path.join(plot_directory, "position.pdf"))
        # plt.show()

    if plot_orientation_from_quat:
        # convert quaternion to euler angles using the function from scipy
        # create a new column for the euler angles
        log_data["roll"] = 0
        log_data["pitch"] = 0
        log_data["yaw"] = 0
        # convert the quaternion to euler angles
        for i in range(len(log_data)):
            q = [log_data["qw"][i], log_data["qx"][i], log_data["qy"][i], log_data["qz"][i]]
            r = R.from_quat(q, scalar_first=True)
            euler = r.as_euler("XYZ", degrees=True)
            log_data["roll"][i] = euler[0]
            log_data["pitch"][i] = euler[1]
            log_data["yaw"][i] = euler[2]
        # plot the orientation
        fig, ax = plt.subplots(3, 1)
        fig.suptitle("Orientation")
        ax[0].plot(log_data["time"], log_data["roll"], label=r"$\phi$")
        ax[0].set_ylabel(r"$\phi$ [deg]")
        ax[0].grid()
        ax[0].set_xlim(0, log_data["time"].max())

        ax[1].plot(log_data["time"], log_data["pitch"], label=r"$\theta$")
        ax[1].set_ylabel(r"$\theta$ [deg]")
        ax[1].grid()
        ax[1].set_xlim(0, log_data["time"].max())

        ax[2].plot(log_data["time"], log_data["yaw"], label=r"$\psi$")
        ax[2].set_ylabel(r"$\psi$ [deg]")
        ax[2].grid()
        ax[2].set_xlim(0, log_data["time"].max())
        plt.xlabel("Time [s]")
        plt.tight_layout()
        plt.savefig(os.path.join(plot_directory, "orientation_from_quat.pdf"))
        # plt.show()

    if plot_orientation_from_rot_mat:
        # convert rotation matrix to euler angles using the function from scipy
        # create a new column for the euler angles
        log_data["roll"] = 0
        log_data["pitch"] = 0
        log_data["yaw"] = 0
        # convert the rotation matrix to euler angles
        for i in range(len(log_data)):
            r = np.array([
                [log_data["r11"][i], log_data["r12"][i], log_data["r13"][i]],
                [log_data["r21"][i], log_data["r22"][i], log_data["r23"][i]],
                [log_data["r31"][i], log_data["r32"][i], log_data["r33"][i]],
            ])
            euler = R.from_matrix(r).as_euler("XYZ", degrees=True)
            log_data["roll"][i] = euler[0]
            log_data["pitch"][i] = euler[1]
            log_data["yaw"][i] = euler[2]
        # plot the orientation
        fig, ax = plt.subplots(3, 1)
        fig.suptitle("Orientation")
        ax[0].plot(log_data["time"], log_data["roll"], label=r"$\phi$")
        ax[0].set_ylabel(r"$\phi$ [deg]")
        ax[0].grid()
        ax[0].set_xlim(0, log_data["time"].max())

        ax[1].plot(log_data["time"], log_data["pitch"], label=r"$\theta$")
        ax[1].set_ylabel(r"$\theta$ [deg]")
        ax[1].grid()
        ax[1].set_xlim(0, log_data["time"].max())

        ax[2].plot(log_data["time"], log_data["yaw"], label=r"$\psi$")
        ax[2].set_ylabel(r"$\psi$ [deg]")
        ax[2].grid()
        ax[2].set_xlim(0, log_data["time"].max())
        plt.xlabel("Time [s]")
        plt.tight_layout()
        plt.savefig(os.path.join(plot_directory, "orientation_from_rotmat.pdf"))
        # plt.show()

    if plot_velocity:
        # plot the velocity
        fig, ax = plt.subplots(3, 1)
        fig.suptitle("Velocity")
        ax[0].plot(log_data["time"], log_data["vx"], label=r"$\mathbf{v}_x$")
        ax[0].set_ylabel(r"$\mathbf{v}_x$ [m/s]")
        ax[0].grid()
        ax[0].set_xlim(0, log_data["time"].max())

        ax[1].plot(log_data["time"], log_data["vy"], label=r"$\mathbf{v}_y$")
        ax[1].set_ylabel(r"$\mathbf{v}_y$ [m/s]")
        ax[1].grid()
        ax[1].set_xlim(0, log_data["time"].max())

        ax[2].plot(log_data["time"], log_data["vz"], label=r"$\mathbf{v}_z$")
        ax[2].set_ylabel(r"$\mathbf{v}_z$ [m/s]")
        ax[2].grid()
        ax[2].set_xlim(0, log_data["time"].max())
        plt.xlabel("Time [s]")
        plt.tight_layout()
        plt.savefig(os.path.join(plot_directory, "velocity.pdf"))
        # plt.show()

    if plot_angular_velocity:
        # plot the angular velocity
        fig, ax = plt.subplots(3, 1)
        fig.suptitle("Angular Velocity")
        ax[0].plot(log_data["time"], log_data["wx"], label=r"$\boldsymbol{\omega}_x$")
        ax[0].set_ylabel(r"$\boldsymbol{\omega}_x$ [rad/s]")
        ax[0].grid()
        ax[0].set_xlim(0, log_data["time"].max())

        ax[1].plot(log_data["time"], log_data["wy"], label=r"$\boldsymbol{\omega}_y$")
        ax[1].set_ylabel(r"$\boldsymbol{\omega}_y$ [rad/s]")
        ax[1].grid()
        ax[1].set_xlim(0, log_data["time"].max())

        ax[2].plot(log_data["time"], log_data["wz"], label=r"$\boldsymbol{\omega}_z$")
        ax[2].set_ylabel(r"$\boldsymbol{\omega}_z$ [rad/s]")
        ax[2].grid()
        ax[2].set_xlim(0, log_data["time"].max())
        plt.xlabel("Time [s]")
        plt.tight_layout()
        plt.savefig(os.path.join(plot_directory, "angular_velocity.pdf"))
        # plt.show()

    if plot_rotors_ang_vel:
        # plot the force tracking
        fig, ax = plt.subplots()
        fig.suptitle("Rotors Angular Velocities")
        ax.plot(log_data["time"], log_data["w1"], label=r"$\boldsymbol{w}_{1}$")
        ax.plot(log_data["time"], log_data["w2"], label=r"$\boldsymbol{w}_{2}$")
        ax.plot(log_data["time"], log_data["w3"], label=r"$\boldsymbol{w}_{3}$")
        ax.plot(log_data["time"], log_data["w4"], label=r"$\boldsymbol{w}_{4}$")
        ax.set_ylabel(r"$\boldsymbol{w}$ [rad/s]")
        ax.legend(title="", frameon=False, loc="best", ncol=3)
        ax.grid()
        ax.set_xlim(0, log_data["time"].max())

        plt.xlabel("Time [s]")
        plt.tight_layout()
        plt.savefig(os.path.join(plot_directory, "rotors_ang_vel.pdf"))
        # plt.show()

    if plot_actions:
        # plot the actions
        fig, ax = plt.subplots()
        fig.suptitle("Actions")
        ax.plot(log_data["time"], log_data["a1"], label=r"$\mathbf{a}_{1}$")
        ax.plot(log_data["time"], log_data["a2"], label=r"$\mathbf{a}_{2}$")
        ax.plot(log_data["time"], log_data["a3"], label=r"$\mathbf{a}_{3}$")
        ax.plot(log_data["time"], log_data["a4"], label=r"$\mathbf{a}_{4}$")
        ax.set_ylabel(r"$\mathbf{a}$ (normalised)")
        ax.legend(title="", frameon=False, loc="best", ncol=3)
        ax.grid()
        ax.set_xlim(0, log_data["time"].max())

        plt.xlabel("Time [s]")
        plt.tight_layout()
        plt.savefig(os.path.join(plot_directory, "actions.pdf"))
        # plt.show()
