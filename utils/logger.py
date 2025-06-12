# Copyright (c) 2025, Kousheek Chakraborty
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# This project uses the IsaacLab framework (https://github.com/isaac-sim/IsaacLab),
# which is licensed under the BSD-3-Clause License.

import csv
import os
from datetime import datetime

import torch

from utils.plotter import generate_plots


class CSVLogger:
    def __init__(self, folder_path="."):
        if not os.path.exists(folder_path):
            raise FileNotFoundError(f"The folder '{folder_path}' does not exist.")

        # Generate the file name with date and time stamp
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.file_path = os.path.join(folder_path, f"log_{timestamp}.csv")
        self.keys = []  # Keeps track of column headers
        self.file_initialized = False

    def log(self, data_dict):
        """
        Logs a dictionary of key-value pairs into a CSV file.

        Args:
            data_dict (dict): A dictionary where keys are column names and values are tensors of shape (n,).
        """
        # Verify that all tensors have n = 1
        for key, tensor in data_dict.items():
            if not isinstance(tensor, torch.Tensor):
                raise ValueError(f"Value for key '{key}' must be a tensor.")
            if tensor.ndim != 1 or tensor.shape[0] != 1:
                raise ValueError(f"Tensor for key '{key}' must have shape (1,), but got {tensor.shape}.")

        # Flatten tensors to scalar values (since n = 1, we can extract the single value)
        flattened_data = {key: tensor.item() for key, tensor in data_dict.items()}

        # Initialize the CSV file if not already done
        if not self.file_initialized:
            self.keys = list(flattened_data.keys())
            with open(self.file_path, mode="w", newline="") as file:
                writer = csv.DictWriter(file, fieldnames=self.keys)
                writer.writeheader()
            self.file_initialized = True

        # Check for new keys and update the CSV header if necessary
        new_keys = [key for key in flattened_data.keys() if key not in self.keys]
        if new_keys:
            self.keys.extend(new_keys)
            # Rewrite the CSV file with the updated header
            with open(self.file_path) as file:
                rows = list(csv.DictReader(file))
            with open(self.file_path, mode="w", newline="") as file:
                writer = csv.DictWriter(file, fieldnames=self.keys)
                writer.writeheader()
                writer.writerows(rows)

        # Write the new row
        with open(self.file_path, mode="a", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=self.keys)
            # Fill in missing keys with empty values
            row = {key: flattened_data.get(key, "") for key in self.keys}
            writer.writerow(row)

    def save(self):
        """
        Saves the current file and reinitializes a new file for logging.
        """
        # Ensure the current file is saved (already handled by the log method)
        if not self.file_initialized:
            raise RuntimeError("No file has been initialized yet. Log some data first.")

        generate_plots(self.file_path)
        # Generate a new file name with a timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.file_path = os.path.join(os.path.dirname(self.file_path), f"log_{timestamp}.csv")

        # Reset keys and reinitialize the file
        self.keys = []
        self.file_initialized = False


def log(env, keys, value):
    """
    Log data to env.extras['metrics'].
    """
    if "metrics" not in env.extras:
        env.extras["metrics"] = {}

    if not isinstance(keys, list) or not all(isinstance(key, str) for key in keys):
        raise TypeError("keys must be a list of strings.")

    if len(keys) != value.shape[1]:
        raise ValueError(f"Length of keys ({len(keys)}) must match the second dimension of value ({value.shape[1]}).")

    for i, key in enumerate(keys):
        env.extras["metrics"][key] = value[:, i]
