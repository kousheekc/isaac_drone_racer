# Copyright (c) 2025, Kousheek Chakraborty
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# This project uses the IsaacLab framework (https://github.com/isaac-sim/IsaacLab),
# which is licensed under the BSD-3-Clause License.

import numpy as np
import onnxruntime as ort

# Path to your ONNX model
model_path = (
    "/home/kousheek/Dev/saxion/isaac_drone_racer/logs/rsl_rl/isaac_drone_racer/2025-07-23_23-37-55/exported/policy.onnx"
)

# Create ONNX Runtime session (CPU or GPU)
session = ort.InferenceSession(model_path)

# Print input / output details
input_name = session.get_inputs()[0].name
input_shape = session.get_inputs()[0].shape
output_name = session.get_outputs()[0].name
output_shape = session.get_outputs()[0].shape

print(f"Input name: {input_name}, shape: {input_shape}")
print(f"Output name: {output_name}, shape: {output_shape}")

# Dummy input (replace with real state observation later)
dummy_input = np.full([dim if dim is not None else 1 for dim in input_shape], 0.5, dtype=np.float32)

# Run inference
outputs = session.run([output_name], {input_name: dummy_input})
action = outputs[0]

print(f"Input: {dummy_input}")
print(f"Action output: {action}")
