# Copyright (c) 2025, Kousheek Chakraborty
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# This project uses the IsaacLab framework (https://github.com/isaac-sim/IsaacLab),
# which is licensed under the BSD-3-Clause License.

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit


def exponential_response(t, tau, initial, setpoint):
    """General exponential response model: y(t) = setpoint + (initial - setpoint) * exp(-t/tau)"""
    return setpoint + (initial - setpoint) * np.exp(-t / tau)


def find_settling_segments(df, data_col="ang_vel_x", min_segment_length=100, change_threshold=None):
    """Find segments where the system is settling from some initial value to a setpoint"""
    data_values = df[data_col].values

    if change_threshold is None:
        # Auto-detect threshold as 2 times the standard deviation of the data
        change_threshold = 2 * np.std(data_values)

    # Find potential start points where |data - final_value| > threshold
    final_value = np.mean(data_values[-min(50, len(data_values) // 4) :])  # Estimate final value
    high_change_indices = np.where(np.abs(data_values - final_value) > change_threshold)[0]

    if len(high_change_indices) == 0:
        return []

    # Group consecutive indices into segments
    segments = []
    current_segment_start = high_change_indices[0]

    for i in range(1, len(high_change_indices)):
        # If there's a gap larger than 10 samples, start a new segment
        if high_change_indices[i] - high_change_indices[i - 1] > 10:
            # End current segment
            if high_change_indices[i - 1] - current_segment_start >= min_segment_length:
                segments.append((current_segment_start, high_change_indices[i - 1]))
            current_segment_start = high_change_indices[i]

    # Add the last segment
    if high_change_indices[-1] - current_segment_start >= min_segment_length:
        segments.append((current_segment_start, high_change_indices[-1]))

    # If no segments found, analyze the entire data if it shows significant change
    if len(segments) == 0 and np.abs(data_values[0] - final_value) > change_threshold:
        segments.append((0, len(data_values) - 1))

    return segments


def analyze_settling_response(df, start_idx=0, end_idx=None, data_col="ang_vel_x", time_col="timestamp", plot=False):
    """Analyze settling response where system transitions from initial value to setpoint

    Uses actual timestamp column for time measurements
    """

    if end_idx is None:
        end_idx = len(df) - 1

    if end_idx - start_idx < 50:
        return None

    # Use actual timestamp values
    time_data = df[time_col].iloc[start_idx:end_idx].values
    data_values = df[data_col].iloc[start_idx:end_idx].values

    # Normalize time to start from 0
    time_normalized = time_data - time_data[0]  # Start from 0 seconds

    # Estimate initial and setpoint values
    initial_value = data_values[0]

    # Estimate setpoint as the mean of last 20% of data
    setpoint_start = int(0.8 * len(data_values))
    if setpoint_start < len(data_values) - 5:
        setpoint_value = np.mean(data_values[setpoint_start:])
    else:
        setpoint_value = data_values[-1]

    # Skip if the response magnitude is too small to analyze
    response_magnitude = abs(setpoint_value - initial_value)
    if response_magnitude < 0.01:  # Adjust threshold as needed
        return None

    try:
        # Fit exponential response
        # Initial guess for tau (time constant in seconds)
        tau_guess = time_normalized[-1] / 3  # Rough estimate in seconds

        popt, pcov = curve_fit(
            exponential_response,
            time_normalized,
            data_values,
            p0=[tau_guess, initial_value, setpoint_value],  # [tau, initial, setpoint]
            maxfev=2000,
        )

        tau_fitted = abs(popt[0])  # Time constant in seconds
        fitted_initial = popt[1]
        fitted_setpoint = popt[2]

        # Calculate R-squared for goodness of fit
        data_fitted = exponential_response(time_normalized, tau_fitted, fitted_initial, fitted_setpoint)
        ss_res = np.sum((data_values - data_fitted) ** 2)
        ss_tot = np.sum((data_values - np.mean(data_values)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0

        # Calculate response characteristics (all in seconds)
        # 63.2% response time (time constant)
        time_63 = tau_fitted

        # 95% settling time (approximately 3 * tau)
        settling_time_95 = 3 * tau_fitted

        # 98% settling time (approximately 4 * tau)
        settling_time_98 = 4 * tau_fitted

        # Time to reach specific percentages of final change
        # For exponential approach: y = setpoint + (initial - setpoint) * exp(-t/tau)
        # So: (y - setpoint) / (initial - setpoint) = exp(-t/tau)
        # Therefore: t = -tau * ln((y - setpoint) / (initial - setpoint))

        change_magnitude = fitted_setpoint - fitted_initial
        if abs(change_magnitude) > 1e-10:  # Avoid division by zero
            # 50% of the way to setpoint
            time_50_percent = tau_fitted * np.log(2)
            # 90% of the way to setpoint
            time_90_percent = tau_fitted * np.log(10)
        else:
            time_50_percent = float("inf")
            time_90_percent = float("inf")

        results = {
            "start_index": start_idx,
            "end_index": end_idx,
            "time_constant": tau_fitted,  # in seconds
            "settling_time_95": settling_time_95,  # in seconds
            "settling_time_98": settling_time_98,  # in seconds
            "time_to_50_percent": time_50_percent,  # in seconds
            "time_to_90_percent": time_90_percent,  # in seconds
            "initial_value": fitted_initial,
            "setpoint_value": fitted_setpoint,
            "response_magnitude": abs(change_magnitude),
            "response_direction": "increase" if change_magnitude > 0 else "decrease",
            "r_squared": r_squared,
            "time_data": time_normalized,
            "data_values": data_values,
            "data_fitted": data_fitted,
        }

        if plot:
            plt.figure(figsize=(12, 8))

            # Main plot
            plt.subplot(2, 1, 1)
            plt.plot(time_normalized, data_values, "b-", label="Measured", linewidth=2)
            plt.plot(time_normalized, data_fitted, "r--", label=f"Fitted (τ={tau_fitted:.4f}s)", linewidth=2)

            # Mark important time points
            plt.axvline(x=time_63, color="g", linestyle=":", alpha=0.7, label=f"63.2% response: {time_63:.4f}s")
            plt.axvline(
                x=settling_time_95,
                color="orange",
                linestyle=":",
                alpha=0.7,
                label=f"95% settled: {settling_time_95:.4f}s",
            )

            # Mark important levels
            if abs(change_magnitude) > 1e-10:
                # 63.2% level
                level_632 = fitted_initial + 0.632 * change_magnitude
                plt.axhline(y=level_632, color="g", linestyle="--", alpha=0.5, label=f"63.2% level: {level_632:.3f}")
                # 95% level
                level_95 = fitted_initial + 0.95 * change_magnitude
                plt.axhline(y=level_95, color="orange", linestyle="--", alpha=0.5, label=f"95% level: {level_95:.3f}")

            # Mark initial and setpoint
            plt.axhline(y=fitted_initial, color="b", linestyle=":", alpha=0.7, label=f"Initial: {fitted_initial:.3f}")
            plt.axhline(
                y=fitted_setpoint, color="r", linestyle=":", alpha=0.7, label=f"Setpoint: {fitted_setpoint:.3f}"
            )

            plt.xlabel("Time (seconds)")
            plt.ylabel(f"{data_col}")
            plt.title(f"Settling Response Analysis (R²={r_squared:.3f})")
            plt.legend()
            plt.grid(True, alpha=0.3)

            # Residuals plot
            plt.subplot(2, 1, 2)
            residuals = data_values - data_fitted
            plt.plot(time_normalized, residuals, "r-", linewidth=1)
            plt.axhline(y=0, color="k", linestyle="-", alpha=0.3)
            plt.xlabel("Time (seconds)")
            plt.ylabel(f"Residuals ({data_col})")
            plt.title("Fit Residuals")
            plt.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.show()

        return results

    except Exception as e:
        print(f"Failed to fit settling response from index {start_idx} to {end_idx}: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description="Analyze settling response from log file (using timestamp column)")
    parser.add_argument("log_file", help="Path to the CSV log file")
    parser.add_argument("--column", default="ang_vel_x", help="Column name to analyze (default: ang_vel_x)")
    parser.add_argument("--time-col", default="timestamp", help="Time column name (default: timestamp)")
    parser.add_argument("--plot", action="store_true", help="Show plots for settling response")
    parser.add_argument("--save-plots", action="store_true", help="Save plots to files")
    parser.add_argument("--start-idx", type=int, default=0, help="Start index for analysis (default: 0)")
    parser.add_argument("--end-idx", type=int, help="End index for analysis (default: end of data)")
    parser.add_argument("--auto-segment", action="store_true", help="Automatically find settling segments")
    parser.add_argument("--threshold", type=float, help="Threshold for detecting significant changes")

    args = parser.parse_args()

    # Check if file exists
    if not os.path.exists(args.log_file):
        print(f"Error: Log file '{args.log_file}' not found!")
        return

    # Load data
    print(f"Loading data from {args.log_file}...")
    try:
        df = pd.read_csv(args.log_file)
        print(f"Loaded {len(df)} data points")
        print(f"Columns: {list(df.columns)}")
        print(f"Using timestamp column: {args.time_col}")
        print(f"Analyzing column: {args.column}")
    except Exception as e:
        print(f"Error loading file: {e}")
        return

    # Check required columns
    missing_cols = []
    if args.column not in df.columns:
        missing_cols.append(args.column)
    if args.time_col not in df.columns:
        missing_cols.append(args.time_col)

    if missing_cols:
        print(f"Error: Columns not found in data: {missing_cols}")
        print(f"Available columns: {list(df.columns)}")
        return

    # Show initial data info
    initial_val = df[args.column].iloc[0]
    final_val = df[args.column].iloc[-1]
    max_val = df[args.column].max()
    min_val = df[args.column].min()

    initial_time = df[args.time_col].iloc[0]
    final_time = df[args.time_col].iloc[-1]
    total_time = final_time - initial_time

    print(f"\nData Summary for '{args.column}':")
    print(f"  Initial value: {initial_val:.4f}")
    print(f"  Final value: {final_val:.4f}")
    print(f"  Max value: {max_val:.4f}")
    print(f"  Min value: {min_val:.4f}")
    print(f"  Time span: {total_time:.4f} seconds ({len(df)} data points)")

    all_results = []

    if args.auto_segment:
        # Automatically find settling segments
        print("\nDetecting settling segments...")
        segments = find_settling_segments(df, data_col=args.column, change_threshold=args.threshold)
        print(f"Found {len(segments)} settling segments")

        if len(segments) == 0:
            print("No settling segments detected. Analyzing entire dataset.")
            segments = [(0, len(df) - 1)]

        for i, (start_idx, end_idx) in enumerate(segments):
            print(f"\nAnalyzing segment {i+1}/{len(segments)} (indices {start_idx} to {end_idx})...")

            results = analyze_settling_response(
                df, start_idx, end_idx, data_col=args.column, time_col=args.time_col, plot=args.plot
            )

            if results is not None:
                all_results.append(results)
                print(f"  Time constant: {results['time_constant']:.4f} seconds")
                print(f"  Settling time (95%): {results['settling_time_95']:.4f} seconds")
                print(f"  Settling time (98%): {results['settling_time_98']:.4f} seconds")
                print(f"  Initial value: {results['initial_value']:.4f}")
                print(f"  Setpoint value: {results['setpoint_value']:.4f}")
                print(f"  Response direction: {results['response_direction']}")
                print(f"  Response magnitude: {results['response_magnitude']:.4f}")
                print(f"  R²: {results['r_squared']:.3f}")

                if args.save_plots:
                    plt.figure(figsize=(12, 8))

                    # Main plot
                    plt.subplot(2, 1, 1)
                    plt.plot(results["time_data"], results["data_values"], "b-", label="Measured", linewidth=2)
                    plt.plot(
                        results["time_data"],
                        results["data_fitted"],
                        "r--",
                        label=f'Fitted (τ={results["time_constant"]:.4f}s)',
                        linewidth=2,
                    )
                    plt.axvline(
                        x=results["time_constant"],
                        color="g",
                        linestyle=":",
                        alpha=0.7,
                        label=f'63.2% response: {results["time_constant"]:.4f}s',
                    )
                    plt.axvline(
                        x=results["settling_time_95"],
                        color="orange",
                        linestyle=":",
                        alpha=0.7,
                        label=f'95% settled: {results["settling_time_95"]:.4f}s',
                    )
                    plt.xlabel("Time (seconds)")
                    plt.ylabel(f"{args.column}")
                    plt.title(f'Settling Response {i+1} (R²={results["r_squared"]:.3f})')
                    plt.legend()
                    plt.grid(True, alpha=0.3)

                    # Residuals plot
                    plt.subplot(2, 1, 2)
                    residuals = results["data_values"] - results["data_fitted"]
                    plt.plot(results["time_data"], residuals, "r-", linewidth=1)
                    plt.axhline(y=0, color="k", linestyle="-", alpha=0.3)
                    plt.xlabel("Time (seconds)")
                    plt.ylabel(f"Residuals ({args.column})")
                    plt.title("Fit Residuals")
                    plt.grid(True, alpha=0.3)

                    plt.tight_layout()
                    plt.savefig(f"settling_response_{i+1}.png", dpi=300, bbox_inches="tight")
                    plt.close()
            else:
                print(f"  Failed to analyze segment {i+1}")

    else:
        # Analyze single segment
        print(f"\nAnalyzing settling response from index {args.start_idx} to {args.end_idx or 'end'}...")

        results = analyze_settling_response(
            df, args.start_idx, args.end_idx, data_col=args.column, time_col=args.time_col, plot=args.plot
        )

        if results is not None:
            all_results.append(results)
            print(f"  Time constant: {results['time_constant']:.4f} seconds")
            print(f"  Settling time (95%): {results['settling_time_95']:.4f} seconds")
            print(f"  Settling time (98%): {results['settling_time_98']:.4f} seconds")
            print(f"  Time to 50%: {results['time_to_50_percent']:.4f} seconds")
            print(f"  Time to 90%: {results['time_to_90_percent']:.4f} seconds")
            print(f"  Initial value: {results['initial_value']:.4f}")
            print(f"  Setpoint value: {results['setpoint_value']:.4f}")
            print(f"  Response direction: {results['response_direction']}")
            print(f"  Response magnitude: {results['response_magnitude']:.4f}")
            print(f"  R²: {results['r_squared']:.3f}")

            if args.save_plots:
                plt.figure(figsize=(12, 8))

                # Main plot
                plt.subplot(2, 1, 1)
                plt.plot(results["time_data"], results["data_values"], "b-", label="Measured", linewidth=2)
                plt.plot(
                    results["time_data"],
                    results["data_fitted"],
                    "r--",
                    label=f'Fitted (τ={results["time_constant"]:.4f}s)',
                    linewidth=2,
                )
                plt.xlabel("Time (seconds)")
                plt.ylabel(f"{args.column}")
                plt.title(f'Settling Response Analysis (R²={results["r_squared"]:.3f})')
                plt.legend()
                plt.grid(True, alpha=0.3)

                # Residuals plot
                plt.subplot(2, 1, 2)
                residuals = results["data_values"] - results["data_fitted"]
                plt.plot(results["time_data"], residuals, "r-", linewidth=1)
                plt.axhline(y=0, color="k", linestyle="-", alpha=0.3)
                plt.xlabel("Time (seconds)")
                plt.ylabel(f"Residuals ({args.column})")
                plt.title("Fit Residuals")
                plt.grid(True, alpha=0.3)

                plt.tight_layout()
                plt.savefig("settling_response_analysis.png", dpi=300, bbox_inches="tight")
                plt.close()
        else:
            print("  Failed to analyze settling response")

    # Summary statistics
    if all_results:
        time_constants = [r["time_constant"] for r in all_results]
        settling_times_95 = [r["settling_time_95"] for r in all_results]
        settling_times_98 = [r["settling_time_98"] for r in all_results]
        r_squared_values = [r["r_squared"] for r in all_results]

        print(f"\n{'='*60}")
        print("SETTLING RESPONSE SUMMARY STATISTICS")
        print(f"{'='*60}")
        print(f"Number of analyzed segments: {len(all_results)}")
        print("\nTime Constant (τ) [in seconds]:")
        print(f"  Mean: {np.mean(time_constants):.6f} s")
        print(f"  Std:  {np.std(time_constants):.6f} s")
        print(f"  Min:  {np.min(time_constants):.6f} s")
        print(f"  Max:  {np.max(time_constants):.6f} s")

        print("\nSettling Time (95%) [in seconds]:")
        print(f"  Mean: {np.mean(settling_times_95):.6f} s")
        print(f"  Std:  {np.std(settling_times_95):.6f} s")

        print("\nSettling Time (98%) [in seconds]:")
        print(f"  Mean: {np.mean(settling_times_98):.6f} s")
        print(f"  Std:  {np.std(settling_times_98):.6f} s")

        print("\nFit Quality (R²):")
        print(f"  Mean: {np.mean(r_squared_values):.3f}")
        print(f"  Min:  {np.min(r_squared_values):.3f}")

        # Compare with model parameter
        print("\nComparison with System Model:")
        print(f"  Measured τ (mean): {np.mean(time_constants):.6f} seconds")
        print(f"  Column analyzed: {args.column}")
        print(f"  Time column used: {args.time_col}")

        # Save summary to file
        summary_file = args.log_file.replace(".csv", "_settling_analysis.csv")
        summary_df = pd.DataFrame(all_results)
        # Remove non-serializable columns
        summary_df = summary_df.drop(["time_data", "data_values", "data_fitted"], axis=1, errors="ignore")
        summary_df.to_csv(summary_file, index=False)
        print(f"\nDetailed results saved to: {summary_file}")

    else:
        print("\nNo valid settling responses found for analysis.")


if __name__ == "__main__":
    main()
