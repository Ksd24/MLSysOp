import os
import re
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# --- Configuration ---
LOG_DIRECTORY = 'logs'
LOG_FILE_PATTERN = re.compile(r"baseline_(\d+)_actor.*")
SMOOTHING_WINDOW = 30

# --- Regex Patterns for Parsing Logs ---
TIMESTAMP_FORMAT = "%Y-%m-%d %H:%M:%S,%f"
PATTERNS = {
    # This pattern now only finds the start of the reward message and captures the timestamp
    'episode_reward_start': re.compile(
        r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}) - Actor-\d+ - INFO - .* finished episode"
    ),
    # This new pattern extracts the reward value from the *next* line
    'reward_value': re.compile(r"reward=([\d.-]+)"),
    'learner_step': re.compile(
        r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}) - Learner - INFO - .* Current Loss: ([\d.]+).* Learner Steps: (\d+)"
    ),
    'total_env_steps': re.compile(
        r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}) - Actor-\d+ - INFO - .* stopping after (\d+) environment steps"
    )
}

def parse_log_file(filepath):
    """Parses a single log file, handling UTF-16 encoding and multi-line rewards."""
    data = {'rewards': [], 'losses': [], 'total_env_steps': 0, 'timestamps': []}
    
    try:
        # MODIFICATION: Read with 'utf-16' encoding and handle potential errors
        with open(filepath, 'r', encoding='utf-16', errors='ignore') as f:
            lines = f.readlines()
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return None, 0

    for i, line in enumerate(lines):
        # --- Handle Multi-line Reward Messages ---
        match_reward_start = PATTERNS['episode_reward_start'].search(line)
        if match_reward_start:
            ts_str = match_reward_start.groups()[0]
            # Check if there is a next line to read
            if (i + 1) < len(lines):
                next_line = lines[i+1]
                match_reward_value = PATTERNS['reward_value'].search(next_line)
                if match_reward_value:
                    try:
                        ts = datetime.strptime(ts_str, TIMESTAMP_FORMAT)
                        reward = float(match_reward_value.groups()[0])
                        data['rewards'].append((ts, reward))
                        data['timestamps'].append(ts)
                    except (ValueError, IndexError):
                        continue # Skip if timestamp or reward parsing fails
            continue # Move to the next line regardless

        # --- Handle Single-line Learner and Step Messages ---
        match_learner = PATTERNS['learner_step'].search(line)
        if match_learner:
            ts_str, loss, step = match_learner.groups()
            try:
                ts = datetime.strptime(ts_str, TIMESTAMP_FORMAT)
                data['losses'].append((ts, float(loss), int(step)))
                data['timestamps'].append(ts)
            except (ValueError, IndexError):
                continue
            continue
            
    if not data['timestamps']:
        return None, 0

    total_duration_seconds = (max(data['timestamps']) - min(data['timestamps'])).total_seconds()
    return data, max(1, total_duration_seconds)

# ... (The rest of the functions analyze_results, plot_results, etc., remain the same) ...
def analyze_results(all_data):
    """Calculates summary metrics from parsed data."""
    summary = {}
    for num_actors, (data, duration) in sorted(all_data.items()):
        if not data: continue
        total_learner_steps = data['losses'][-1][2] if data['losses'] else 0
        summary[num_actors] = {
            'Wall-Clock Time (s)': round(duration),
            'Total Learner Steps': total_learner_steps,
            'Learner Steps/s': round(total_learner_steps / duration, 2),
            'Final Avg Reward': np.mean([r[1] for r in data['rewards'][-10:]]) if data['rewards'] else 0
        }
    
    baseline_throughput = summary.get(1, {}).get('Learner Steps/s')
    if baseline_throughput and baseline_throughput > 0:
        for metrics in summary.values():
            # Calculate speedup based on learner steps per second
            speedup = round(metrics['Learner Steps/s'] / baseline_throughput, 2)
            metrics['Speedup'] = speedup
    return summary


def plot_results(all_data, summary):
    """Generates and saves plots."""
    plt.style.use('seaborn-v0_8-whitegrid')
    # Reward vs. Time
    fig, ax = plt.subplots(figsize=(12, 7))
    for num_actors, (data, _) in sorted(all_data.items()):
        if data and data['rewards']:
            df = pd.DataFrame(data['rewards'], columns=['timestamp', 'reward']).sort_values('timestamp')
            df['smoothed_reward'] = df['reward'].rolling(window=SMOOTHING_WINDOW, min_periods=1).mean()
            start_time = df['timestamp'].min()
            df['time_sec'] = (df['timestamp'] - start_time).dt.total_seconds()
            ax.plot(df['time_sec'], df['smoothed_reward'], label=f'{num_actors} Actor(s)')
    ax.set_title('Learning Convergence: Smoothed Episode Reward vs. Time')
    ax.set_xlabel('Wall-Clock Time (seconds)')
    ax.set_ylabel(f'Average Reward (Smoothed, Window={SMOOTHING_WINDOW})')
    ax.legend()
    ax.grid(True)
    plt.savefig('plot_reward_vs_time.png')
    plt.close()
    print("Generated plot: plot_reward_vs_time.png")

    # Throughput
    actor_counts = sorted(summary.keys())
    if not actor_counts: return
    learner_throughput = [summary[n]['Learner Steps/s'] for n in actor_counts]
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(actor_counts, learner_throughput, marker='s', linestyle='-', label='Learner Steps/s (Training Throughput)')
    ax.set_title('System Throughput vs. Number of Actors')
    ax.set_xlabel('Number of Actors'); ax.set_ylabel('Steps per Second'); ax.set_xticks(actor_counts); ax.legend(); ax.grid(True)
    plt.savefig('plot_throughput.png')
    plt.close()
    print("Generated plot: plot_throughput.png")

    # Speedup
    if 1 in summary:
        speedup = [summary[n].get('Speedup', 0) for n in actor_counts]
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(actor_counts, speedup, marker='o', linestyle='--', label='Actual Speedup')
        ax.plot(actor_counts, actor_counts, linestyle=':', color='gray', label='Ideal Linear Speedup')
        ax.set_title('Training Speedup vs. Number of Actors')
        ax.set_xlabel('Number of Actors'); ax.set_ylabel('Speedup (vs. 1 Actor)'); ax.set_xticks(actor_counts); ax.legend(); ax.grid(True)
        plt.savefig('plot_speedup.png')
        plt.close()
        print("Generated plot: plot_speedup.png")


def print_summary_table(summary):
    """Prints a summary table in Markdown format."""
    if not summary: return
    df = pd.DataFrame.from_dict(summary, orient='index')
    df.index.name = "Actors"
    print("\n--- Experimental Results Summary ---")
    print(df.to_markdown(floatfmt=".2f")) # Format floats to 2 decimal places
    print("------------------------------------\n")


if __name__ == "__main__":
    if not os.path.isdir(LOG_DIRECTORY):
        print(f"Error: The directory '{LOG_DIRECTORY}' was not found in the current folder.")
    else:
        all_log_data = {}
        for filename in os.listdir(LOG_DIRECTORY):
            match = LOG_FILE_PATTERN.match(filename)
            if match:
                num_actors = int(match.groups()[0])
                full_filepath = os.path.join(LOG_DIRECTORY, filename)
                print(f"Parsing {full_filepath}...")
                data, duration = parse_log_file(full_filepath)
                if data:
                    all_log_data[num_actors] = (data, duration)

        if not all_log_data:
            print(f"\nError: Parsing failed. No usable data was extracted from log files in the '{LOG_DIRECTORY}' folder.")
            print("Please check that the log files contain valid 'finished episode' and 'Current Loss' messages.")
        else:
            summary_metrics = analyze_results(all_log_data)
            print_summary_table(summary_metrics)
            plot_results(all_log_data, summary_metrics)
            print("Analysis complete.")

