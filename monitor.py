import sys
from datetime import datetime
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--filename", default=f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.txt", help="Log file path")
args = parser.parse_args()
output_file = args.filename

# Define your target string
SKIPPED_STRING = "Unable to solve directly, using least squares instead."
output_stack = []
start_time = datetime.now().time()
with open(f"logs/{output_file}", "a") as f:
    f.write(f"Monitoring started at {start_time}\n")

# Read line-by-line from the terminal stream
for line in sys.stdin:
    # Print the output back to your screen so you don't lose sight of it
    sys.stdout.write(line)
    sys.stdout.flush()

    # Trigger condition
    if SKIPPED_STRING in line:
        if output_stack:
            continue  # Skip logging if the line is already in the output stack
        else:
            with open(f"logs/{output_file}", "a") as f:
                f.write(f"Alert! Found {SKIPPED_STRING} in line: {line} at time {datetime.now().time()}\n")
                output_stack.append(line)
    else:
        with open(f"logs/{output_file}", "a") as f:
            f.write(f"Alert! Found line: {line} at time {datetime.now().time()}\n")

with open(f"logs/{output_file}", "a") as f:
    f.write(f"Monitoring ended at {datetime.now().time()}\n")