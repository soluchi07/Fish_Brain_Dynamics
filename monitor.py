import sys
import os
from datetime import datetime
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--filename", default=f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.txt", help="Log file path")
args = parser.parse_args()
output_file = args.filename

SKIPPED_STRING = "Unable to solve directly, using least squares instead."

os.makedirs("logs", exist_ok=True)

trigger_count = 0

with open(f"logs/{output_file}", "a") as f:
    f.write(f"Monitoring started at {datetime.now().time()}\n")

    for line in sys.stdin:
        sys.stdout.write(line)
        sys.stdout.flush()

        if SKIPPED_STRING in line:
            trigger_count += 1
        else:
            if trigger_count > 0:
                f.write(f"Alert! '{SKIPPED_STRING}' fired {trigger_count}x — last at {datetime.now().time()}\n")
                trigger_count = 0
            f.write(f"{line.strip()} at time {datetime.now().time()}\n")

    if trigger_count > 0:
        f.write(f"Alert! '{SKIPPED_STRING}' fired {trigger_count}x — last at {datetime.now().time()}\n")

    f.write(f"Monitoring ended at {datetime.now().time()}\n")
