import sys
from datetime import datetime

# Define your target string
TARGET_STRING1 = "fit_file starting — MC first, then CNMF init"
TARGET_STRING2 = "fit_file done"
start_time = datetime.now().time()
with open("logs/triggered_output.txt", "a") as f:
    f.write(f"Monitoring started at {start_time}\n")

# Read line-by-line from the terminal stream
for line in sys.stdin:
    # Print the output back to your screen so you don't lose sight of it
    sys.stdout.write(line)
    sys.stdout.flush()

    # Trigger condition
    if TARGET_STRING1 in line:
        with open("logs/triggered_output.txt", "a") as f:
            f.write(f"Monitoring started at {start_time}\n")
            f.write(f"Alert! Found {TARGET_STRING1} in line: {line} at time {datetime.now().time()}\n")
    elif TARGET_STRING2 in line:
        with open("logs/triggered_output.txt", "a") as f:
            f.write(f"Alert! Found {TARGET_STRING2} in line: {line} at time {datetime.now().time()}\n")
