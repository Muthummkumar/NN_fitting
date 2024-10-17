#!/bin/bash


# Defining file path & log file path ...
SCRIPT_NAME=$(pwd)/"fitting.py"  # Your Python script filename
LOG_FILE="training_log_$(date +'%Y_%m_%d_%H_%M_%S').log"  # Log file with timestamp


# Check if the Python script exists
if [[ ! -f "$SCRIPT_NAME" ]]; then
  echo "Error: Python script '$SCRIPT_NAME' not found!"
  exit 1
fi

# Run the Python script and log the output
echo "Running $SCRIPT_NAME and logging output to $LOG_FILE..."
python3 "$SCRIPT_NAME" > "$LOG_FILE" 2>&1

# Checking if the script ran successfully
# 0 means the previous process ran successfully 1 means it fails 
if [[ $? -eq 0 ]]; then
  echo "Script ran successfully. Logs saved in $LOG_FILE."
else
  echo "Error occurred during execution. Check $LOG_FILE for details."
fi

