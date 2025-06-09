#!/bin/bash

# Define the activity IDs to process
ACTIVITY_IDS=(5 9 10 11 12) # All valid activities including lifting

# Path to your MATLAB executable
MATLAB_EXECUTABLE="/Applications/MATLAB_R2025a.app/bin/matlab"

# Get the directory of this script, which should be the project root
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# Log file for combined output
LOG_FILE="${SCRIPT_DIR}/matlab_processing_log.txt"

# Clear previous log file
echo "Starting MATLAB batch processing at $(date)" > "$LOG_FILE"
echo "Processing activities: ${ACTIVITY_IDS[*]}" >> "$LOG_FILE"
echo "-------------------------------------------------" >> "$LOG_FILE"

# Loop through each activity ID
for id in "${ACTIVITY_IDS[@]}"
do
  echo ""
  echo "======================================================================"
  echo "Shell loop: Starting processing for Activity ID: $id"
  echo "======================================================================"
  
  # Run the MATLAB script as a function, passing the activity ID
  (cd "$SCRIPT_DIR" && "$MATLAB_EXECUTABLE" -batch "generate_wifi_from_video($id)") 2>&1 | tee -a "$LOG_FILE"
  
  MATLAB_EXIT_STATUS=${PIPESTATUS[0]}

  if [ $MATLAB_EXIT_STATUS -ne 0 ]; then
    echo ""
    echo "----------------------------------------------------------------------"
    echo "ERROR: MATLAB script failed for activity_id = $id (Exit Status: $MATLAB_EXIT_STATUS)"
    echo "----------------------------------------------------------------------"
    echo "ERROR logged for activity_id = $id at $(date)" >> "$LOG_FILE"
    # Uncomment the next line if you want the script to stop on the first error
    # exit 1 
  else
    echo ""
    echo "----------------------------------------------------------------------"
    echo "Successfully processed Activity ID: $id"
    echo "----------------------------------------------------------------------"
    echo "Successfully processed Activity ID: $id at $(date)" >> "$LOG_FILE"
  fi
done

echo ""
echo "======================================================================"
echo "All specified activities processed. Check $LOG_FILE for details."
echo "======================================================================" 