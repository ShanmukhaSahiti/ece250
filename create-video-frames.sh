#!/bin/bash

# This script creates the 'video_frames' directory with the correct structure.
# It copies .jpg files directly from each video segment folder in 'ucf action',
# ignoring any 'jpeg' or 'gt' subdirectories.

set -e

# Define source and destination directories
SRC_DIR="ucf action"
DEST_DIR="video_frames"

echo "--- Creating 'video_frames' directory from '$SRC_DIR' ---"

# 1. Clean up and recreate the destination directory
echo "Removing old '$DEST_DIR' and creating a new one."
rm -rf "$DEST_DIR"
mkdir -p "$DEST_DIR"

# 2. Function to process and copy an activity
process_activity() {
    local src_name="$1"
    local dest_name="$2"
    local src_path="$SRC_DIR/$src_name"
    local dest_path="$DEST_DIR/$dest_name"

    if [ -d "$src_path" ]; then
        echo "Processing '$src_name' -> '$dest_name'..."
        mkdir -p "$dest_path"
        
        # Loop through each item in the source activity folder
        for segment_dir in "$src_path"/*; do
            # Check if it's a directory (i.e., a video segment folder)
            if [ -d "$segment_dir" ]; then
                local segment_name=$(basename "$segment_dir")

                # Skip any 'jpeg' or 'gt' folders as per user instruction
                if [ "$segment_name" == "jpeg" ] || [ "$segment_name" == "gt" ]; then
                    echo "  - Ignoring directory: $segment_name"
                    continue
                fi

                local dest_segment_path="$dest_path/v-$segment_name"
                mkdir -p "$dest_segment_path"

                # Find and copy only .jpg files
                if ls "$segment_dir"/*.jpg 1> /dev/null 2>&1; then
                    echo "  + Copying JPGs from '$segment_name'..."
                    cp "$segment_dir"/*.jpg "$dest_segment_path/"
                else
                    echo "  ! No .jpg files found in '$segment_dir'."
                fi
            fi
        done
    else
        echo "Warning: Source directory '$src_path' not found. Skipping."
    fi
}

# 3. Process all 8 activities
process_activity "Diving-Side" "Diving-Side"
process_activity "Golf-Swing-Side" "Golf-Swing-Side"
process_activity "Kicking-Front" "Kicking-Front"
process_activity "Lifting" "lifting"
process_activity "Swing-Bench" "Swing-Bench"
process_activity "Run-Side" "run-side"
process_activity "SkateBoarding-Front" "skateboarding-front"
process_activity "Walk-Front" "walk-front"

echo "--- 'video_frames' directory has been successfully created. ---" 