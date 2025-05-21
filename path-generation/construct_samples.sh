#!/bin/bash

# Check if a directory path is provided as an argument
if [[ $# -ne 1 ]]; then
  echo "Usage: $0 <directory>"
  exit 1
fi

# Get the directory path from the command line argument
output_dir="$1"

# Check if the directory exists; if not, create it
if [[ ! -d "$output_dir" ]]; then
  echo "Directory $output_dir does not exist. Creating it now."
  mkdir -p "$output_dir"
fi

echo "Output file is: $1"

echo "Building Pairs..." | tee -a "$output_file"
python3 generate_samples.py $1
if [[ $? -ne 0 ]]; then
  echo "generate_samples.py failed with an error." | tee -a "$output_file"
  exit 1
fi

# Run the second Python script and echo output to the file
echo "Generating Images..." | tee -a "$output_file"
python3 generate_image.py $1
if [[ $? -ne 0 ]]; then
  echo "generate_image.py failed with an error." | tee -a "$output_file"
  exit 1
fi