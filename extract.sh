#!/bin/bash

# Directory containing the .tar.gz files
SOURCE_DIR="./nihcc"

# Target directory for the extracted images
TARGET_DIR="${SOURCE_DIR}/"

# Create the target directory if it doesn't exist
mkdir -p "${TARGET_DIR}"

# Find and extract all .tar.gz files, then move the images to the target directory
find "${SOURCE_DIR}" -name "*.tar.gz" -exec tar -xzvf {} -C "${TARGET_DIR}" \;

echo "All images have been extracted to ${TARGET_DIR}."
