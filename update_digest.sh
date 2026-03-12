#!/bin/bash

# Exit on error
set -e

echo "Starting research digest update..."

# Run the python script
# The script already has logic to find missing intervals
python3 paper_search.py

echo "Digest update complete."
