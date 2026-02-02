#!/bin/bash
# Test script for merging reconstructions using the 5014 sample

# Split the 5014 reconstruction into two overlapping parts
echo "Step 1: Splitting reconstruction into two overlapping parts..."
python split_reconstruction.py ../assets/5014 ./test_data --overlap 0.3

# Merge the two parts back together
echo ""
echo "Step 2: Merging the two parts..."
python ESSN_merge_reconstruction.py \
    ./test_data/rec_A \
    ./test_data/rec_B \
    ./test_data/merged \
    --mode se3 \
    --save-transform

echo ""
echo "Done! Check the results in ./test_data/"
echo "  - rec_A/          : First half of reconstruction"
echo "  - rec_B/          : Second half of reconstruction"
echo "  - merged/         : Merged reconstruction"
echo "  - merged/transformation.npy : The computed transformation matrix"
