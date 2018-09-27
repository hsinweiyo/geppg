#!/bin/bash
for i in {0..10000..60}; do
        python test_gep_2.py --trial_id $i --n_neighbors 1 --noise 0.1 --exp_folder reacher-ablative/ --output reacher_goal_origin.csv | grep "Average error" >> perf_output
done
