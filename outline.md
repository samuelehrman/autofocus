# SEM Autofocus Algorithm

This is a 1D Optimization Problem
Find the optimal working distance
Optimize based off of both total time and accuracy

Explanation of algorithm:

Step 1: It tries to do a quick initial search with a large HFW

Steo 2: It tries to do a quick search with a small HFW

Step 3: If it hasn't converged, it slows down and has twice the dwell time (ie. higher S/N ratio)
Check both large and small HFW and determines which one gives a better signal
Tries to converge with the best metric with higher S/N ratio

CI index is a measure of the confidence of the measurment being in the right region

