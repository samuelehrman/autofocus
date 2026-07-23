# SEM Autofocus Algorithm

This is a 1D Optimization Problem
Find the optimal working distance
Optimize based off of both total time and accuracy

Explanation of algorithm:

Step 1: Quick coarse search with a large HFW. If CI is high enough, refine (convergent search) around the best WD.

Step 2: If not, quick coarse search with a small HFW. If CI is high enough, refine around that best WD.

Step 3: If it still hasn't converged, increase dwell (higher S/N).
Coarse-search both large and small HFW, pick the better signal (CI, with optional preference factors),
then refine around that best WD with still-higher dwell.

CI index = max(IQ) / mean(IQ): confidence that the focus peak is in the right region.
