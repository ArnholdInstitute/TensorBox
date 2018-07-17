This C++ code is used to "solve the Hungarian problem".
Specifically, we have the following problem: given a set of ground-truth bounding boxes,
and a sequence of predicted bounding boxes (wherein each GT box may have multiple predictions
corresponding to it), assign at most one prediction box to each ground-truth box.

Potential assignments are given a score: 
(# of non-overlapping boxes, sum of ranks of predictions in sequence, 
 sum of L1-distances between prediction and GT box).
Assignments are then ordered lexicographically by this score, and the least assignment is chosen.
See [https://arxiv.org/pdf/1506.04878.pdf] for details.