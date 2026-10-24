# the estimators take one point x_partial (D,) or a batch (B, D). a batch is
# one call over all points, so the gpu does the work in a few large operations
# instead of many small ones. the intermediate tensors grow with the number of
# points, so the batch is split into chunks that keep them below max_elements

import numpy as np
import torch

# fn on the points of x_partial, in chunks of points, each point costs
# per_point elements of the largest intermediate tensor. a float for one point,
# an array of shape (B,) for a batch
def in_chunks(fn, x_partial, per_point, max_elements=2**25):
    if x_partial.dim() == 1:
        return float(fn(x_partial))
    size = max(1, max_elements // per_point)
    return torch.cat([fn(part) for part in x_partial.split(size)]).cpu().numpy()
