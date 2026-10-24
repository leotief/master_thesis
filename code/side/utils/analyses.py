# helpers for the experiments of estimate.py
#
# every method is an estimator called as estimator(w, x_partial, **kwargs). on
# a batch of test points x_partial (B, D) it returns the estimates of
# E[sigmoid(w^T x) | x_obs] for all of them in one call, as an array (B,). the
# kwargs are the sample or node counts that are swept (n_samples, n_outer,
# n_inner, n_nodes), without them the estimator uses the ones of its constructor.
#   rmse        error of one method over the test points
#   timed_rmse  the same, the time of one call over all test points and its estimates

import time

import numpy as np

# root mean squared error over the test points (B, D), against their exact
# values (B,). for the random methods every test point gets one run, so the
# noise is part of the error
def rmse(estimator, w, points, truths, **kwargs):
    return float(np.sqrt(np.mean(np.square(estimator(w, points, **kwargs) - truths))))

# rmse and the wall clock time of one batched call over all points, and the
# estimates of that call. the truths belong to the first len(truths) points,
# the error is over those, the time is for all of them. after one call to warm
# up (gpu kernels, cached nodes), the call is repeated `repeats` times and the
# time is the median. the estimators return arrays on the cpu, so the gpu has
# finished when the clock stops
def timed_rmse(estimator, w, points, truths, repeats=3, **kwargs):
    estimator(w, points, **kwargs)
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        estimates = estimator(w, points, **kwargs)
        times.append(time.perf_counter() - t0)
        if len(times) == 1:
            first = estimates
            error = float(np.sqrt(np.mean(np.square(estimates[:len(truths)] - truths))))
    return error, float(np.median(times)), first
