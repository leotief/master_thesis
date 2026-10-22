# experiments for main.py
#
# every method is passed as (estimator, budget_kwargs): the estimator is called as
# estimator(w, x_partial, **budget_kwargs(budget)) and returns one estimate of
# E[sigmoid(w^T x) | x_obs] as a float. the budget is the number of cf
# evaluations (samples for mc), so methods are compared at the same cost.
# budget_kwargs None means the method is deterministic, takes no budget and
# only needs to run once.
#   compare_estimators    all methods at one budget, table with the error vs gt

import time

import numpy as np

# repeated runs of one method: estimates and time per run in ms
def _run(method, w, x_partial, budget, n_runs):
    estimator, budget_kwargs = method
    kwargs = budget_kwargs(budget) if budget_kwargs else {}
    n_runs = n_runs if budget_kwargs else 1
    t0   = time.perf_counter()
    vals = [estimator(w, x_partial, **kwargs) for _ in range(n_runs)]
    return np.array(vals), (time.perf_counter() - t0) / n_runs * 1000

# prints mean +- std, absolute error of the mean against the gt method and time
# per run for every method at the same budget
def compare_estimators(methods, w, x_partial, gt, budget, n_runs=10):
    gt_val = _run(methods[gt], w, x_partial, budget, 1)[0].mean()
    print(f"budget {budget}, {n_runs} runs for the random methods\n")
    print(f"{'method':<20}  {'mean':>8}  {'std':>9}  {'|error|':>9}  {'ms/run':>8}")
    for name, method in methods.items():
        vals, ms = _run(method, w, x_partial, budget, n_runs)
        print(f"{name:<20}  {vals.mean():8.5f}  {vals.std():9.2e}  {abs(vals.mean() - gt_val):9.2e}  {ms:8.1f}")
    print()
