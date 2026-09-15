"""Vectorized Monte Carlo; all reported task values are evaluated exactly."""
from __future__ import annotations
import numpy as np
from scipy.special import xlogy
from .core import Problem

METHODS = {
    "task_task": ("task", "task"),
    "full_task": ("full", "task"),
    "full_full": ("full", "full"),
    "task_full": ("task", "full"),
    "uniform_task": ("uniform", "task"),
    "entropy_task": ("entropy", "task"),
}


def simulate(problem: Problem, method: str, repetitions: int = 200,
             cap: int = 20000, delta: float = .05, seed: int = 1701,
             actual_kernels: np.ndarray | None = None,
             log_radius: np.ndarray | None = None,
             checkpoints: tuple[int, ...] = ()) -> tuple[list[dict], list[dict]]:
    """Run independent calibration replicas with paired random-number streams.

    Truth is drawn uniformly from candidate permutations. At a cap an
    uncertified run remains explicitly censored; it is never called certified.
    Method prefixes control query design and suffixes control stopping.
    entropy_task maximizes the full-mapping posterior mutual information.
    log_radius, when supplied, is the justified per-context log model error.
    The chain deployment dynamics remain known even when calibration kernels
    are estimated; this isolates likelihood-model error.
    """
    if method not in METHODS or repetitions <= 0 or cap <= 0 or not 0 < delta < 1:
        raise ValueError("Invalid simulation configuration")
    sampling, stopping = METHODS[method]
    rng = np.random.default_rng(seed)
    true = rng.integers(problem.M, size=repetitions)
    actual = problem if actual_kernels is None else Problem(
        actual_kernels, problem.required, problem.horizon, problem.good, problem.bad)
    if actual.probs.shape != problem.probs.shape:
        raise ValueError("True and nominal models have incompatible shapes")
    weights = None
    if sampling in ("task", "full"):
        weights, _ = problem.designs(sampling)
    elif sampling == "uniform":
        weights = np.full((problem.M, problem.Q), 1 / problem.Q)
    entropy = -np.sum(xlogy(problem.probs, problem.probs), axis=-1)
    conflicts = problem.task_conflicts if stopping == "task" else problem.full_conflicts
    ll = np.zeros((repetitions, problem.M))
    active = np.ones(repetitions, dtype=bool)
    n = np.full(repetitions, cap, dtype=int)
    predicted = np.zeros(repetitions, dtype=int)
    csize = np.full(repetitions, problem.M, dtype=int)
    diagnostic = np.zeros(repetitions, dtype=int)
    coverage_failure = np.zeros(repetitions, dtype=bool)
    error_budget = np.zeros(repetitions)
    radius = np.zeros(problem.contexts) if log_radius is None else np.asarray(log_radius)
    if radius.shape != (problem.contexts,) or np.any(radius < 0):
        raise ValueError("Invalid log-error radii")
    B = np.log((problem.M - 1) / delta)
    checkpoint_set = set(checkpoints)
    curves = []
    for t in range(1, cap + 1):
        # Draw for ALL replicas at every step, maintaining common random
        # numbers across methods even when their stopping times differ.
        u_query = rng.random(repetitions)
        u_outcome = rng.random(repetitions)
        ix = np.flatnonzero(active)
        if not len(ix):
            break
        hats = np.argmax(ll[ix], axis=1)
        if sampling == "entropy":
            posterior = np.exp(ll[ix] - ll[ix].max(axis=1, keepdims=True))
            posterior /= posterior.sum(axis=1, keepdims=True)
            mixture = np.einsum('bm,mqy->bqy', posterior, problem.probs, optimize=True)
            information = -np.sum(xlogy(mixture, mixture), axis=-1) - posterior @ entropy
            w = np.zeros((len(ix), problem.Q))
            w[np.arange(len(ix)), np.argmax(information, axis=1)] = 1
        else:
            w = weights[hats]
        exploration = t ** -.5
        if sampling != "uniform":
            w = (1-exploration) * w + exploration / problem.Q
        q = np.minimum((u_query[ix, None] > np.cumsum(w, axis=1)).sum(axis=1), problem.Q-1)
        probabilities = actual.probs[true[ix], q]
        y = np.minimum((u_outcome[ix, None] > np.cumsum(probabilities, axis=1)).sum(axis=1), problem.outcomes-1)
        ll[ix] += problem.logs[:, q, y].T
        ll[ix] -= ll[ix].max(axis=1, keepdims=True)
        error_budget[ix] += radius[q // problem.m]
        threshold = B + 2 * error_budget[ix]
        conf = ll[ix] >= -threshold[:, None]
        coverage_failure[ix] |= ~conf[np.arange(len(ix)), true[ix]]
        hats = np.argmax(ll[ix], axis=1)
        predicted[ix] = hats
        csize[ix] = conf.sum(axis=1)
        diagnostic[ix] += q < problem.m
        stop = ~np.any(conf & conflicts[hats], axis=1)
        done = ix[stop]
        n[done] = t
        active[done] = False
        if t in checkpoint_set:
            values = problem.values[true, predicted]
            curves.append({"budget": t, "mean_value": float(values.mean()),
                           "mean_regret": float((problem.optimal_value-values).mean()),
                           "certified_fraction": float((~active).mean()),
                           "task_correct_fraction": float(np.mean(~problem.task_conflicts[true, predicted])),
                           "full_correct_fraction": float(np.mean(true == predicted))})
    rows = []
    for b in range(repetitions):
        value = float(problem.values[true[b], predicted[b]])
        rows.append({"replicate": b, "seed": seed, "method": method, "true_mapping": int(true[b]),
                     "estimated_mapping": int(predicted[b]), "transitions": int(n[b]),
                     "cap": cap, "certified": bool(not active[b]),
                     "censored": bool(active[b]), "task_correct": bool(not problem.task_conflicts[true[b], predicted[b]]),
                     "mapping_correct": bool(true[b] == predicted[b]),
                     "value": value, "regret": float(problem.optimal_value-value),
                     "confidence_size": int(csize[b]), "diagnostic_fraction": float(diagnostic[b]/n[b]),
                     "coverage_failure": bool(coverage_failure[b]),
                     "log_error_budget": float(error_budget[b])})
    return rows, curves
