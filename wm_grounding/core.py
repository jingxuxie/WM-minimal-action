"""Finite latent-action grounding, optimal designs, and anytime certificates.

The exact-model guarantees assume a common finite support and an unknown
permutation shared across all allowed calibration contexts. No action labels
are inferred from arbitrary passive data by this module.
"""
from __future__ import annotations
from dataclasses import dataclass
from itertools import permutations, combinations
from typing import Iterable
import numpy as np
from numpy.typing import NDArray
from scipy.optimize import linprog

Array = NDArray[np.float64]


def all_permutations(m: int) -> NDArray[np.int64]:
    if not 2 <= m <= 8:
        raise ValueError("Explicit enumeration supports 2 <= m <= 8.")
    return np.asarray(list(permutations(range(m))), dtype=np.int64)


def validate_kernels(kernels: Array) -> Array:
    p = np.asarray(kernels, dtype=float)
    if p.ndim != 3 or not np.all(np.isfinite(p)) or np.any(p <= 0):
        raise ValueError("kernels must be positive finite [context,effect,outcome] arrays")
    if not np.allclose(p.sum(-1), 1, atol=1e-10):
        raise ValueError("Every conditional distribution must sum to one.")
    return p


def categorical_kl(p: Array, q: Array) -> Array:
    return np.sum(p * (np.log(p) - np.log(q)), axis=-1)


@dataclass
class Problem:
    kernels: Array
    required: tuple[int, ...] = (0, 1)
    horizon: int = 4
    good: float = 0.8
    bad: float = 0.2

    def __post_init__(self) -> None:
        self.kernels = validate_kernels(self.kernels)
        self.contexts, self.m, self.outcomes = self.kernels.shape
        if not self.required or any(r < 0 or r >= self.m for r in self.required):
            raise ValueError("required must be a nonempty subset of the effects")
        if len(set(self.required)) != len(self.required):
            raise ValueError("required effects must not repeat")
        if self.horizon < len(self.required) or not 0 < self.bad < self.good < 1:
            raise ValueError("Invalid chain parameters")
        self.perms = all_permutations(self.m)
        self.M = len(self.perms)
        self.Q = self.contexts * self.m
        # Query q = context*m + executable_action.
        self.probs = np.stack([self.kernels[:, p, :].reshape(self.Q, self.outcomes)
                               for p in self.perms])
        self.logs = np.log(self.probs)
        self.inverse = np.argsort(self.perms, axis=1)
        self.answers = self.inverse[:, self.required]
        self.task_conflicts = np.any(self.answers[:, None] != self.answers[None, :], axis=-1)
        self.full_conflicts = ~np.eye(self.M, dtype=bool)
        # Each column is the chain policy obtained from that mapping.
        seq = np.asarray([self.required[k % len(self.required)] for k in range(self.horizon)])
        commands = self.inverse[:, seq]
        self.values = np.empty((self.M, self.M))
        for theta in range(self.M):
            effects = self.perms[theta, commands]
            self.values[theta] = np.prod(np.where(effects == seq, self.good, self.bad), axis=-1)
        self.optimal_value = self.good ** self.horizon
        self.regrets = self.optimal_value - self.values
        self._kl_cache: dict[int, Array] = {}
        self._design_cache: dict[str, tuple[Array, Array]] = {}

    def kl_from(self, theta: int) -> Array:
        """KL against every candidate, with shape [candidate,query]."""
        if theta not in self._kl_cache:
            self._kl_cache[theta] = np.maximum(categorical_kl(self.probs[theta][None], self.probs), 0)
        return self._kl_cache[theta]

    def designs(self, target: str = "task") -> tuple[Array, Array]:
        if target not in ("task", "full"):
            raise ValueError("target must be task or full")
        if target not in self._design_cache:
            conflicts = self.task_conflicts if target == "task" else self.full_conflicts
            weights, rates = [], []
            for h in range(self.M):
                w, d = maximin_design(self.kl_from(h)[conflicts[h]])
                weights.append(w)
                rates.append(d)
            self._design_cache[target] = np.asarray(weights), np.asarray(rates)
        return self._design_cache[target]


def maximin_design(divergences: Array) -> tuple[Array, float]:
    """Maximize the worst alternative's information rate using a linear program."""
    d = np.asarray(divergences, dtype=float)
    if d.ndim != 2 or not d.shape[0] or np.any(d < -1e-10):
        raise ValueError("Need a nonempty matrix of nonnegative divergences")
    n_alt, nq = d.shape
    objective = np.r_[np.zeros(nq), -1.0]
    result = linprog(objective, A_ub=np.c_[-d, np.ones(n_alt)],
                     b_ub=np.zeros(n_alt), A_eq=np.r_[np.ones(nq), 0.][None],
                     b_eq=[1.], bounds=[(0., None)] * (nq + 1), method="highs")
    if not result.success:
        raise RuntimeError(f"Optimal-design LP failed: {result.message}")
    w = np.maximum(result.x[:-1], 0)
    w /= w.sum()
    # Return the actually achieved rate, not the solver objective alone.
    return w, float(np.min(d @ w))


def simple_cycles(m: int, required: Iterable[int] | None = None):
    """Enumerate directed cycles once each, modulo rotation (not reversal)."""
    required = set(range(m)) if required is None else set(required)
    for k in range(2, m + 1):
        for subset in combinations(range(m), k):
            if not required.intersection(subset):
                continue
            root = min(subset)
            for order in permutations([x for x in subset if x != root]):
                yield (root,) + order


def cycle_cost(cost: Array, cycle: tuple[int, ...]) -> float:
    return float(sum(cost[cycle[k], cycle[(k + 1) % len(cycle)]] for k in range(len(cycle))))


def shortest_relevant_cycle(cost: Array, required: Iterable[int]) -> tuple[float, tuple[int, ...]]:
    """O(m^3 + |R|m) shortest directed nontrivial cycle touching R.

    Nonnegative edge costs are essential. Ties are left unchanged during
    Floyd-Warshall so zero-weight cycles cannot create next-hop loops.
    """
    cost = np.asarray(cost, dtype=float)
    m = len(cost)
    if cost.shape != (m, m) or np.any(cost < -1e-12):
        raise ValueError("Expected a square nonnegative cost matrix")
    required = tuple(required)
    if not required or any(r < 0 or r >= m for r in required):
        raise ValueError("Invalid required vertices")
    dist = cost.copy()
    np.fill_diagonal(dist, 0)
    nxt = np.tile(np.arange(m), (m, 1))
    for k in range(m):
        candidate = dist[:, k, None] + dist[k, None, :]
        improve = candidate < dist - 1e-13
        rows, cols = np.where(improve)
        dist[rows, cols] = candidate[rows, cols]
        nxt[rows, cols] = nxt[rows, k]
    best, best_cycle = np.inf, ()
    for r in required:
        for j in range(m):
            if j == r:
                continue
            val = cost[r, j] + dist[j, r]
            if val < best:
                path = [r, j]
                u = j
                for _ in range(m):
                    u = int(nxt[u, r])
                    if u == r:
                        break
                    if u in path:
                        raise RuntimeError("Unexpected cycle in shortest path reconstruction")
                    path.append(u)
                else:
                    raise RuntimeError("Shortest path did not terminate")
                best, best_cycle = float(val), tuple(path)
    return best, best_cycle


def grounding_cost_matrix(kernels: Array, theta: NDArray[np.int64], weights: Array) -> Array:
    p = validate_kernels(kernels)
    x, m, _ = p.shape
    w = np.asarray(weights).reshape(x, m)
    inv = np.argsort(theta)
    dij = categorical_kl(p[:, :, None, :], p[:, None, :, :])
    return np.einsum("xi,xij->ij", w[:, inv], dij)


def cycle_design(kernels: Array, theta: NDArray[np.int64], required: Iterable[int],
                 tolerance: float = 1e-9, max_iterations: int = 10000) -> tuple[Array, float, int]:
    """Solve the information-design LP by adding violated cycle constraints.

    This avoids enumerating alternative permutations for design computation.
    Likelihood inference elsewhere in this reference code is still factorial.
    """
    p = validate_kernels(kernels)
    x, m, _ = p.shape
    required = tuple(required)
    inv = np.argsort(theta)
    dij = categorical_kl(p[:, :, None, :], p[:, None, :, :])
    known = {(min(r, j), max(r, j)) for r in required for j in range(m) if r != j}

    def row(c):
        out = np.zeros((x, m))
        for i, j in zip(c, c[1:] + c[:1]):
            out[:, inv[i]] += dij[:, i, j]
        return out.ravel()

    rows = [row(c) for c in sorted(known)]
    for iteration in range(1, max_iterations + 1):
        w, current = maximin_design(np.asarray(rows))
        cost = grounding_cost_matrix(p, theta, w)
        actual, violated = shortest_relevant_cycle(cost, required)
        if actual >= current - tolerance:
            return w, actual, iteration
        if violated in known:
            raise RuntimeError("Numerical inconsistency in cutting-plane LP")
        known.add(violated)
        rows.append(row(violated))
    raise RuntimeError("Cycle design did not converge within the iteration limit")


def confidence_set(log_likelihoods: Array, delta: float, error_budget: Array | float = 0.) -> NDArray[np.bool_]:
    """Time-uniform likelihood-ratio set, including robust 2*sum eta correction.

    error_budget is accumulated per-query log-model error, NOT its doubled
    value. It must be justified externally; zero is for exact models.
    """
    ll = np.asarray(log_likelihoods)
    if not 0 < delta < 1 or ll.shape[-1] < 2:
        raise ValueError("Require 0 < delta < 1 and at least two hypotheses")
    threshold = np.log((ll.shape[-1] - 1) / delta) + 2 * np.asarray(error_budget)
    return ll.max(axis=-1, keepdims=True) - ll <= np.expand_dims(threshold, -1)


def policy_certificate(confidence: NDArray[np.bool_], regrets: Array,
                       epsilon: float, value_error: float = 0.) -> tuple[int | None, float]:
    """Find one candidate policy with uniformly small regret.

    regrets[model,policy] must come from the same model used in the set.
    value_error uniformly bounds |V_true(pi)-V_nominal(pi)|, e.g. H*rho.
    """
    if not np.any(confidence):
        return None, float("inf")
    bounds = regrets[confidence].max(axis=0) + 2 * value_error
    policy = int(np.argmin(bounds))
    return (policy if bounds[policy] <= epsilon else None), float(bounds[policy])
