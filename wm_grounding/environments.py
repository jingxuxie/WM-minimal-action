"""Small synthetic families with an explicit passive-pretraining interface."""
from __future__ import annotations
import numpy as np
from .core import Problem


def two_bit_distribution(p: float, z: float) -> np.ndarray:
    return np.asarray([(1-p)*(1-z), (1-p)*z, p*(1-z), p*z])


def diagnostic_chain(gamma: float = .08, contexts: int = 9, weak: float = .15,
                     required: tuple[int, ...] = (0, 1), horizon: int = 4) -> Problem:
    """Four effects, with a pair of increasingly similar nuisance effects.

    Calibration contexts emit two observable bits; reset access is allowed.
    Deployment is a layered stochastic chain alternating required effects.
    A global command permutation is shared between calibration and deployment.
    """
    if not 0 < gamma < .3 or contexts < 1 or not 0 < weak <= 1:
        raise ValueError("Invalid environment parameters")
    parameters = np.asarray([[.8, .5], [.2, .8], [.2, .5-gamma], [.2, .5+gamma]])
    p = np.empty((contexts, 4, 4))
    for x in range(contexts):
        strength = 1. if x == 0 else weak
        for j in range(4):
            a, b = .5 + strength * (parameters[j] - .5)
            p[x, j] = two_bit_distribution(a, b)
    return Problem(p, required=required, horizon=horizon)


def learned_anchor_kernels(problem: Problem, samples_per_effect_context: int,
                           rng: np.random.Generator) -> np.ndarray:
    """Categorical MLE with Laplace smoothing from anonymous pure demonstrators.

    Each anonymous demonstrator executes one fixed latent effect, consistently
    across contexts. Its identity is observable but its executable command is
    not. This strong anchor structure is explicit; this is NOT arbitrary-video
    latent-action discovery, and no neural encoder is trained.
    """
    if samples_per_effect_context < 1:
        raise ValueError("Need positive pretraining sample size")
    p = problem.kernels
    learned = np.empty_like(p)
    for x in range(problem.contexts):
        for j in range(problem.m):
            counts = rng.multinomial(samples_per_effect_context, p[x, j])
            learned[x, j] = (counts + 1) / (samples_per_effect_context + problem.outcomes)
    return learned


def exact_log_radius(true: np.ndarray, learned: np.ndarray) -> np.ndarray:
    """Per-context worst log error, using simulator truth for diagnostic runs.

    Oracle radii must not be described as data-derived confidence radii.
    """
    return np.max(np.abs(np.log(true) - np.log(learned)), axis=(1, 2))
