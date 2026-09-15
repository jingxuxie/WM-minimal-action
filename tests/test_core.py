from itertools import permutations
import numpy as np
import pytest
from wm_grounding.core import (Problem, categorical_kl, simple_cycles, cycle_cost,
    shortest_relevant_cycle, grounding_cost_matrix, cycle_design, confidence_set,
    policy_certificate)
from wm_grounding.environments import diagnostic_chain, two_bit_distribution
from wm_grounding.simulation import simulate


def test_probabilities_and_passive_invariance():
    p = diagnostic_chain()
    assert np.allclose(p.probs.sum(-1), 1)
    marginal = p.probs.reshape(p.M, p.contexts, p.m, p.outcomes).mean(axis=2)
    assert np.allclose(marginal, marginal[0])


def test_kl_identity():
    p = diagnostic_chain()
    for h in [0, 7, 23]:
        assert np.allclose(p.kl_from(h)[h], 0, atol=1e-12)
        assert np.all(p.kl_from(h) >= 0)


@pytest.mark.parametrize('m', [2,3,4,5])
def test_cycle_oracle_matches_enumeration(m):
    rng = np.random.default_rng(122+m)
    for _ in range(15):
        costs = rng.random((m,m))
        costs[rng.random((m,m)) < .15] = 0
        np.fill_diagonal(costs, 0)
        for required in [(0,), tuple(range(m))]:
            expected = min(cycle_cost(costs,c) for c in simple_cycles(m,required))
            actual, c = shortest_relevant_cycle(costs,required)
            assert len(c) >= 2 and len(set(c)) == len(c)
            assert set(c).intersection(required)
            assert actual == pytest.approx(expected, abs=1e-10)
            assert cycle_cost(costs,c) == pytest.approx(actual, abs=1e-10)


@pytest.mark.parametrize('m', [2,3,4,5])
def test_permutation_cycle_theorem(m):
    rng = np.random.default_rng(m)
    kernels = rng.dirichlet(np.ones(4), size=(2,m))
    p = Problem(kernels, required=(0,), horizon=3)
    for h in rng.choice(p.M, size=min(4,p.M), replace=False):
        weights = rng.dirichlet(np.ones(p.Q))
        cost = grounding_cost_matrix(kernels,p.perms[h],weights)
        expected = np.min(p.kl_from(h)[p.task_conflicts[h]] @ weights)
        actual,c = shortest_relevant_cycle(cost,(0,))
        assert actual == pytest.approx(expected, abs=1e-9)


def test_cycle_lp_matches_full_lp():
    for gamma in [.02,.08,.16]:
        p=diagnostic_chain(gamma,contexts=2)
        for required,target in [(p.required,'task'),(tuple(range(4)),'full')]:
            w,d,it=cycle_design(p.kernels,p.perms[7],required)
            _,expected=p.designs(target)
            assert d == pytest.approx(expected[7],rel=1e-7,abs=1e-9)
            assert np.sum(w)==pytest.approx(1)


def test_task_complexity_monotonicity():
    p=diagnostic_chain(.02,contexts=2)
    _,task=p.designs('task')
    _,full=p.designs('full')
    assert np.all(task >= full - 1e-9)
    assert task[0] > 20*full[0]


def test_anytime_set_uses_multiplicity_and_robust_correction():
    ll=np.array([0.,-7.,-1.])
    c=confidence_set(ll,.05)
    assert np.array_equal(c,[True,False,True])
    assert np.all(confidence_set(ll,.05,error_budget=2))


def test_certificate_is_one_common_policy_not_individual_optima():
    regrets=np.array([[0.,1.],[1.,0.]])
    pi,bound=policy_certificate(np.array([True,True]),regrets,.1)
    assert pi is None and bound==1.
    pi,bound=policy_certificate(np.array([True,False]),regrets,.1)
    assert pi==0 and bound==0
    pi,bound=policy_certificate(np.array([True,False]),regrets,.1,value_error=.1)
    assert pi is None


def test_all_actions_negative_control_is_identical():
    p=diagnostic_chain(.08,contexts=2,required=(0,1,2,3))
    assert np.array_equal(p.task_conflicts,p.full_conflicts)
    w,d=p.designs('task'); w2,d2=p.designs('full')
    assert np.allclose(w,w2) and np.allclose(d,d2)
    a,_=simulate(p,'task_task',repetitions=12,cap=300,seed=19)
    b,_=simulate(p,'full_full',repetitions=12,cap=300,seed=19)
    for x,y in zip(a,b):
        assert x['transitions']==y['transitions']
        assert x['estimated_mapping']==y['estimated_mapping']


def test_cap_is_not_a_certificate():
    p=diagnostic_chain()
    rows,_=simulate(p,'task_task',repetitions=5,cap=1)
    assert all(r['censored'] and not r['certified'] for r in rows)


def test_robust_set_inclusion_algebra():
    rng=np.random.default_rng(2)
    p=diagnostic_chain(.04,contexts=2)
    nominal=.95*p.kernels+.05/4
    eta=np.max(np.abs(np.log(p.kernels)-np.log(nominal)),axis=(1,2))
    q=rng.integers(p.Q,size=70); y=rng.integers(4,size=70)
    from wm_grounding.core import Problem
    ph=Problem(nominal)
    ll=p.logs[:,q,y].sum(axis=1); lh=ph.logs[:,q,y].sum(axis=1)
    for true in range(p.M):
        exact=ll.max()-ll[true]
        robust=lh.max()-lh[true]-2*eta[q//4].sum()
        assert robust <= exact+1e-10


def test_independent_bits_kl_decomposition():
    p=two_bit_distribution(.2,.5-.02); q=two_bit_distribution(.2,.5+.02)
    gamma=.02
    expected=2*gamma*np.log((.5+gamma)/(.5-gamma))
    assert categorical_kl(p,q)==pytest.approx(expected)
