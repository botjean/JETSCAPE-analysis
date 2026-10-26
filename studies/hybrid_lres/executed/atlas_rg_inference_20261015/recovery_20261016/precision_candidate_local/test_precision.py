"""Local prototype tests: strict controls, propagated source bounds and guardrails."""
from decimal import Decimal, localcontext
import math
from pathlib import Path
import sys

import numpy as np
import pytest

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import _rg_precision_candidate as candidate


def run(rows,*,digits=6,single=True,bounds=(),scale=.0001):
    return candidate.reconstruct(np.asarray(rows).tolist(),-1,1000.,scale,'signed_prongs',True,
        digits,single,np.asarray(bounds).tolist())


@pytest.fixture
def rows():
    with np.load(HERE.parent/'diagnostic_aa510_chunk35/event_2025.npz') as z:return z['rows'].copy()


def test_new_case_requires_declared_decimal_precision(rows):
    for single in (False,True):
        with pytest.raises(RuntimeError,match='lost jet constituents'):run(rows,digits=0,single=single)
    before=rows.copy()
    result=run(rows)
    assert result['precision_boundary_interventions']==2
    assert all(d['all_constituents_retained'] for d in result['reclustering'])
    for decision in result['precision_decisions']:
        assert decision['accepted'] and decision['norm']>.0001
        assert decision['nominal_scale']==.0001 and decision['lower_norm']<.0001
    for d in result['reclustering']:
        np.testing.assert_allclose(d['jet_four_vector'],d['reclustered_four_vector'],atol=1e-10,rtol=1e-12)
    np.testing.assert_array_equal(rows,before)


@pytest.mark.parametrize('value',[.0001,.001,.1,1.,10.,1000000.,-.0001])
def test_decimal_decade_uses_source_token_not_float32_log10(value):
    v=float(np.float32(value))
    bounds=candidate.source_bounds([[v,0,0,2*abs(value)+1,1,1,0]],6,True)[0]
    exponent=int(f'{value:.5e}'.split('e')[1])
    assert bounds[0]>=.5*10**(exponent-5)


def test_precision_declarations_are_checked(rows):
    bad=rows.copy();bad[0,0]=float(np.float32(1.234567));bad[0,3]=10
    with pytest.raises(ValueError,match='decimal precision'):run(bad)
    bad=rows.copy();bad[0,0]=1.2345678912345;bad[0,3]=10
    with pytest.raises(ValueError,match='float32 storage'):run(bad)
    with pytest.raises(ValueError,match='six-digit'):run(rows,digits=5)
    with pytest.raises(ValueError,match='all rows'):run(rows,bounds=np.zeros((len(rows),3)))
    with pytest.raises(ValueError,match='all rows'):run(rows,digits=0,single=False,bounds=np.zeros((2,3)))
    bad=np.zeros((len(rows),3));bad[0,0]=-1
    with pytest.raises(ValueError,match='Invalid transported'):run(rows,digits=0,single=False,bounds=bad)


@pytest.mark.parametrize('overshoot,accepted',[(1e-10,True),(5e-10,False)])
def test_inside_and_outside_source_interval(overshoot,accepted):
    rows=[[1,.0001+overshoot,0,2,1,1,0],[1,0,0,2,-1,1,0]]
    result=candidate._test_combine(rows,[[0,1]],input_bounds=[[0,1e-10,0],[0,1e-10,0]])
    decision=result['decisions'][0]
    assert decision['accepted'] is accepted
    assert result['geometry_interventions']==int(accepted)
    assert result['nodes'][0]['p4'][3]==0


def test_multilevel_parents_keep_sum_of_original_leaf_bounds():
    delta=.0001+3e-10
    rows=[[1,0,0,2,1,1,0],[0,1,0,2,1,1,0],
          [1,0,0,2,-1,1,0],[0,1-delta,0,2,-1,1,0]]
    bounds=np.tile([1e-8,1e-10,1e-8],(4,1))
    result=candidate._test_combine(rows,[[0,1],[2,3],[4,5]],input_bounds=bounds.tolist())
    last=result['nodes'][-1]
    assert last['leaf_count']==4
    assert 4e-10<=last['radius'][1]<4.00001e-10
    assert result['precision_boundary_interventions']==1
    assert result['decisions'][-1]['accepted']


@pytest.mark.parametrize('seed',range(20))
def test_propagated_bounds_contain_high_precision_signed_sum(seed):
    rng=np.random.default_rng(seed)
    truth=rng.uniform(-2,2,(12,3))
    stored=np.array([[np.float32(f'{v:.6g}') for v in row] for row in truth],dtype=float)
    signs=rng.choice([-1,1],12)
    rows=np.c_[stored,np.full(12,100.),signs,np.ones(12),np.zeros(12)]
    plan=[[0,1]]+[[12+i-2,i] for i in range(2,12)]
    result=candidate._test_combine(rows.tolist(),plan,6,True)
    last=result['nodes'][-1]
    assert last['leaf_count']==12
    with localcontext() as ctx:
        ctx.prec=80
        for j in range(3):
            exact=sum((Decimal.from_float(float(v))*int(s) for v,s in zip(truth[:,j],signs)),Decimal(0))
            observed=Decimal.from_float(float(last['p4'][j]))*last['sign']
            assert abs(observed-exact)<=Decimal.from_float(float(last['radius'][j]))


@pytest.mark.parametrize('mode',['same_sign','nonzero_energy','finite_direction'])
def test_non_norm_guards_are_unchanged(mode):
    rows=[[1,.0001000002,0,2,1,1,0],[1,0,0,2,-1,1,0]]
    if mode=='same_sign':rows[1][4]=1
    elif mode=='nonzero_energy':rows[1][3]-=1e-8
    else:rows=[[1,0,0,2,1,1,0],[1,0,0,2,-1,1,0]]
    result=candidate._test_combine(rows,[[0,1]],input_bounds=np.full((2,3),1e-8).tolist())
    assert not result['decisions'] and result['geometry_interventions']==0


def test_disabled_geometry_stays_disabled(rows):
    with pytest.raises(RuntimeError,match='lost jet constituents'):run(rows,scale=0)


def test_large_spacelike_case_stays_rejected():
    def row(pt,phi=0,eta=0,mass=0,sign=1):
        values=[pt*math.cos(phi),pt*math.sin(phi),pt*math.sinh(eta)]
        values=[float(np.float32(f'{v:.6g}')) for v in values]
        return values+[float(np.float32(math.sqrt((pt*math.cosh(eta))**2+mass**2))),sign,1,0]
    mass=15*math.sqrt(math.cosh(.25)**2-math.cosh(.2)**2)
    rows=[row(200),row(15,.1,.2,mass),row(15,.1,.25,sign=-1)]
    with pytest.raises(RuntimeError,match='lost jet constituents'):run(rows)


@pytest.mark.parametrize('angle',[.1,1.,2.8,3.14])
def test_rotated_event_transports_bounds_instead_of_redeclaring_source(rows,angle):
    matrix=np.array([[math.cos(angle),-math.sin(angle),0],[math.sin(angle),math.cos(angle),0],[0,0,1]])
    bounds=np.array(candidate.source_bounds(rows.tolist(),6,True))
    rotated=rows.copy();rotated[:,:3]=rows[:,:3]@matrix.T
    transported=bounds@abs(matrix).T+8*np.finfo(float).eps*(abs(rows[:,:3])@abs(matrix).T)
    actual=run(rotated,digits=0,single=False,bounds=transported)
    np.testing.assert_allclose(actual['jets'],run(rows)['jets'],atol=1e-10,rtol=1e-9)


@pytest.mark.parametrize('seed',[1,21,38])
def test_particle_reordering(rows,seed):
    shuffled=np.random.default_rng(seed).permutation(rows)
    np.testing.assert_allclose(run(shuffled)['jets'],run(rows)['jets'],atol=1e-10,rtol=1e-9)


def test_bound_addition_rounds_outward_at_each_operation():
    rows=[[1,0,0,2,1,1,0],[1,0,0,2,1,1,0]]
    result=candidate._test_combine(rows,[[0,1]],input_bounds=[[1,0,0],[2**-65,0,0]])
    with localcontext() as ctx:
        ctx.prec=80
        assert Decimal.from_float(result['nodes'][0]['radius'][0])>=Decimal(1)+Decimal(2)**-65


def test_zero_includes_both_serialization_stages():
    bound=candidate.source_bounds([[0,0,0,1,1,1,0]],6,True)[0][0]
    assert Decimal.from_float(bound)>=Decimal('7.006494e-46')


@pytest.mark.parametrize('seed',range(10))
def test_subnormal_source_rounding_bound(seed):
    truth=np.random.default_rng(seed).uniform(-1e-44,1e-44,3)
    stored=np.array([np.float32(f'{x:.6g}') for x in truth],dtype=float)
    bound=candidate.source_bounds([[*stored,1,1,1,0]],6,True)[0]
    for exact,observed,error in zip(truth,stored,bound):
        assert abs(Decimal.from_float(float(exact))-Decimal.from_float(float(observed)))<=Decimal.from_float(error)


def test_ambiguous_extra_boundary_is_rejected():
    scale=.0001
    result=candidate._test_norm([np.nextafter(scale,np.inf),0,0],[np.spacing(scale),0,0],scale)
    assert result['ambiguous_rejected'] and not result['accepted']


@pytest.mark.parametrize('seed',range(20))
def test_norm_interval_encloses_exact_componentwise_distance(seed):
    rng=np.random.default_rng(seed)
    value=rng.uniform(-.001,.001,3)
    bound=rng.uniform(0,1e-5,3)
    result=candidate._test_norm(value.tolist(),bound.tolist(),.0001)
    with localcontext() as ctx:
        ctx.prec=80
        exact=sum((max(Decimal(0),abs(Decimal.from_float(float(x)))-Decimal.from_float(float(u)))**2
            for x,u in zip(value,bound)),Decimal(0))
        assert Decimal.from_float(result['lower_squared'])<=exact<=Decimal.from_float(result['upper_squared'])
        if result['accepted']:
            assert exact<=Decimal.from_float(.0001)**2


def test_transported_bounds_reject_non_nearest_rounding():
    import ctypes
    lib=ctypes.CDLL('libm.so.6')
    old=lib.fegetround()
    assert lib.fesetround(0x800)==0
    try:
        with pytest.raises(ValueError,match='round-to-nearest'):
            candidate._test_combine([[1,0,0,2,1,1,0],[1,0,0,2,1,1,0]],[[0,1]],input_bounds=[[0,0,0],[0,0,0]])
    finally:
        assert lib.fesetround(old)==0
