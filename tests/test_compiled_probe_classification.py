import numpy as np
import pytest

from cognitive_ultrasound.compute_lab.engines import compiled_probe_checks


def test_b_difference_is_recorded_not_equivalent_or_quality_pass():
    expected=np.array([1.,2.],dtype=np.float32)
    actual=expected+.01
    r=compiled_probe_checks([actual],[expected],[actual.copy()],approximate=True)
    assert r['passed'] is False
    assert r['checks'][0]['passed'] is False
    assert 'quality unassessed' in r['disposition']
    with pytest.raises(AssertionError,match='unchanged'):
        compiled_probe_checks([actual],[expected],[actual.copy()],approximate=False)


@pytest.mark.parametrize('kind',['nan','shape','dtype','repeat'])
def test_b_does_not_bypass_runtime_failures(kind):
    a=np.array([1.,2.],dtype=np.float32);b=a.copy();repeat=a.copy()
    if kind=='nan':a[0]=np.nan
    elif kind=='shape':a=a[:1]
    elif kind=='dtype':a=a.astype(np.float64)
    else:repeat[0]+=1
    with pytest.raises((FloatingPointError,ValueError,RuntimeError)):
        compiled_probe_checks([a],[b],[repeat],approximate=True)


def test_exact_results_still_pass():
    a=np.array([1.,2.],dtype=np.float32)
    assert compiled_probe_checks([a],[a],[a],approximate=False)['passed']
