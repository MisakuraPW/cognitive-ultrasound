import importlib.util
from pathlib import Path

import pytest

from cognitive_ultrasound.preparation.common import atomic_json, read_json
from cognitive_ultrasound.provenance import sha256

spec=importlib.util.spec_from_file_location("gap_export",Path(__file__).parents[1]/"scripts/export_compute_lab_gap_audit.py")
module=importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


@pytest.mark.parametrize("outcome",["completed","failed"])
def test_addendum_preserves_failures_and_does_not_adopt(tmp_path,monkeypatch,outcome):
    root=tmp_path/'batch';root.mkdir()
    repair=root/'gap_repair';repair.mkdir()
    atomic_json(repair/'status.json',dict(status="completed" if outcome=="completed" else "completed_with_rejections"))
    base=tmp_path/'base.tar.gz';base.write_bytes(b'original archive')
    atomic_json(tmp_path/'batch.bundle.json',dict(path=str(base),sha256=sha256(base)))
    plan=dict(jobs=['short_torch_50_fp16_b14'],original_results={})
    for name,key in [('identity.json','original_identity_sha256'),('selection.json','original_selection_sha256'),('config.json','scientific_config_sha256')]:
        atomic_json(root/name,{});plan[key]=sha256(root/name)
    old=root/'jobs/short_torch_50_fp16_b14/result.json'
    atomic_json(old,dict(status='failed',error='historical'))
    old_bytes=old.read_bytes();plan['original_results'][old.parent.name]=sha256(old)
    atomic_json(repair/'plan.json',plan)
    row=dict(case='a',seed=42,frame=2,psnr=20,ssim=.8,mae=.1,cold=False,core_s=.1,closed_loop_s=.2)
    atomic_json(root/'jobs/short_official_b14/result.json',dict(rows=[row]))
    atomic_json(repair/'jobs'/old.parent.name/'result.json',dict(status=outcome,error=None if outcome=='completed' else 'AssertionError: mismatch',rows=[row] if outcome=='completed' else [],operator_checks=dict(internal_correctness=outcome=='completed'),equivalence_passed=False))
    v=dict(recovery=dict(passed=True,bitwise=True),recovery_loss=dict(passed=True,bitwise=True),elapsed_s=1)
    atomic_json(root/'training_report.json',dict(casl=dict(eager=v,graph=v,engineering_comparison=dict(passed=False),engineering_loss=dict(passed=False),recommended='eager')))
    (root/'ACCELERATION_SUMMARY.md').write_text('Historical summary')
    monkeypatch.setattr(module,'archive',lambda folder,destination,**k:dict(path=str(destination),sha256='newhash'))
    module.finalize(root)
    final=read_json(tmp_path/'batch.final_export.json')
    assert final['automatic_adoption'] is False
    assert old.read_bytes()==old_bytes
    assert final['failed_after_retry']==([] if outcome=='completed' else [old.parent.name])
    assert final['status']==('completed' if outcome=='completed' else 'completed_with_rejected_candidates')
    assert read_json(repair/'repair_matrix.json')[0]['cross_official_equivalent'] is False
    assert (root/'analysis_delivery/FINAL_REPORT.md').exists()
    assert not list((root/'analysis_delivery').rglob('*.npz'))
    assert not list((root/'analysis_delivery').rglob('*.tar.gz'))
    assert 'analysis' in final


def test_running_cannot_be_exported_as_finished(tmp_path):
    atomic_json(tmp_path/'gap_repair/status.json',dict(status='running'))
    with pytest.raises(RuntimeError,match='not terminal'):
        module.finalize(tmp_path)
