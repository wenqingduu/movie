"""Run selected20 scoring; normalize status metadata without touching frozen scorer/plan."""
from pathlib import Path
import qwen140_original40_selected_gain_20261109 as job

def status(stage,**kwargs):
    data=dict(stage=stage,utc=job.trial.core.now(),api_calls=0,vlm_calls=0,media_generation=False,
              quality_gate_enabled=False,selection_rule='dynamic increases vs frozen current benchmark only',old_media_overwritten=False)
    data.update(kwargs)
    job.write(job.OUT/'status.json',data)

if __name__=='__main__':
    job.status=status
    job.write(job.OUT/'runtime_adapter.json',dict(path=str(Path(__file__).resolve()),sha256=job.sha(__file__),
        frozen_preparation_helper_sha256=job.sha(job.__file__),
        purpose='Status metadata merge only: avoid duplicate quality_gate_enabled keyword; no scoring, selection, media or manifest change.'))
    job.run()
