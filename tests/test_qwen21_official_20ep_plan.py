"""Check cohort/seed/reference routing without loading GPU models."""
import ast
import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'pretest'))
from prepare_entitybench_qwen21_20ep import prepare,BASELINE,PROJECT,immutable,reference_prompt,shot_prompt


class QwenPlanTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);self.baseline=json.loads((BASELINE/'manifest.json').read_text())

    def test_same_twenty_scripts_seeds_cuts_and_no_imported_frames(self):
        plan=prepare(self.root)
        self.assertEqual([e['episode_id'] for e in plan['episodes']],[e['episode_id'] for e in self.baseline['episodes']])
        for ep,prior in zip(plan['episodes'],self.baseline['episodes']):
            for shot,old in zip(ep['shots'],prior['shots']):
                for key in ['seed','shot_index','shot_key','prompt','cut','previous_shot_key','entity_schedule']:
                    self.assertEqual(shot[key],old[key])
                if shot['cut']:self.assertEqual(shot['firstframe_source']['source_kind'],'pending_qwen_hardcut')
        self.assertEqual(sum(len(e['shots']) for e in plan['episodes']),363)
        self.assertEqual(sum(s['cut'] for e in plan['episodes'] for s in e['shots']),191)
        self.assertEqual(prepare(self.root),plan)

    def test_only_needed_reference_roles_and_single_hardcut_3d(self):
        plan=prepare(self.root);self.assertEqual(len(plan['roles']),112)
        self.assertEqual(sum(r['needs_face_asset'] for r in plan['roles']),40)
        for ep in plan['episodes']:
            actual={r['character'] for r in plan['roles'] if r['episode_id']==ep['episode_id']}
            expected={n for s in ep['shots'] if s['cut'] for n in s['entity_schedule']['characters']}
            self.assertEqual(actual,expected)
        self.assertEqual(len({r['reference'] for r in plan['roles']}),112)
        self.assertEqual(len({r['seed'] for r in plan['roles']}),112)

    def test_freeze_rejects_mutation_without_rewriting(self):
        plan=prepare(self.root);path=self.root/'preparation_plan.json';original=path.read_bytes()
        changed=copy.deepcopy(plan);changed['firstframe_settings']['steps']=4
        with self.assertRaises(AssertionError):immutable(path,changed)
        self.assertEqual(path.read_bytes(),original)

    def test_original_age_species_and_no_extra_identity_metrics(self):
        prompt=reference_prompt('Dog','A Dalmatian pup',False,'animated story')
        self.assertIn('Dalmatian pup',prompt);self.assertNotIn('Single adult',prompt)
        self.assertEqual(shot_prompt('A landscape',[]),'A landscape')
        self.assertIn('Image 2',shot_prompt('Two characters',['A','B']))
        plan=prepare(self.root);self.assertFalse(plan['firstframe_settings']['evaluate_identity'])
        self.assertFalse(plan['firstframe_settings']['multi_face_injection'])
        source=(PROJECT/'pretest/run_entitybench_qwen21_20ep.py').read_text()
        ast.parse(source)
        self.assertIn("stages=('facelift','calibrate')",source)


if __name__=='__main__':unittest.main()
