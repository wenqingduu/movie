"""Cohort, immutable resume, whole-episode import and algorithm boundary checks."""
import ast
import copy
import json
import sys
import tempfile
import unittest
from unittest import mock
from pathlib import Path

PROJECT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PROJECT/'pretest'))
from prepare_entitybench_qwen21_full140 import prepare, SOURCE, read
from prepare_entitybench_qwen21_20ep import sha256, write_json
from run_lightx2v_a14b_4step_qwen140 import import_episode
from audit_entitybench_qwen21_full140 import audit_media
import run_entitybench_qwen21_full140 as route

def functions(name):
    return {n.name:ast.dump(n,include_attributes=False) for n in ast.parse((PROJECT/'pretest'/name).read_text()).body if isinstance(n,ast.FunctionDef)}

class Full140(unittest.TestCase):
    def test_gpu_device_mapping_and_storage_budget(self):
        with tempfile.TemporaryDirectory() as td:
            usage=mock.Mock(free=int(59.74*1024**3))
            nvml=mock.Mock(returncode=0,stdout='0, NVIDIA GeForce RTX 4090, 49140, 49139, 0\n')
            def fake_read(path,*args,**kwargs):
                return '2000000 100000' if str(path).endswith('cpu.max') else str(96*1024**3)
            with mock.patch.object(route.shutil,'disk_usage',return_value=usage),mock.patch.object(route.subprocess,'run',return_value=nvml),mock.patch.object(Path,'read_text',fake_read):
                blocked=route.resources(Path(td))
                self.assertEqual(blocked['reasons'],['startup_free_space_below_60GiB'])
                self.assertTrue(route.resources(Path(td),50)['ready'])
                with self.assertRaises(AssertionError):route.resources(Path(td),49)

    def test_original_generation_and_face_algorithms_unchanged(self):
        old=functions('run_lightx2v_a14b_4step_full140.py');new=functions('run_lightx2v_a14b_4step_qwen140.py')
        for key in ['fingerprint','reusable','decode_video','load_pipeline','import_episode']:
            self.assertEqual(old[key],new[key],key)
        old=functions('evaluate_entitybench_face_only_dino.py');new=functions('evaluate_entitybench_face_only_dino_qwen140.py')
        for key in ['intersection','area','iou','track_similarity','build_tracks','canonical_bbox','geometry','decode']:
            self.assertEqual(old[key],new[key],key)
        self.assertEqual(functions('run_qwen21_entitybench_20ep_images.py')['face_score'],
            functions('run_qwen21_entitybench_full140_images.py')['face_score'])

    def test_complete_plan_and_exact_resume(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);plan=prepare(root);self.assertEqual(prepare(root),plan)
            self.assertEqual((plan['counts']['shots'],plan['counts']['continuations']),(2491,1355))
            self.assertEqual((len(plan['roles']),sum(r['needs_face_asset'] for r in plan['roles'])),(711,226))
            original={e['episode_id']:e for e in read(SOURCE/'manifest.json')['episodes']}
            roles=read(SOURCE/'preparation_plan.json')
            self.assertEqual(plan['reference_settings'],roles['reference_settings'])
            self.assertEqual(plan['firstframe_settings'],roles['firstframe_settings'])
            for ep in plan['episodes']:
                if ep.get('reuse_generation_root'):
                    self.assertEqual(ep['shots'],original[ep['episode_id']]['shots'])
                else:
                    self.assertTrue(ep['shots'][0]['cut'])
                    self.assertEqual(len(ep['shots']),len({s['shot_key'] for s in ep['shots']}))
            plan['roles'][0]['seed']+=1;write_json(root/'preparation_plan.json',plan)
            with self.assertRaises(AssertionError):prepare(root)

    def test_readonly_whole_episode_import_and_parent_tamper(self):
        ep=copy.deepcopy(read(SOURCE/'manifest.json')['episodes'][0]);ep['reuse_generation_root']=str(SOURCE)
        runtime=read(SOURCE/'generation_manifest.json')
        origin_hash=sha256(SOURCE/'generation_report.json')
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);rows=import_episode(root,ep,runtime)
            self.assertEqual(rows,import_episode(root,ep,runtime))
            self.assertTrue(all(Path(r['output_video']).is_symlink() and Path(r['input_image']).is_symlink() for r in rows))
            write_json(root/'manifest.json',dict(episodes=[ep]))
            write_json(root/'generation_report.json',dict(status='complete',jobs=rows))
            n=sum(not s['cut'] for s in ep['shots'])
            self.assertEqual(audit_media(root,1,len(rows),n,decode=False)['continuations'],n)
            next(r for r in rows if not r['cut'])['parent_video_sha256']='changed'
            write_json(root/'generation_report.json',dict(status='complete',jobs=rows))
            with self.assertRaises(AssertionError):audit_media(root,1,len(rows),n,decode=False)
            bad=copy.deepcopy(runtime);bad['config']['sampling_steps']=40
            with self.assertRaises(AssertionError):import_episode(root,ep,bad)
        self.assertEqual(sha256(SOURCE/'generation_report.json'),origin_hash)

    def test_hardcut_adapter_skips_imported_episodes_and_sizes_from_plan(self):
        text=(PROJECT/'pretest/run_qwen21_entitybench_full140_images.py').read_text()
        self.assertIn("if ep.get('reuse_generation_root'):continue",text)
        self.assertIn("assert len(rows)==plan['counts']['new_hardcuts']",text)
        self.assertNotIn('total_hardcuts=191',text)

    def test_face_review_binds_actual_manifest_before_embedding(self):
        text=(PROJECT/'pretest/evaluate_entitybench_face_only_dino_qwen140.py').read_text()
        binding=text.index("validate_review(manifest,review,sha(OUT/'face_crop_manifest.json'))")
        embedding=text.index('mod.compute_dino_embedding')
        self.assertLess(binding,embedding)
        self.assertIn('independent_audit()',text)

    def test_every_face_crop_is_reviewed_and_wrong_manifest_is_rejected(self):
        tree=ast.parse((PROJECT/'pretest/evaluate_entitybench_face_only_dino_qwen140.py').read_text())
        function=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='validate_review')
        namespace={};exec(compile(ast.Module(body=[function],type_ignores=[]),'review_validation','exec'),namespace)
        validate=namespace['validate_review']
        manifest=dict(pages=[dict(page=1)],rows=[dict(id='face1',crop_path='1.png'),dict(id='face2',crop_path='2.png')])
        review=dict(face_crop_manifest_sha256='sha',reviewer='actual reviewer',reviewed_at='date',reviewed_pages=[1],
            reviewed_ids=['face1','face2'],rejected_ids=[],criteria='face-localization only; retain generated identity/style drift; no DINO-score filtering')
        validate(manifest,review,'sha')
        with self.assertRaises(AssertionError):validate(manifest,review,'different_sha')
        review['reviewed_ids']=['face1']
        with self.assertRaises(AssertionError):validate(manifest,review,'sha')

if __name__=='__main__':unittest.main()
