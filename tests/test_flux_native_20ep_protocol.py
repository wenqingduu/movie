"""Contract checks for independent native-FLUX assets and unchanged cohort/video/evaluation scope."""
import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'pretest'))
from prepare_entitybench_flux_native_20ep import prepare,read,sha256,write_json,frame_fingerprint,reusable_frame,assert_new_root,BASELINE,QWEN
from run_flux_native_20ep_images import reference_fingerprint,single_arguments


class FluxNativeProtocolTests(unittest.TestCase):
    def test_frozen_cohort_order_prompts_cuts_and_seeds_match_both_completed_routes(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);plan=prepare(root);baseline=read(BASELINE/'manifest.json');qwen=read(QWEN/'preparation_plan.json')
            self.assertEqual(plan['counts'],qwen['counts'])
            self.assertEqual(plan['video_config'],baseline['video_config'])
            for new,old,other in zip(plan['episodes'],baseline['episodes'],qwen['episodes']):
                self.assertEqual(new['episode_id'],old['episode_id']);self.assertEqual(new['episode_id'],other['episode_id'])
                self.assertEqual(len(new['shots']),len(old['shots']))
                for a,b in zip(new['shots'],old['shots']):
                    for key in ('seed','prompt','cut','shot_key','previous_shot_key'):self.assertEqual(a[key],b[key])
            self.assertEqual(sum(len(e['shots']) for e in plan['episodes']),363)
            self.assertEqual(plan['evaluation']['scope'],'partial_without_vlm');self.assertFalse(plan['evaluation']['vlm_enabled'])
            self.assertFalse(plan['firstframe_settings']['multi_reference']);self.assertEqual(plan['firstframe_settings']['inject_start'],30)

    def test_reference_roles_keep_qwen_prompts_and_seeds_but_all_paths_are_new(self):
        with tempfile.TemporaryDirectory() as directory:
            plan=prepare(Path(directory));old=read(QWEN/'preparation_plan.json')
            self.assertEqual(len(plan['roles']),112);self.assertEqual(sum(x['needs_face_asset'] for x in plan['roles']),40)
            for a,b in zip(plan['roles'],old['roles']):
                for k in ('character','episode_id','prompt','seed','needs_face_asset'):self.assertEqual(a[k],b[k])
                for k in ('reference','metadata','result'):
                    self.assertNotEqual(a[k],b[k]);self.assertTrue(Path(a[k]).is_relative_to(Path(directory)))

    def test_historic_outputs_and_nested_output_roots_are_rejected(self):
        for root in (BASELINE,QWEN,QWEN/'nested'):
            with self.assertRaises(AssertionError):assert_new_root(root)

    def test_frozen_prepare_rejects_changes_in_saved_protocol(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);plan=prepare(root);plan['reference_settings']['steps']=4;write_json(root/'preparation_plan.json',plan)
            with self.assertRaises(AssertionError):prepare(root)

    def test_changed_reference_source_or_wrong_gaussian_calibration_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);reference=root/'ref.png';reference.write_bytes(b'new native FLUX reference')
            meta=root/'meta.json';write_json(meta,dict(selected_sha256=sha256(reference),face_eligible=True))
            model=root/'gaussians.ply';model.write_bytes(b'new gaussian');cal=model.with_suffix('.pose_calibration.json')
            write_json(cal,dict(gaussian_model_sha256=sha256(model)))
            result=root/'facelift_result.json';write_json(result,dict(model_path=str(model),source_reference_sha256=sha256(reference),facelift_status='success'))
            role=dict(episode_id='episode',character='Person',reference=str(reference),metadata=str(meta),result=str(result),needs_face_asset=True)
            (root/'runtime').mkdir();write_json(root/'runtime/flux_implementation.json',dict(source='frozen'))
            plan=dict(roles=[role],firstframe_settings=dict(inject_start=30));ep=dict(episode_id='episode',episode_sha256='script')
            shot=dict(entity_schedule=dict(characters=['Person']),shot_key='1:1',seed=7,prompt='original')
            frame_fingerprint(root,plan,ep,shot)
            write_json(cal,dict(gaussian_model_sha256='foreign asset'))
            with self.assertRaises(AssertionError):frame_fingerprint(root,plan,ep,shot)
            write_json(cal,dict(gaussian_model_sha256=sha256(model)));reference.write_bytes(b'changed reference')
            with self.assertRaises(AssertionError):frame_fingerprint(root,plan,ep,shot)

    def test_resume_rejects_changed_media_or_dependency(self):
        with tempfile.TemporaryDirectory() as directory:
            image=Path(directory)/'final.png';image.write_bytes(b'formal output')
            record=dict(dependency_fingerprint='frozen',final_image=str(image),final_sha256=sha256(image))
            self.assertTrue(reusable_frame(record,'frozen'));self.assertFalse(reusable_frame(record,'changed'))
            image.write_bytes(b'tampered');self.assertFalse(reusable_frame(record,'frozen'))

    def test_reference_cache_depends_on_model_implementation_prompt_and_candidate_seed(self):
        role=dict(seed=1,prompt='official description');plan=dict(reference_settings=dict(steps=40));impl=dict(model_sha='frozen')
        first=reference_fingerprint(role,plan,impl)
        for changed in [dict(seed=2,prompt=role['prompt']),dict(role,prompt='different')]:self.assertNotEqual(reference_fingerprint(changed,plan,impl),first)
        self.assertNotEqual(reference_fingerprint(role,plan,dict(model_sha='changed')),first)


if __name__=='__main__':unittest.main()
