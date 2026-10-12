"""Regression checks for complete-episode reuse and full-cohort expansion; no GPU load."""
import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'pretest'))
from prepare_entitybench_a14b_full140 import prepare, SOURCE, freeze_face_routing, injectable_hardcut, NON_FACE_ROLES
from run_lightx2v_a14b_4step_full140 import import_episode
from prepare_entitybench_pulid_pairs import _ordered_shots


class FullCohortTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source_manifest = json.loads((SOURCE / 'manifest.json').read_text())
        self.runtime = json.loads((SOURCE / 'generation_manifest.json').read_text())
        self.episode = copy.deepcopy(self.source_manifest['episodes'][0])
        self.episode['reuse_generation_root'] = str(SOURCE)

    def test_full_plan_covers_all_and_calibrates_only_single_hardcut_roles(self):
        plan = prepare(self.root)
        self.assertEqual(len(plan['episodes']), 140)
        self.assertEqual(plan['counts']['shots'], 2491)
        self.assertEqual(plan['counts']['new_shots'], 2128)
        self.assertEqual(plan['counts']['new_single_hardcut'], 352)
        self.assertEqual(plan['counts']['new_basic_hardcut'], 593)
        self.assertEqual(len(plan['roles']), 226)
        self.assertEqual(sum(bool(e.get('reuse_generation_root')) for e in plan['episodes']), 20)
        self.assertEqual(prepare(self.root), plan)

    def test_import_reuses_media_and_preserves_original_dependency_fingerprints(self):
        imported = import_episode(self.root, self.episode, self.runtime)
        originals = {r['shot_key']: r for r in json.loads((SOURCE / 'generation_report.json').read_text())['jobs']
                     if r['episode_id'] == self.episode['episode_id']}
        self.assertEqual(len(imported), len(self.episode['shots']))
        for row in imported:
            self.assertTrue(Path(row['output_video']).is_symlink())
            self.assertEqual(row['video_sha256'], originals[row['shot_key']]['video_sha256'])
            self.assertEqual(row['dependency_fingerprint'], originals[row['shot_key']]['dependency_fingerprint'])

    def test_non_face_routing_preserves_seeds_and_all_shots(self):
        plan = prepare(self.root)
        original = (self.root / 'expansion_plan.json').read_bytes()
        routing = freeze_face_routing(self.root, plan)
        self.assertEqual(routing['counts']['injection_roles'], 217)
        self.assertEqual(routing['counts']['basic_only_roles'], 9)
        excluded = {(r['episode_id'], r['character']) for r in routing['roles'] if not r['eligible']}
        self.assertEqual(excluded, set(NON_FACE_ROLES))
        injected, basic = [], []
        for ep in plan['episodes']:
            if ep.get('reuse_generation_root'):
                continue
            for shot in ep['shots']:
                if shot['cut']:
                    (injected if injectable_hardcut(ep, shot, routing) else basic).append(shot)
        self.assertEqual(len(injected) + len(basic), 945)
        self.assertEqual(len(injected), 352-routing['counts']['basic_only_single_hardcuts'])
        self.assertEqual((self.root / 'expansion_plan.json').read_bytes(), original)
        self.assertEqual(freeze_face_routing(self.root, plan), routing)
        for role in plan['roles']:
            self.assertEqual(role['seed'], next(r['seed'] for r in json.loads(original)['roles']
                if r['episode_id']==role['episode_id'] and r['character']==role['character']))

    def test_changed_face_eligibility_is_rejected(self):
        plan = prepare(self.root)
        routing = freeze_face_routing(self.root, plan)
        routing['roles'][0]['eligible'] = False
        (self.root / 'face_injection_routing.json').write_text(json.dumps(routing))
        with self.assertRaises(AssertionError):
            freeze_face_routing(self.root, plan)

    def test_incompatible_backend_cannot_reuse_existing_twenty(self):
        changed = copy.deepcopy(self.runtime)
        changed['config']['sampling_steps'] = 40
        with self.assertRaises(AssertionError):
            import_episode(self.root, self.episode, changed)

    def test_changed_seed_cannot_reuse_same_episode_video(self):
        self.episode['shots'][0]['seed'] += 1
        with self.assertRaises(AssertionError):
            import_episode(self.root, self.episode, self.runtime)

    def test_missing_or_changed_parent_is_rejected_without_touching_sources(self):
        fixture = self.root / 'source'
        fixture.mkdir()
        for name in ['manifest.json', 'generation_manifest.json']:
            (fixture / name).write_bytes((SOURCE / name).read_bytes())
        report = json.loads((SOURCE / 'generation_report.json').read_text())
        row = next(r for r in report['jobs'] if r['episode_id'] == self.episode['episode_id'] and not r['cut'])
        row['parent_video_sha256'] = 'changed-parent'
        (fixture / 'generation_report.json').write_text(json.dumps(report))
        self.episode['reuse_generation_root'] = str(fixture)
        with self.assertRaises(AssertionError):
            import_episode(self.root / 'destination', self.episode, self.runtime)

    def test_hardcut_selection_retains_original_global_seed_indices(self):
        episode = dict(scenes=[dict(scene_num=5, video_prompts=['A', 'B'], cut=[True, False]),
                               dict(scene_num=9, video_prompts=['C'], cut=[True])], entity_schedule={})
        full = _ordered_shots(episode)
        hardcuts = [s for s in full if s['cut']]
        self.assertEqual([s['shot_index'] for s in hardcuts], [0, 2])
        self.assertEqual([s['shot_key'] for s in hardcuts], ['5:1', '9:1'])


if __name__ == '__main__':
    unittest.main()
