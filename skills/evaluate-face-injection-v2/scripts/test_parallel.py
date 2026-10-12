"""Read-only regression checks against the completed serial cohort; no GPU inference."""
import importlib.util,json,tempfile,shutil,sys,unittest
from pathlib import Path
skill=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('parallel_helper',skill/'scripts/evaluate_parallel.py'); h=importlib.util.module_from_spec(spec); spec.loader.exec_module(h)
root=h.DEFAULT_ROOT; episodes,split,inputs=h.freeze_inputs(root); serial=root/'evaluation/official'; shards,loads=h.partition(episodes,split,2)
class WorkflowTests(unittest.TestCase):
 def setUp(self):
  self.temp=tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup); self.temp_root=Path(self.temp.name)
 def source_shards(self):
  assignments=[]
  for i,items in enumerate(shards):
   source=self.temp_root/f'worker_{i}'; source.mkdir(); shutil.copy2(serial/'run_manifest.json',source/'run_manifest.json')
   for eid,tier in items: shutil.copy2(serial/f'{eid}.json',source/f'{eid}.json')
   assignments.append((source,items))
  return assignments
 def test_shards_are_disjoint_complete_and_balanced(self):
  ids=[eid for shard in shards for eid,tier in shard]
  self.assertEqual(len(ids),len(set(ids))); self.assertEqual(set(ids),set(episodes)); self.assertEqual(sum(loads),363); self.assertLessEqual(abs(loads[0]-loads[1]),1)
 def test_two_shard_aggregation_matches_original_official_report(self):
  actual=h.aggregate(root,self.temp_root/'merged',episodes,split,inputs,self.source_shards(),'validation')
  expected=h.read(serial/'report_pulid_a14b_4step_continuation_20ep_no_vlm.json')
  for tier in ['all','easy','medium','hard']: self.assertEqual(actual[tier],expected[tier])
 def test_incomplete_coverage_is_rejected(self):
  assigned=self.source_shards()
  with self.assertRaises(AssertionError): h.aggregate(root,self.temp_root/'merged',episodes,split,inputs,assigned[:1],'invalid')
 def test_duplicate_episode_is_rejected(self):
  assigned=self.source_shards()
  with self.assertRaises(AssertionError): h.aggregate(root,self.temp_root/'merged',episodes,split,inputs,assigned+[assigned[0]],'invalid')
 def test_changed_config_is_rejected(self):
  assigned=self.source_shards(); p=assigned[1][0]/'run_manifest.json'; m=h.read(p); m['config']['clip_match_threshold']+=.01; h.write(p,m)
  with self.assertRaises(AssertionError): h.aggregate(root,self.temp_root/'merged',episodes,split,inputs,assigned,'invalid')
 def test_recorded_vlm_call_is_rejected(self):
  assigned=self.source_shards(); p=assigned[0][0]/'run_manifest.json'; m=h.read(p); m['vlm_call_stats']['total_calls']=1; h.write(p,m)
  with self.assertRaises(AssertionError): h.aggregate(root,self.temp_root/'merged',episodes,split,inputs,assigned,'invalid')
 def test_changed_video_record_is_rejected(self):
  fixture=self.temp_root/'run'; fixture.mkdir()
  for name in ['manifest.json','split_20ep.json']: shutil.copy2(root/name,fixture/name)
  (fixture/'runtime').symlink_to(root/'runtime',target_is_directory=True); (fixture/'official_results').symlink_to(root/'official_results',target_is_directory=True)
  g=h.read(root/'generation_report.json'); g['jobs'][0]['video_sha256']='0'*64; h.write(fixture/'generation_report.json',g)
  with self.assertRaises(AssertionError): h.freeze_inputs(fixture)
 def test_environment_drops_api_credentials_and_limits_threads(self):
  env=h.environment(4,0); self.assertEqual(env['OMP_NUM_THREADS'],'4'); self.assertEqual(env['CUDA_VISIBLE_DEVICES'],'0'); self.assertFalse(any('OPENAI' in k or 'GEMINI' in k or 'AZURE' in k for k in env))
 def test_worker_commands_keep_official_non_vlm_contract(self):
  cmd=h.command(root,self.temp_root/'.shards/worker_0','test'); self.assertIn('--skip_vlm',cmd); self.assertIn('--resume',cmd); self.assertNotIn('--no_strict',cmd); self.assertNotIn('--n_frames',cmd); self.assertEqual(cmd[cmd.index('--device')+1],'cuda:0')
unittest.main(verbosity=2)
