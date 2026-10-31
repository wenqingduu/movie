"""CPU-only coverage for supervision, safe repair and frozen input provenance."""
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'pretest'))
import qwen140_overnight_watchdog as watchdog


class DeadlineAndProcessesTest(unittest.TestCase):
    def test_deadline_uses_absolute_timezone_and_includes_noon(self):
        deadline = '2026-10-30T12:00:00+08:00'
        self.assertFalse(watchdog.deadline_passed(deadline, datetime(2026, 10, 30, 3, 59, tzinfo=timezone.utc)))
        self.assertTrue(watchdog.deadline_passed(deadline, datetime(2026, 10, 30, 4, 0, tzinfo=timezone.utc)))
        with self.assertRaises(ValueError):
            watchdog.parse_time('2026-10-30T12:00:00')

    def test_only_actual_python_processes_and_no_zombies(self):
        with tempfile.TemporaryDirectory() as directory:
            proc = Path(directory)
            for pid, executable, state in [(11, '/bin/bash', 'S'), (12, '/venv/bin/python', 'S'), (13, '/bin/python3', 'Z')]:
                folder = proc / str(pid)
                folder.mkdir()
                fields = [state, '1'] + ['0'] * 17 + ['12345']
                (folder / 'stat').write_text(str(pid) + ' (name) ' + ' '.join(fields))
                (folder / 'cmdline').write_bytes((executable + '\0-u\0pretest/run.py\0').encode())
            self.assertEqual(set(watchdog.process_table(proc)), {12})

    def test_driver_matching_and_reparented_children_use_start_tokens(self):
        command = ['/venv/bin/python', '-u', 'pretest/batch.py', 'run']
        self.assertTrue(watchdog.command_matches(['/bin/python3', 'pretest/batch.py', 'run'], command))
        self.assertFalse(watchdog.command_matches(['/bin/python3', 'pretest/other.py', 'batch.py run'], command))
        rows = {2: {'ppid': 1, 'start': '20'}, 3: {'ppid': 2, 'start': '30'}}
        self.assertEqual(watchdog.discover_children(rows, {2: '20'}), {2: '20', 3: '30'})
        self.assertEqual(watchdog.tracked_alive({3: {'start': '30', 'ppid': 1}}, {2: '20', 3: '30'}), {3: '30'})
        self.assertEqual(watchdog.tracked_alive(rows, {2: 'stale_pid'}), {})

    def test_deferred_episodes_are_terminal_without_claiming_complete(self):
        with tempfile.TemporaryDirectory() as directory:
            stage = {'root': directory}
            watchdog.write(Path(directory) / 'status.json', {'status': 'finished_with_deferred_episodes'})
            self.assertTrue(watchdog.terminal_state(stage)[0])
            self.assertFalse(watchdog.stage_finished(stage))


class TechnicalRepairTest(unittest.TestCase):
    BEFORE = "def compile_record(data):\n    assert isinstance(data, dict)\n    return data['name']\n"
    AFTER = "def compile_record(data):\n    assert isinstance(data, dict)\n    return data.get('name', '')\n"

    def test_guards_reject_prompt_threshold_assert_and_unlisted_function_changes(self):
        path = Path('compile.py')
        self.assertEqual(watchdog.validate_patch(path, self.BEFORE, self.AFTER, ['compile_record']), ['compile_record'])
        with self.assertRaises(ValueError):
            watchdog.validate_patch(path, self.BEFORE, self.AFTER, ['another'])
        with self.assertRaises(ValueError):
            watchdog.validate_patch(path, self.BEFORE, self.AFTER.replace('    assert isinstance(data, dict)\n', ''), ['compile_record'])
        before = 'def f():\n    return {"model": "qwen", "threshold": 0.4}\n'
        for after in [before.replace('qwen', 'other'), before.replace('0.4', '0.1')]:
            with self.assertRaises(ValueError):
                watchdog.validate_patch(path, before, after, ['f'])
        with self.assertRaises(ValueError):
            watchdog.validate_patch(Path('evaluation.py'), self.BEFORE, self.AFTER, ['compile_record'])

    def test_fake_model_reads_stages_tests_and_updates_versioned_policy(self):
        with tempfile.TemporaryDirectory() as temporary, patch.object(watchdog, 'ROOT', Path(temporary)):
            root = Path(temporary)
            source = root / 'pretest/compat.py'
            source.parent.mkdir()
            source.write_text(self.BEFORE)
            previous = watchdog.sha(source)
            cohort_root = root / 'outputs/cohort'
            experiment = cohort_root / 'experiments/run1'
            shot = experiment / 'episodes/ep/shot_1_1'
            watchdog.write(shot / 'result.json', {'video': str(shot / 'video.mp4')})
            (shot / 'video.mp4').write_bytes(b'preserve-media')
            result_before = watchdog.sha(shot / 'result.json')
            frozen = cohort_root / 'runtime/frozen_source/pretest/compat.py'
            frozen.parent.mkdir(parents=True)
            frozen.write_text(self.BEFORE)
            policy = experiment / 'face_injection_policy.json'
            watchdog.write(policy, {'references': {'role': {'sha256': 'identity'}},
                'frozen_files': [{'path': str(source), 'sha256': previous}]})
            policy_before = watchdog.sha(policy)
            cohort = cohort_root / 'cohort.json'
            watchdog.write(cohort, {'frozen_code': [{'path': str(source), 'sha256': previous}],
                'episodes': [{'output_root': str(experiment)}]})
            stage = {'root': str(cohort_root), 'name': 'example'}
            cfg = {'stages': [stage], 'repair_allowed_sources': {str(source): ['compile_record']}}
            def call(name, args, identifier):
                return {'role': 'assistant', 'content': None, 'tool_calls': [{'id': identifier, 'type': 'function',
                    'function': {'name': name, 'arguments': json.dumps(args)}}]}
            responses = iter([call('read_file', {'path': str(source)}, '1'),
                call('replace_text', {'path': str(source), 'old': "return data['name']", 'new': "return data.get('name', '')"}, '2'),
                call('finish', {'action': 'patch', 'reason': 'Optional missing field compatibility'}, '3')])
            agent = watchdog.RepairAgent(cfg, stage, root / 'repairs', responder=lambda *_: next(responses))
            decision = agent.run({'error': 'KeyError(name)'})
            self.assertEqual(decision['action'], 'patch')
            self.assertEqual(source.read_text(), self.BEFORE, 'Proposals must not mutate source before tests')
            revision = watchdog.apply_and_test(agent, cfg)
            self.assertEqual(source.read_text(), self.AFTER)
            self.assertEqual(watchdog.read(cohort)['frozen_code'][0]['sha256'], watchdog.sha(source))
            self.assertEqual(watchdog.sha(policy), policy_before, 'Original policy must remain immutable')
            pointer = watchdog.read(experiment / 'active_face_injection_policy.json')
            current = watchdog.read(pointer['path'])
            self.assertEqual(current['references'], {'role': {'sha256': 'identity'}})
            self.assertEqual(current['previous_policy_sha256'], policy_before)
            self.assertEqual(watchdog.sha(shot / 'result.json'), result_before)
            self.assertEqual((shot / 'video.mp4').read_bytes(), b'preserve-media')
            self.assertTrue((cohort_root / 'runtime/revisions' / revision / 'cohort_before.json').exists())
            with self.assertRaises(ValueError):
                agent.read_file(root / '.env.portrait')
            with self.assertRaises(ValueError):
                agent.replace_text(str(root / 'other.py'), 'a', 'b')

    def test_source_head_ranges_and_all_tool_calls_receive_responses(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / 'source.py'
            source.write_text('IMPORT_HEADER\n' + ('# padding\n' * 3500) + 'FINAL_LINE\n')
            cfg = {'repair_allowed_sources': {str(source): ['f']}}
            messages_seen = []
            def responder(messages, tools):
                messages_seen.append(list(messages))
                if len(messages_seen) == 1:
                    return {'role': 'assistant', 'content': None, 'tool_calls': [
                        {'id': str(i), 'type': 'function', 'function': {'name': 'read_file',
                         'arguments': json.dumps({'path': str(source)})}} for i in range(5)]}
                return {'role': 'assistant', 'content': None, 'tool_calls': [
                    {'id': 'finish', 'type': 'function', 'function': {'name': 'finish',
                     'arguments': json.dumps({'action': 'skip', 'reason': 'Diagnostic complete'})}}]}
            agent = watchdog.RepairAgent(cfg, {'root': directory}, Path(directory) / 'repair', responder=responder)
            self.assertTrue(agent.read_file(source).startswith('IMPORT_HEADER'))
            self.assertIn('truncated', agent.read_file(source))
            self.assertIn('3502: FINAL_LINE', agent.read_file(source, start_line=3502, max_lines=1))
            self.assertEqual(agent.run({})['action'], 'skip')
            replies = [m for m in messages_seen[1] if m['role'] == 'tool']
            self.assertEqual([m['tool_call_id'] for m in replies], [str(i) for i in range(5)])
            self.assertIn('budget', replies[-1]['content'])

    def test_failed_validation_restores_original_source(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(watchdog, 'ROOT', Path(directory)):
            source = Path(directory) / 'compat.py'
            source.write_text(self.BEFORE)
            cfg = {'stages': [], 'repair_allowed_sources': {str(source): ['compile_record']},
                   'repair_tests': [[sys.executable, '-c', 'raise RuntimeError("regression")']]}
            agent = watchdog.RepairAgent(cfg, {'root': directory}, Path(directory) / 'repair')
            agent.directory.mkdir()
            agent.staged[str(source)] = self.AFTER
            with self.assertRaises(RuntimeError):
                watchdog.apply_and_test(agent, cfg)
            self.assertEqual(source.read_text(), self.BEFORE)


class SupervisorTest(unittest.TestCase):
    def config(self, directory):
        return {'output_root': str(Path(directory) / 'supervisor'), 'poll_seconds': 0,
                'deadline_utc': '2099-01-01T00:00:00Z', 'max_repair_rounds': 2,
                'stages': [{'name': 'focus', 'root': str(Path(directory) / 'focus'),
                            'command': ['/bin/python', 'pretest/fake_driver.py', 'run']}]}

    def test_repeated_failure_bounded_to_two_repair_calls_and_recorded(self):
        with tempfile.TemporaryDirectory() as directory:
            cfg = self.config(directory)
            process = Mock(pid=999, returncode=1)
            failed = {'status': 'batch_finished_with_recorded_exceptions'}
            with patch.object(watchdog, 'process_table', return_value={}), \
                 patch.object(watchdog.Supervisor, 'monitor', return_value=(True, failed)), \
                 patch.object(watchdog.Supervisor, 'context', return_value={}), \
                 patch.object(watchdog.RepairAgent, 'run', return_value={'action': 'retry', 'reason': 'Transient network'}) as repair, \
                 patch.object(watchdog.subprocess, 'Popen', return_value=process) as launch:
                supervisor = watchdog.Supervisor(cfg)
                supervisor.run()
                self.assertEqual(repair.call_count, 2)
                self.assertEqual(launch.call_count, 3)
                self.assertTrue(supervisor.state['stages']['focus']['skipped_after_repair_budget'])
                self.assertEqual(supervisor.state['status'], 'finished_with_recorded_exceptions')

    def test_adopts_active_driver_without_duplicate_launch_or_repair(self):
        with tempfile.TemporaryDirectory() as directory:
            cfg = self.config(directory)
            rows = {123: {'argv': cfg['stages'][0]['command'], 'ppid': 1, 'start': '456'}}
            with patch.object(watchdog, 'process_table', return_value=rows), \
                 patch.object(watchdog.Supervisor, 'monitor', return_value=(True, {'status': 'complete'})), \
                 patch.object(watchdog.subprocess, 'Popen') as launch, \
                 patch.object(watchdog.RepairAgent, 'run') as repair:
                supervisor = watchdog.Supervisor(cfg)
                supervisor.run()
                launch.assert_not_called()
                repair.assert_not_called()
                self.assertEqual(supervisor.state['status'], 'complete')

    def test_crashed_driver_stale_running_flag_becomes_recorded_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            cfg = self.config(directory)
            stage = cfg['stages'][0]
            watchdog.write(Path(stage['root']) / 'status.json', {'status': 'running', 'stage': 'generation'})
            supervisor = watchdog.Supervisor(cfg)
            supervisor.state['stages']['focus'] = {'attempts': 0, 'tracked': {}}
            with patch.object(watchdog, 'process_table', return_value={}):
                terminal, result = supervisor.monitor(stage)
            self.assertTrue(terminal)
            self.assertEqual(result['status'], 'batch_setup_exception_recorded')
            self.assertEqual(result['previous_status']['status'], 'running')
            self.assertTrue(Path(result['supervisor_record']).is_file())
            self.assertEqual(watchdog.read(Path(stage['root']) / 'status.json')['status'], result['status'])

    def test_deadline_does_not_launch_pending_stage(self):
        with tempfile.TemporaryDirectory() as directory:
            cfg = self.config(directory)
            cfg['deadline_utc'] = '2000-01-01T00:00:00Z'
            with patch.object(watchdog, 'process_table', return_value={}), patch.object(watchdog.subprocess, 'Popen') as launch:
                supervisor = watchdog.Supervisor(cfg)
                supervisor.run()
                launch.assert_not_called()
                self.assertEqual(supervisor.state['status'], 'deadline_reached_pending_stages')


if __name__ == '__main__':
    unittest.main()
