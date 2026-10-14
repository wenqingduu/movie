import sys
import unittest
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'pretest'))
from flux_reference_gate_recovery import call_single_with_reference_gate,validate_recovery_record,sha256,ALIGN_FAILURE

class ReferenceGateRecoveryTests(unittest.TestCase):
    def test_success_calls_original_once_without_fallback(self):
        core=SimpleNamespace(run=Mock());args=object()
        self.assertIsNone(call_single_with_reference_gate(core,args))
        core.run.assert_called_once_with(args)

    def test_exact_reference_failure_is_reported(self):
        core=SimpleNamespace(run=Mock(side_effect=RuntimeError(ALIGN_FAILURE)))
        result=call_single_with_reference_gate(core,object())
        self.assertEqual(result['error'],ALIGN_FAILURE)
        self.assertEqual(result['stage'],'PuLID reference identity embedding before image denoising')

    def test_oom_and_other_runtime_errors_are_not_swallowed(self):
        for message in ('CUDA out of memory','target face missing',ALIGN_FAILURE+' unrelated suffix'):
            with self.subTest(message=message):
                core=SimpleNamespace(run=Mock(side_effect=RuntimeError(message)))
                with self.assertRaises(RuntimeError):call_single_with_reference_gate(core,object())

    def test_io_failure_still_aborts(self):
        core=SimpleNamespace(run=Mock(side_effect=FileNotFoundError('reference missing')))
        with self.assertRaises(FileNotFoundError):call_single_with_reference_gate(core,object())

    def test_resume_rejects_changed_fallback_policy_or_reference(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);(root/'runtime').mkdir()
            policy=root/'runtime/reference_gate_recovery.json';policy.write_text('frozen recovery policy')
            reference=root/'reference.png';reference.write_bytes(b'original reference')
            row=dict(plugin_applied=False,source_kind='flux_native_reference_alignment_fallback_hardcut',
                reference_alignment_recovery=dict(error=ALIGN_FAILURE,policy_sha256=sha256(policy),reference=str(reference),reference_sha256=sha256(reference)))
            validate_recovery_record(root,row)
            policy.write_text('changed recovery policy')
            with self.assertRaises(AssertionError):validate_recovery_record(root,row)
            policy.write_text('frozen recovery policy');reference.write_bytes(b'changed appearance')
            with self.assertRaises(AssertionError):validate_recovery_record(root,row)

if __name__=='__main__':unittest.main()
