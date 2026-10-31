"""Recording must preserve solver outputs and distinguish pre/post injection."""
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'pretest'))
from qwen140_export_clara_trajectory import TrajectoryRecorder


class Scheduler:
    def __init__(self):
        self.sigmas = torch.linspace(1., 0., 21)
        self.step_index = 0
    def step(self, velocity, timestep, sample, return_dict=False):
        i = self.step_index
        value = sample + (self.sigmas[i+1] - self.sigmas[i]) * velocity
        self.step_index += 1
        return (value,)


def module():
    return SimpleNamespace(
        _predict=lambda pipe, state, z, i: z * .1 + i * .01,
        _euler_step=lambda z, v, sigmas, i: z + (sigmas[i+1] - sigmas[i]) * v,
        _decode=lambda pipe, z, w, h: z.clone())


class RecorderTest(unittest.TestCase):
    def setUp(self):
        self.pipe = SimpleNamespace(scheduler=Scheduler())
        self.mod = module()
        self.recorder = TrajectoryRecorder(self.pipe, self.mod)
        self.recorder.install()
    def tearDown(self):
        self.recorder.restore()
    def control(self):
        initial = torch.arange(12).reshape(1, 4, 3).float()
        z = initial.clone()
        reference = initial.clone()
        for i in range(20):
            v = z * .1 + i * .01
            z = self.pipe.scheduler.step(v, i, z)[0]
            reference = reference + (self.pipe.scheduler.sigmas[i+1]-self.pipe.scheduler.sigmas[i]) * (reference*.1+i*.01)
        return initial, z, reference
    def test_control_is_exact_and_all_states_recorded(self):
        initial, z, reference = self.control()
        self.assertTrue(torch.equal(z, reference))
        self.assertEqual(set(self.recorder.data['control_z']), set(range(21)))
        self.assertEqual(len(self.recorder.data['control_x0']), 20)
        self.assertTrue(torch.equal(self.recorder.data['control_z'][0], initial))
    def test_actual_injected_state_is_not_euler_before(self):
        self.control()
        state = {'initial': self.recorder.data['control_z'][0], 'sigmas': self.pipe.scheduler.sigmas}
        z = self.recorder.data['control_z'][12].clone()
        self.mod._decode(self.pipe, z, 1, 1)
        for i in range(12, 20):
            prediction = self.mod._predict(self.pipe, state, z, i)
            z = self.mod._euler_step(z, prediction, state['sigmas'], i)
            if i == 12:
                z = z + .25
        self.mod._decode(self.pipe, z, 1, 1)
        data = self.recorder.data
        self.assertTrue(torch.equal(data['control_z'][12], data['treatment_z'][12]))
        self.assertTrue(torch.equal(data['treatment_z'][13], data['treatment_euler_before'][12]+.25))
        self.assertFalse(torch.equal(data['treatment_z'][13], data['treatment_euler_before'][12]))
        self.assertTrue(torch.equal(data['treatment_z'][20], z))
    def test_condition_cache_prefill_is_not_treatment_trajectory(self):
        self.mod._predict(self.pipe, {'initial': torch.ones(1), 'sigmas': self.pipe.scheduler.sigmas}, torch.ones(1), 0)
        self.assertEqual(self.recorder.data['treatment_z'], {})


if __name__ == '__main__':
    unittest.main()
