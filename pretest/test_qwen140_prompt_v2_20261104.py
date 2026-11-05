import unittest
from qwen140_prompt_v2_20261104 import compile_prompts, SYSTEM


class PromptStageTest(unittest.TestCase):
    def plan(self, route):
        return {'decision': {'route': route, 'tail_identity': 'Leo在左侧扶桌。',
                'frame_characters': ['Leo'], 'frame_prompt': '图1的Leo已到右侧。',
                'transition_prompt': 'Leo走到右侧。', 'shot_prompt': '右侧的Leo开抽屉。'},
                'assets': [], 'character_descriptions': {'Leo': '白衬衫男子'}, 'has_tail': True}

    def test_b_uses_previous_layout_only_for_transition(self):
        p = self.plan('B')
        out = compile_prompts(p, '写实')
        self.assertIn('Leo在左侧扶桌。', out['transition'])
        self.assertNotIn('Leo在左侧扶桌。', out['shot'])
        self.assertIn('右侧的Leo开抽屉。', out['shot'])
        self.assertIn('图1', out['frame'])
        self.assertEqual(p['decision']['tail_identity'], 'Leo在左侧扶桌。')

    def test_a_preserves_actual_start_binding(self):
        self.assertIn('Leo在左侧扶桌。', compile_prompts(self.plan('A'), '写实')['shot'])

    def test_system_limits_indices_and_preserves_action(self):
        self.assertIn('图片编号仅用于frame_prompt', SYSTEM)
        self.assertIn('不额外添加动作或运镜', SYSTEM)


if __name__ == '__main__':
    unittest.main()
