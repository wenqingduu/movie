"""Reference bindings for an entering role must not pollute a start frame."""
import copy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "pretest"))
from qwen140_shot_agent import compile_prompts, frame_inputs, include_inherited_descriptions


class FrameInputsTest(unittest.TestCase):
    def plan(self, tail=False):
        start = 2 if tail else 1
        return {
            "has_tail": tail,
            "decision": {"route": "C", "tail_identity": "", "frame_characters": ["Kenji", "Ryan"],
                         "frame_prompt": f"图{start} Kenji下车，图{start+2} Ryan下车。",
                         "shot_prompt": "Kenji和Ryan下车，Chloe在旁。"},
            "character_descriptions": {"Kenji": "black hair", "Chloe": "red hair", "Ryan": "blond hair"},
            "assets": [{"name": name, "image_index": start+i} for i, name in enumerate(["Kenji", "Chloe", "Ryan"])],
        }

    def test_hardcut_drops_later_role_and_renumbers_reference(self):
        plan = self.plan()
        original = copy.deepcopy(plan)
        assets, prompt = frame_inputs(plan)
        self.assertEqual([(a["name"], a["image_index"]) for a in assets], [("Kenji", 1), ("Ryan", 2)])
        self.assertEqual(prompt, "图1 Kenji下车，图2 Ryan下车。")
        self.assertEqual(plan, original)

    def test_previous_tail_keeps_image_one_reserved(self):
        assets, prompt = frame_inputs(self.plan(tail=True))
        self.assertEqual([a["image_index"] for a in assets], [2, 3])
        self.assertEqual(prompt, "图2 Kenji下车，图3 Ryan下车。")

    def test_later_role_stays_in_video_prompt(self):
        prompts = compile_prompts(self.plan(), "photorealistic film")
        self.assertNotIn("Chloe", prompts["frame"])
        self.assertIn("Chloe", prompts["shot"])
        self.assertIn("图2是Ryan", prompts["frame"])


class InheritedCastTest(unittest.TestCase):
    def plan(self, tail=True):
        return {"has_tail": tail,
                "decision": {"route": "B" if tail else "C", "tail_identity": "Nora on couch",
                             "frame_characters": ["Nora", "Chloe", "Leo"],
                             "frame_prompt": "Nora seated; Chloe and Leo beside couch",
                             "transition_prompt": "Chloe and Leo enter", "shot_prompt": "They approach Nora"},
                "assets": [{"name": "Chloe", "image_index": 2}, {"name": "Leo", "image_index": 3}],
                "character_descriptions": {"Chloe": "white blazer", "Leo": "yellow jacket"}}

    def test_existing_tail_role_gets_description_without_extra_reference(self):
        original = self.plan()
        saved = copy.deepcopy(original)
        plan = include_inherited_descriptions(original, {"Nora": "green blouse"})
        prompts = compile_prompts(plan, "film")
        self.assertIn("Nora: green blouse", prompts["frame"])
        self.assertIn("Nora: green blouse", prompts["shot"])
        self.assertEqual([a["name"] for a in frame_inputs(plan)[0]], ["Chloe", "Leo"])
        self.assertEqual(original, saved)

    def test_unknown_role_remains_rejected(self):
        with self.assertRaisesRegex(ValueError, "Nora"):
            include_inherited_descriptions(self.plan(), {"Marco": "grey suit"})

    def test_hardcut_cannot_borrow_prior_role(self):
        with self.assertRaisesRegex(ValueError, "Nora"):
            include_inherited_descriptions(self.plan(tail=False), {"Nora": "green blouse"})


if __name__ == "__main__":
    unittest.main()
