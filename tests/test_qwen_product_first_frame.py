import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from PIL import Image

from multishot.qwen_image21_first_frame import (
    _build_qwen_prompt,
    _build_reference_inputs,
)
from multishot.first_frame_backends import generate_first_frame
from multishot import qwen_image21_first_frame as qwen


class QwenProductFirstFrameInputsTest(unittest.TestCase):
    def test_scene_is_first_and_characters_follow_shot_order(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            scene = root / "scene.png"
            alice = root / "alice.png"
            bob = root / "bob.png"
            for path in (scene, alice, bob):
                Image.new("RGB", (8, 8), "white").save(path)

            references, characters = _build_reference_inputs(
                {"asset_id": "scene_001_background", "path": str(scene)},
                {
                    "char_alice": {
                        "character_name": "Alice",
                        "path": str(alice),
                        "face_3d": {"model_path": "/tmp/alice.ply"},
                    },
                    "char_bob": {
                        "character_name": "Bob",
                        "path": str(bob),
                        "face_3d": {"model_path": "/tmp/bob.ply"},
                    },
                },
                ["char_bob", "char_alice"],
            )

            self.assertEqual(
                [(item["kind"], item["id"]) for item in references],
                [
                    ("scene", "scene_001_background"),
                    ("character", "char_bob"),
                    ("character", "char_alice"),
                ],
            )
            self.assertEqual(
                [item["name"] for item in characters],
                ["char_bob", "char_alice"],
            )
            self.assertEqual(characters[0]["face_3d"]["model_path"], "/tmp/bob.ply")

    def test_prompt_labels_scene_and_identity_image_indices(self):
        references = [
            {"kind": "scene", "id": "scene_001_background"},
            {"kind": "character", "id": "char_001", "name": "Alice"},
            {"kind": "character", "id": "char_002", "name": "Bob"},
        ]

        prompt = _build_qwen_prompt("Alice greets Bob.", references)

        self.assertIn("Image 1 is the exact scene and background reference", prompt)
        self.assertIn("Image 2 is the exact identity reference for Alice", prompt)
        self.assertIn("Image 3 is the exact identity reference for Bob", prompt)
        self.assertIn("exactly 2 named people", prompt)
        self.assertTrue(prompt.endswith("Alice greets Bob."))

    def test_background_only_prompt_does_not_require_a_character(self):
        prompt = _build_qwen_prompt(
            "An empty street at dawn.",
            [{"kind": "scene", "id": "scene_001_background"}],
        )

        self.assertIn("without any named foreground character", prompt)

    @patch("multishot.qwen_image21_first_frame.generate_qwen_image21_first_frame")
    def test_backend_dispatch_passes_scene_and_character_assets(self, generate):
        generate.return_value = {"frame_path": "/tmp/frame.png"}
        scene_asset = {"path": "/tmp/scene.png"}
        character_assets = {"char_001": {"path": "/tmp/character.png"}}

        result = generate_first_frame(
            shot_id="shot_001",
            first_frame_prompt="A close-up.",
            backend="qwen_image21",
            scene_asset=scene_asset,
            character_assets=character_assets,
            character_ids=["char_001"],
            project_dir="/tmp/project",
            legacy_generator=lambda **kwargs: None,
        )

        self.assertEqual(result["frame_path"], "/tmp/frame.png")
        generate.assert_called_once_with(
            shot_id="shot_001",
            prompt="A close-up.",
            scene_asset=scene_asset,
            character_assets=character_assets,
            character_ids=["char_001"],
            project_dir=Path("/tmp/project"),
        )

    def test_one_failed_3d_role_does_not_block_other_mapped_roles(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reference = root / "portrait.png"
            Image.new("RGB", (64, 64), "white").save(reference)
            model = root / "good.ply"
            model.touch()
            model.with_suffix(".pose_calibration.json").touch()
            characters = [
                {"name": "bad", "display_name": "Bad", "reference_image": str(reference)},
                {"name": "good", "display_name": "Good", "reference_image": str(reference),
                 "face_3d": {"model_path": str(model)}},
            ]
            faces = [SimpleNamespace(bbox=[0, 0, 64, 64], det_score=0.9)] * 2
            with (
                patch("multishot.multi_face_reference.detected_faces", return_value=faces),
                patch("multishot.multi_face_reference.select_faces", return_value=faces),
                patch("multishot.multi_face_reference.assign_characters_to_faces",
                      return_value=(characters, {"unassigned_scheduled_characters": []})),
                patch("multishot.multi_face_reference.build_color_safe_reference",
                      return_value={"bbox": [0, 0, 64, 64], "pose": {}}),
            ):
                result = qwen._prepare_color_safe_references(
                    preview=Image.new("RGB", (128, 128)), characters=characters,
                    output_dir=root / "prepared", minimum_confidence=0.5,
                    minimum_face_height=24, app=None, parser=None,
                )
        self.assertEqual([target["name"] for target in result["targets"]], ["good"])
        self.assertEqual(result["target_failures"][0]["name"], "bad")


class QwenSingleTrajectoryTest(unittest.TestCase):
    def run_generation(self, directory, prompt, ids, prepare_error=None):
        root = Path(directory)
        scene, portrait = root / "scene.png", root / "portrait.png"
        Image.new("RGB", (32, 32), "white").save(scene)
        Image.new("RGB", (32, 32), "white").save(portrait)
        state = {
            "initial": torch.zeros(1, 4, 1),
            "sigmas": torch.linspace(1, 0, 21),
        }
        target = {
            "name": "char_001", "alpha": torch.ones(1, 4, 1),
            "reference_x0": torch.ones(1, 4, 1),
            "effective_strength": 0.4, "active_injection_steps": 1,
        }
        prepared = {
            "targets": [], "target_failures": [],
            "unassigned_scheduled_characters": [],
        }
        pipeline = unittest.mock.MagicMock()
        def prediction(pipe, runtime, latent, index):
            self.assertFalse(torch.is_grad_enabled())
            return torch.zeros_like(latent)
        with (
            patch.dict("os.environ", {"QWEN_IMAGE21_MODEL": str(root),
                                     "QWEN_IMAGE21_WIDTH": "32", "QWEN_IMAGE21_HEIGHT": "32"}),
            patch("torch.cuda.is_available", return_value=True),
            patch("torch.cuda.reset_peak_memory_stats"),
            patch("torch.cuda.empty_cache"),
            patch("torch.cuda.max_memory_allocated", return_value=0),
            patch.object(qwen, "_load_pipeline", return_value=pipeline),
            patch.object(qwen, "_prepare_conditioning", return_value=state) as condition,
            patch.object(qwen, "_predict", side_effect=prediction) as predict,
            patch.object(qwen, "_decode", return_value=Image.new("RGB", (32, 32))) as decode,
            patch.object(qwen, "_load_face_runtime", return_value=(None, None)),
            patch.object(qwen, "_prepare_color_safe_references",
                         return_value=prepared, side_effect=prepare_error) as prepare,
            patch.object(qwen, "_prepare_target_records", return_value=([target], [])),
        ):
            result = qwen.generate_qwen_image21_first_frame(
                shot_id="shot_001", prompt=prompt,
                scene_asset={"path": str(scene)},
                character_assets={"char_001": {"path": str(portrait)}},
                character_ids=ids, project_dir=root,
            )
        self.assertEqual(condition.call_count, 1)
        self.assertEqual(predict.call_count, 20)
        self.assertEqual([call.args[3] for call in predict.call_args_list], list(range(20)))
        self.assertEqual(pipeline.call_count, 0)
        self.assertTrue(Path(result["frame_path"]).is_file())
        self.assertFalse(list(root.rglob("control.png")))
        return result["metadata"], decode.call_count, prepare.call_count

    def test_injection_is_prepared_once_and_applied_on_same_trajectory(self):
        with tempfile.TemporaryDirectory() as directory:
            meta, decodes, preparations = self.run_generation(directory, "A portrait.", ["char_001"])
        self.assertTrue(meta["injection_applied"])
        self.assertEqual(decodes, 2)  # Mid-step preview and final frame.
        self.assertEqual(preparations, 1)
        self.assertEqual([row["step"] for row in meta["step_log"] if row["injected"]], [12])

    def test_closed_eye_and_background_only_shots_finish_without_injection(self):
        for prompt, ids in [("Her eyes are closed.", ["char_001"]), ("Empty street.", [])]:
            with self.subTest(prompt=prompt), tempfile.TemporaryDirectory() as directory:
                meta, decodes, preparations = self.run_generation(directory, prompt, ids)
            self.assertFalse(meta["injection_applied"])
            self.assertEqual(decodes, 1)
            self.assertEqual(preparations, 0)

    def test_preparation_failure_continues_without_restarting_generation(self):
        with tempfile.TemporaryDirectory() as directory:
            meta, _, preparations = self.run_generation(
                directory, "A portrait.", ["char_001"], RuntimeError("detected 0 reliable faces")
            )
        self.assertFalse(meta["injection_applied"])
        self.assertEqual(meta["skip_reason"], "no_reliable_face_detected")
        self.assertEqual(preparations, 1)
        self.assertTrue(meta["target_failures"])

    def test_out_of_memory_is_not_reported_as_successful_fallback(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(torch.OutOfMemoryError):
                self.run_generation(
                    directory, "A portrait.", ["char_001"], torch.OutOfMemoryError("OOM")
                )


if __name__ == "__main__":
    unittest.main()
