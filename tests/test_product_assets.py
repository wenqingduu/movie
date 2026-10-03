import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from multishot.agents import _mcp_env
from multishot.mcp_asset_server import _generate_image
from multishot.product_face_assets import calibrate_product_face


class ProductAssetTest(unittest.TestCase):
    def test_product_mcp_defaults_to_evaluation_asset_model(self):
        with patch.dict(os.environ, {}, clear=True):
            env = _mcp_env(Path("/tmp/project"))
        self.assertEqual(env["MULTISHOT_ASSET_GENERATION_MODEL"], "sdxl-base-1.0")
        self.assertEqual(env["MULTISHOT_DIFFUSION_STEPS"], "30")
        self.assertEqual(env["MULTISHOT_GUIDANCE_SCALE"], "5.0")

    def test_product_model_selection_preserves_explicit_overrides(self):
        with patch.dict(os.environ, {"MULTISHOT_ASSET_GENERATION_MODEL": "sdxl-base-1.0"}), patch(
            "multishot.mcp_asset_server.get_diffusion_backend"
        ) as get_backend:
            _generate_image("A garage", "/tmp/scene.png")
            get_backend.assert_called_with("sdxl-base-1.0")
            _generate_image("A garage", "/tmp/scene.png", "legacy-model")
            get_backend.assert_called_with("legacy-model")

    def test_calibration_is_role_local_and_uses_evaluation_render_size(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = root / "gaussians.ply"
            model.touch()
            calibration = model.with_suffix(".pose_calibration.json")
            def calibrate(command, **kwargs):
                self.assertEqual(command[command.index("--model") + 1], str(model))
                self.assertEqual(command[command.index("--image-size") + 1], "512")
                calibration.write_text("{}")
            with patch.dict(os.environ, {}, clear=True), patch(
                "multishot.product_face_assets.subprocess.run", side_effect=calibrate
            ) as run:
                result = calibrate_product_face({"model_path": str(model), "path": str(root)})
                calibrate_product_face(result)
            self.assertEqual(run.call_count, 1)
            self.assertEqual(result["pose_calibration_path"], str(calibration))

    def test_failed_facelift_is_not_accepted_as_a_3d_asset(self):
        with self.assertRaisesRegex(RuntimeError, "usable Gaussian"):
            calibrate_product_face({"model_path": "/tmp/status.json", "facelift_status": "failed"})


if __name__ == "__main__":
    unittest.main()
