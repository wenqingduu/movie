import json
from pathlib import Path


class AssetMemory:
    def __init__(self, project_dir: str):
        self.project_dir = Path(project_dir)
        self.index_path = self.project_dir / "asset_index.json"#存入的路径
        self.data = {
            "scene_assets": {},
            "character_assets": {},
        }

    def add_scene_asset(self, subscript_id: str, asset: dict):
        self.data["scene_assets"][subscript_id] = asset

    def add_character_asset(self, character_id: str, asset: dict):
        self.data["character_assets"][character_id] = asset

    def save(self):
        self.index_path.write_text(
            json.dumps(self.data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return str(self.index_path)
