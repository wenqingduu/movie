import math
from functools import lru_cache
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INSIGHTFACE_ROOT = PROJECT_ROOT / "models" / "insightface"


def cosine_similarity(left: list[float], right: list[float]):
    if not left or not right:
        return 0.0
    return round(sum(a * b for a, b in zip(left, right)), 4)


def _as_float_list(values):
    return [round(float(value), 6) for value in values]


class InsightFaceBackend:
    """InsightFace 人脸检测、姿态和身份特征后端。"""

    def __init__(self):
        self._app = None

    def _load(self):
        if self._app is not None:
            return self._app

        from insightface.app import FaceAnalysis

        # 先用 CPU provider，保证和当前服务器 CUDA/onnxruntime 版本解耦。
        # 后续如果要提速，可以把 MULTISHOT_INSIGHTFACE_PROVIDERS 设成 CUDAExecutionProvider,CPUExecutionProvider。
        import os
        providers = [
            item.strip()
            for item in os.getenv("MULTISHOT_INSIGHTFACE_PROVIDERS", "CPUExecutionProvider").split(",")
            if item.strip()
        ]
        app = FaceAnalysis(
            name="buffalo_l",
            root=str(INSIGHTFACE_ROOT),
            providers=providers,
        )
        ctx_id = 0 if providers and providers[0] == "CUDAExecutionProvider" else -1
        app.prepare(ctx_id=ctx_id, det_size=(640, 640))
        self._app = app
        return app

    def analyze(self, image_path: str):
        """返回当前图中的人脸列表，按 x 坐标从左到右排序。"""

        import cv2

        image = cv2.imread(str(image_path))
        if image is None:
            return []

        app = self._load()
        variants = [(image, 1.0, "original")]
        # 生成的人物参考图里脸有时偏小；检测不到时尝试 2x 放大。
        import cv2
        variants.append((cv2.resize(image, None, fx=2.0, fy=2.0), 2.0, "upscaled_2x"))

        faces = []
        scale = 1.0
        source = "original"
        for candidate_image, candidate_scale, candidate_source in variants:
            candidate_faces = app.get(candidate_image)
            if candidate_faces:
                faces = candidate_faces
                scale = candidate_scale
                source = candidate_source
                break

        results = []
        for index, face in enumerate(sorted(faces, key=lambda item: float(item.bbox[0]))):
            bbox = [round(float(value) / scale, 2) for value in face.bbox]
            embedding = _as_float_list(face.normed_embedding)
            pose = getattr(face, "pose", [0.0, 0.0, 0.0])
            results.append({
                "face_id": f"face_{index}",
                "face_index": index,
                "face_embedding": embedding,
                "face_bbox": bbox,
                "face_confidence": round(float(getattr(face, "det_score", 0.0)), 4),
                "detection_source": source,
                "pose": {
                    "pitch": round(float(pose[0]), 4),
                    "yaw": round(float(pose[1]), 4),
                    "roll": round(float(pose[2]), 4),
                    "view": self._pose_to_view(float(pose[1])),
                },
            })
        return results

    def first_face_embedding(self, image_path: str):
        faces = self.analyze(image_path)
        if not faces:
            return None
        return max(
            faces,
            key=lambda face: (face["face_bbox"][2] - face["face_bbox"][0])
            * (face["face_bbox"][3] - face["face_bbox"][1]),
        )["face_embedding"]

    @staticmethod
    def _pose_to_view(yaw: float):
        if yaw <= -20:
            return "left"
        if yaw >= 20:
            return "right"
        return "near_frontal"


@lru_cache(maxsize=1)
def get_face_backend():
    return InsightFaceBackend()
