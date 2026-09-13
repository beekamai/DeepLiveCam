"""Reusable frame-processor implementation for ONNX face restoration models.

Every enhancer module (GPEN, CodeFormer, RestoreFormer++, GFPGAN) is the same
pipeline with a different model file, crop size and alignment template, so the
per-model modules are thin wrappers around :class:`OnnxEnhancer`.
"""

import gc
import os
import threading
from typing import Any, List, Optional

import modules.globals
import modules.processors.frame.core
from modules import imread_unicode, imwrite_unicode
from modules.core import update_status
from modules.face_analyser import get_one_face
from modules.typing import Frame, Face
from modules.utilities import is_image, is_video
from modules.processors.frame._onnx_enhancer import (
    create_onnx_session,
    warmup_session,
    enhance_face_onnx,
)

MODELS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))))),
    "models",
)


class OnnxEnhancer:
    """One ONNX face restoration model exposed as a frame processor."""

    def __init__(
        self,
        name: str,
        model_file: str,
        input_size: int,
        template: str,
        mirror_url: Optional[str] = None,
    ) -> None:
        self.name = name
        self.model_file = model_file
        self.input_size = input_size
        self.template = template
        self.mirror_url = mirror_url
        self._session = None
        self._lock = threading.Lock()

    # ── model acquisition ────────────────────────────────────────────────

    def _obtain_model(self) -> Optional[str]:
        from modules.model_downloader import ensure_model

        model_path = ensure_model(self.model_file)
        if model_path is not None:
            return model_path
        if self.mirror_url is None:
            return None

        update_status(f"Retrying {self.model_file} from the mirror...", self.name)
        from modules.utilities import conditional_download

        try:
            conditional_download(MODELS_DIR, [self.mirror_url])
        except Exception as error:
            update_status(f"Mirror download failed: {error}", self.name)
            return None
        fallback = os.path.join(MODELS_DIR, self.model_file)
        return fallback if os.path.exists(fallback) else None

    def pre_check(self) -> bool:
        if self._obtain_model() is None:
            update_status(
                f"Could not obtain {self.model_file}. Place it in the models "
                "folder manually or check your internet connection.",
                self.name,
            )
            return False
        return True

    def pre_start(self) -> bool:
        if not is_image(modules.globals.target_path) and not is_video(
            modules.globals.target_path
        ):
            update_status("Select an image or video for target path.", self.name)
            return False
        return True

    def get_session(self) -> Any:
        with self._lock:
            if self._session is None:
                model_path = self._obtain_model()
                if model_path is None:
                    raise FileNotFoundError(
                        f"Model file not found: "
                        f"{os.path.join(MODELS_DIR, self.model_file)}"
                    )
                print(f"{self.name}: Loading ONNX model from {model_path}")
                session = create_onnx_session(model_path)
                shape = session.get_inputs()[0].shape
                if isinstance(shape[-1], int) and shape[-1] > 0:
                    self.input_size = shape[-1]
                warmup_session(session)
                self._session = session
                print(f"{self.name}: Model loaded ({self.input_size}px, "
                      f"template={self.active_template()}).")
        return self._session

    def active_template(self) -> str:
        """Alignment template for this run — the UI setting overrides the spec."""
        if getattr(modules.globals, "enhancer_alignment", "legacy") == "legacy":
            return "dlc_legacy"
        return self.template

    # ── frame processing ─────────────────────────────────────────────────

    def enhance_face(self, temp_frame: Frame, face: Face) -> Frame:
        try:
            session = self.get_session()
        except Exception as error:
            print(f"{self.name}: {error}")
            return temp_frame
        try:
            return enhance_face_onnx(
                temp_frame, face, session, self.input_size, self.active_template(),
            )
        except Exception as error:
            print(f"{self.name}: Error during face enhancement: {error}")
            return temp_frame

    def process_frame(
        self, source_face: Face | None, temp_frame: Frame, detected_faces=None
    ) -> Frame:
        faces = detected_faces if detected_faces else None
        if faces is None:
            target_face = get_one_face(temp_frame)
            faces = [target_face] if target_face else []
        for face in faces:
            temp_frame = self.enhance_face(temp_frame, face)
        return temp_frame

    def process_frame_v2(self, temp_frame: Frame, detected_faces=None) -> Frame:
        return self.process_frame(None, temp_frame, detected_faces)

    def process_frames(
        self, source_path: str | None, temp_frame_paths: List[str], progress: Any = None
    ) -> None:
        for temp_frame_path in temp_frame_paths:
            temp_frame = imread_unicode(temp_frame_path)
            if temp_frame is None:
                if progress:
                    progress.update(1)
                continue
            imwrite_unicode(temp_frame_path, self.process_frame(None, temp_frame))
            if progress:
                progress.update(1)

    def process_image(
        self, source_path: str | None, target_path: str, output_path: str
    ) -> None:
        target_frame = imread_unicode(target_path)
        if target_frame is None:
            print(f"{self.name}: Error: Failed to read target image {target_path}")
            return
        imwrite_unicode(output_path, self.process_frame(None, target_frame))
        print(f"{self.name}: Enhanced image saved to {output_path}")

    def process_video(self, source_path: str | None, temp_frame_paths: List[str]) -> None:
        modules.processors.frame.core.process_video(
            source_path, temp_frame_paths, self.process_frames,
        )

    def release(self) -> None:
        """Drop the ONNX session so its weights leave VRAM.

        Sessions are cached for the lifetime of the process, so switching
        models in the UI would otherwise keep every model ever selected
        resident on the GPU.
        """
        with self._lock:
            if self._session is None:
                return
            self._session = None
        gc.collect()
        print(f"{self.name}: model unloaded")

    def as_module_api(self) -> dict:
        """Return the frame-processor interface for ``globals().update(...)``."""
        return {
            "NAME": self.name,
            "MODEL_FILE": self.model_file,
            "INPUT_SIZE": self.input_size,
            "TEMPLATE": self.template,
            "pre_check": self.pre_check,
            "pre_start": self.pre_start,
            "enhance_face": self.enhance_face,
            "process_frame": self.process_frame,
            "process_frame_v2": self.process_frame_v2,
            "process_frames": self.process_frames,
            "process_image": self.process_image,
            "process_video": self.process_video,
            "release": self.release,
        }
