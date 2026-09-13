"""ONNX face swappers that are not insightface's INSwapper (HyperSwap, AlphaFace).

:class:`OnnxSwapper` mimics the slice of the ``INSwapper`` API the pipeline
uses — ``input_size`` and ``get(frame, target_face, source_face,
paste_back=False) -> (bgr_fake, M)`` — so the existing crop/paste/mask code
works with any of these models.
"""

import threading
from typing import Any, Optional, Tuple

import cv2
import numpy as np
import onnxruntime
from insightface.utils import face_align

from modules.swapper_registry import SwapperSpec
from modules.cuda_graph import GRAPH_LOCK
from modules.processors.frame._onnx_enhancer import WARP_TEMPLATES


class OnnxSwapper:
    """A swap model driven directly through onnxruntime."""

    def __init__(self, spec: SwapperSpec, model_path: str, providers: list) -> None:
        self.spec = spec
        self.input_size = (spec.input_size, spec.input_size)
        # DFM graphs emit their own face mask; callers check this rather than
        # probing for the method, which every instance of this class has.
        self.returns_mask = spec.kind == "dfm"
        self.morph_name = None

        self._graph: Optional[dict] = None
        self._lock = threading.Lock()

        # The CUDA-graph session is tried first and then kept as *the* session:
        # holding a second plain session would mean a second copy of the model
        # weights in VRAM (400-550 MB for these models).
        wants_cuda = self.spec.cuda_graph and self.spec.kind != "dfm" and any(
            p == "CUDAExecutionProvider" or
            (isinstance(p, tuple) and p[0] == "CUDAExecutionProvider")
            for p in providers
        )
        if not (wants_cuda and self._init_cuda_graph(model_path)):
            session_options = onnxruntime.SessionOptions()
            session_options.graph_optimization_level = (
                onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
            )
            self.session = onnxruntime.InferenceSession(
                model_path, sess_options=session_options, providers=providers,
            )
            self._read_io_names()

    # ── inference ────────────────────────────────────────────────────────

    def _init_cuda_graph(self, model_path: str) -> bool:
        """Record a CUDA graph — shapes are static, so replay is nearly free."""
        try:
            session = onnxruntime.InferenceSession(
                model_path, providers=[("CUDAExecutionProvider", {"enable_cuda_graph": "1"})],
            )
            self.session = session
            self._read_io_names()
            size = self.spec.input_size
            ort_target = onnxruntime.OrtValue.ortvalue_from_numpy(
                np.zeros((1, 3, size, size), dtype=np.float32), "cuda", 0,
            )
            ort_source = onnxruntime.OrtValue.ortvalue_from_numpy(
                np.zeros((1, 512), dtype=np.float32), "cuda", 0,
            )
            io_binding = session.io_binding()
            io_binding.bind_ortvalue_input(self.target_name, ort_target)
            io_binding.bind_ortvalue_input(self.source_name, ort_source)
            io_binding.bind_output(self.output_name, "cuda", 0)
            with GRAPH_LOCK:
                session.run_with_iobinding(io_binding)  # records the graph
            self._graph = {
                "session": session,
                "io_binding": io_binding,
                "target": ort_target,
                "source": ort_source,
            }
            print(f"[DLC.FACE-SWAPPER] CUDA graph session initialized ({self.spec.key})")
            return True
        except Exception as error:
            print(f"[DLC.FACE-SWAPPER] CUDA graph init failed for "
                  f"{self.spec.key}, using standard session: {error}")
            self._graph = None
            self.session = None
            return False

    def _read_io_names(self) -> None:
        inputs = {i.name: i for i in self.session.get_inputs()}
        self.target_name = "target" if "target" in inputs else list(inputs)[0]
        self.source_name = (
            "source" if "source" in inputs
            else (list(inputs)[1] if len(inputs) > 1 else None)
        )
        self.output_name = self.session.get_outputs()[0].name
        if self.spec.kind == "dfm":
            # DeepFaceLive graphs use in_face:0 [1, S, S, 3] and, on AMP
            # models, a morph_value:0 scalar.
            self.target_name = "in_face:0" if "in_face:0" in inputs else list(inputs)[0]
            self.source_name = None
            self.morph_name = "morph_value:0" if "morph_value:0" in inputs else None
            shape = inputs[self.target_name].shape
            if isinstance(shape[1], int) and shape[1] > 0:
                self.input_size = (shape[1], shape[1])
                self.spec = self.spec._replace(input_size=shape[1])

    def _run(self, blob: np.ndarray, embedding: np.ndarray) -> np.ndarray:
        if self._graph is not None:
            with self._lock, GRAPH_LOCK:
                self._graph["target"].update_inplace(blob)
                self._graph["source"].update_inplace(embedding)
                self._graph["session"].run_with_iobinding(self._graph["io_binding"])
                return self._graph["io_binding"].get_outputs()[0].numpy()
        return self.session.run(
            None, {self.target_name: blob, self.source_name: embedding},
        )[0]

    def _warp(self, img: np.ndarray, face: Any, size: int):
        """Align a face with this model's template (insightface for arcface)."""
        if self.spec.template == "arcface_128":
            return face_align.norm_crop2(img, face.kps, size)
        template = WARP_TEMPLATES[self.spec.template] * size
        M = cv2.estimateAffinePartial2D(
            np.asarray(face.kps, dtype=np.float32), template, method=cv2.LMEDS,
        )[0]
        crop = cv2.warpAffine(img, M, (size, size), borderMode=cv2.BORDER_REPLICATE)
        return crop, M

    def get_with_mask(self, img: np.ndarray, target_face: Any, source_face: Any):
        """DFM swap: returns the face plus the model's own mask in crop space."""
        size = self.spec.input_size
        aimg, M = self._warp(img, target_face, size)

        # DeepFaceLive sharpens the crop before inference; the models were
        # trained on that input, so skipping it softens the result.
        sharpened = cv2.addWeighted(
            aimg, 1.75, cv2.GaussianBlur(aimg, (0, 0), 2), -0.75, 0,
        )
        blob = (sharpened / 255.0)[np.newaxis, ...].astype(np.float32)

        feed = {self.target_name: blob}
        if self.morph_name is not None:
            feed[self.morph_name] = np.array([1.0], dtype=np.float32)
        with self._lock:
            target_mask, celeb_face, source_mask = self.session.run(None, feed)

        bgr_fake = np.clip(celeb_face[0] * 255.0, 0, 255).astype(np.uint8)

        mask = np.minimum(source_mask[0], target_mask[0]).reshape(size, size).clip(0, 1)
        mask = cv2.erode(
            mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=2,
        )
        mask = cv2.GaussianBlur(mask, (0, 0), 6.25)
        return bgr_fake, M, mask

    # ── INSwapper-compatible API ─────────────────────────────────────────

    def get(
        self, img: np.ndarray, target_face: Any, source_face: Any,
        paste_back: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray]:
        if self.spec.kind == "dfm":
            bgr_fake, M, _ = self.get_with_mask(img, target_face, source_face)
            return bgr_fake, M

        size = self.spec.input_size
        aimg, M = face_align.norm_crop2(img, target_face.kps, size)

        blob = aimg[:, :, ::-1].astype(np.float32) / 255.0
        if self.spec.std != 1.0 or self.spec.mean != 0.0:
            blob = (blob - self.spec.mean) / self.spec.std
        blob = np.expand_dims(blob.transpose(2, 0, 1), axis=0)

        if self.spec.embedding == "normed":
            embedding = target_face.normed_embedding if source_face is None \
                else source_face.normed_embedding
        else:
            embedding = source_face.embedding
        embedding = np.asarray(embedding, dtype=np.float32).reshape(1, -1)

        output = self._run(np.ascontiguousarray(blob, dtype=np.float32), embedding)

        fake = output[0].transpose(1, 2, 0)
        if self.spec.denormalize:
            fake = fake * self.spec.std + self.spec.mean
        fake = np.clip(fake, 0, 1)[:, :, ::-1] * 255.0
        bgr_fake = fake.astype(np.uint8)

        if paste_back:
            inv_M = cv2.invertAffineTransform(M)
            h, w = img.shape[:2]
            warped = cv2.warpAffine(bgr_fake, inv_M, (w, h), borderValue=(0, 0, 0))
            mask = cv2.warpAffine(
                np.ones((size, size), dtype=np.float32), inv_M, (w, h), borderValue=0,
            )[:, :, np.newaxis]
            merged = warped.astype(np.float32) * mask + img.astype(np.float32) * (1 - mask)
            return np.clip(merged, 0, 255).astype(np.uint8), M

        return bgr_fake, M
