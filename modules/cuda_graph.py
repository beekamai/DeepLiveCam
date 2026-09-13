"""CUDA graph replay for small ONNX models.

Networks built from many small layers (the face detector, for one) spend more
time launching kernels than running them — shrinking the input barely helps.
Recording the launch sequence once and replaying it removes that overhead;
measured 12 ms -> 6 ms on det_10g at 640x640.

Requires static input shapes, so callers pass the exact shape they will use and
the session falls back to a normal run for anything else.
"""

import threading
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import onnxruntime

# One lock for every CUDA graph in the process.  Capturing a graph while
# another thread issues CUDA work on the device fails with "operation not
# permitted when stream is capturing" and poisons both sessions, and replays
# from two threads can collide the same way.  The GPU serialises the work
# regardless, so holding the lock only around the device call costs nothing —
# the CPU-side warps and blends still overlap.
GRAPH_LOCK = threading.RLock()


class GraphSession:
    """An ONNX session that replays a recorded CUDA graph for one input shape.

    Exposes ``run(output_names, input_feed)`` so it can stand in for an
    ``InferenceSession`` inside third-party code (e.g. insightface's SCRFD).
    """

    def __init__(self, model_path: str, input_name: str,
                 input_shape: Sequence[int]) -> None:
        self._input_name = input_name
        self._input_shape = tuple(input_shape)
        self._lock = threading.Lock()
        # TensorRT compiles the whole model into one engine — no launch
        # sequence to record; plain runs are the fastest path there.
        from modules.providers import make_session

        self._trt = make_session(model_path)
        if self._trt is not None:
            self._session = self._trt
            self._output_names = [o.name for o in self._session.get_outputs()]
            return
        self._session = onnxruntime.InferenceSession(
            model_path, providers=[("CUDAExecutionProvider", {"enable_cuda_graph": "1"})],
        )

        self._ort_input = onnxruntime.OrtValue.ortvalue_from_numpy(
            np.zeros(self._input_shape, dtype=np.float32), "cuda", 0,
        )
        self._binding = self._session.io_binding()
        self._binding.bind_ortvalue_input(input_name, self._ort_input)
        self._output_names = [o.name for o in self._session.get_outputs()]
        for name in self._output_names:
            self._binding.bind_output(name, "cuda", 0)
        with GRAPH_LOCK:
            self._session.run_with_iobinding(self._binding)  # records the graph

    @property
    def input_shape(self) -> Tuple[int, ...]:
        return self._input_shape

    def run(self, output_names: Optional[List[str]],
            input_feed: Dict[str, np.ndarray], **kwargs: Any) -> List[np.ndarray]:
        blob = input_feed.get(self._input_name)
        if self._trt is not None or blob is None or tuple(blob.shape) != self._input_shape:
            with GRAPH_LOCK:
                return self._session.run(output_names, input_feed, **kwargs)

        with self._lock, GRAPH_LOCK:
            self._ort_input.update_inplace(np.ascontiguousarray(blob, dtype=np.float32))
            self._session.run_with_iobinding(self._binding)
            outputs = [value.numpy() for value in self._binding.get_outputs()]

        if not output_names:
            return outputs
        index = {name: i for i, name in enumerate(self._output_names)}
        return [outputs[index[name]] for name in output_names]

    def __getattr__(self, name: str) -> Any:
        return getattr(self._session, name)


def build_static_model(source_path: str, target_path: str,
                       dims: Dict[int, int]) -> Optional[str]:
    """Write a copy of the model with the given input dimensions pinned.

    ``dims`` maps input-axis index to size, e.g. ``{0: 1, 2: 640, 3: 640}``.
    Returns the new path, or ``None`` when the rewrite is not possible.
    """
    import os

    if os.path.isfile(target_path):
        return target_path
    try:
        import onnx

        model = onnx.load(source_path)
        shape = model.graph.input[0].type.tensor_type.shape.dim
        for axis, value in dims.items():
            shape[axis].Clear()
            shape[axis].dim_value = value
        os.makedirs(os.path.dirname(target_path) or ".", exist_ok=True)
        onnx.save(model, target_path)
        return target_path
    except Exception as error:
        print(f"[cuda_graph] could not pin input shape for {source_path}: {error}")
        return None
