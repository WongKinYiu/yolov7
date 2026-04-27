"""CPU postprocessor for DLA-exported YOLOv7 segmentation/detection models.

When the model is exported with ``--dla``, the on-device graph emits 4 raw
4D tensors (proto + 3 detection scales) and skips the YOLO decode entirely.
This module reproduces that decode on the CPU in pure NumPy so the DLA
engine can stay DLA-only (no GPU fallback for reshape/permute/sigmoid/cat,
whose 5D and concat-dim-32760 variants are not supported by DLA).

Expected raw outputs from TensorRT on the Jetson Orin NX:
    proto   : (bs, nm,        H/4,  W/4)   -- already decoded, mask protos
    det_p3  : (bs, na*no,     H/8,  W/8)
    det_p4  : (bs, na*no,     H/16, W/16)
    det_p5  : (bs, na*no,     H/32, W/32)

With ``na=3`` and ``no=nc + 5 + nm`` (segmentation) or ``no=nc + 5``
(detection).

Typical budget on Orin NX at 832x832, ~32760 anchors, nc small, nm=32:
    - decode + flatten + cat: ~0.5-1.0 ms
    - NMS (batched-NMS fallback, ~top-1000 pre-NMS): ~0.5-1.0 ms
    - mask generation (per surviving det): ~0.1 ms each
Total budget fits comfortably within the remaining ~15 ms headroom after
the ~16-17 ms DLA latency for VVD_1 @ 60 Hz.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


# Default YOLOv7 anchors (P3/P4/P5). Override via ``DLAPostprocessor(anchors=...)``
# if the production YAML uses different values.
DEFAULT_ANCHORS = np.array([
    [[12, 16], [19, 36], [40, 28]],       # P3/8
    [[36, 75], [76, 55], [72, 146]],      # P4/16
    [[142, 110], [192, 243], [459, 401]], # P5/32
], dtype=np.float32)

DEFAULT_STRIDES = np.array([8, 16, 32], dtype=np.float32)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    # Numerically stable logistic. In-place avoids a transient for 32k*no
    # floats on CPU. np.negative is faster than unary minus for large arrays.
    out = np.empty_like(x)
    np.negative(x, out=out)
    np.exp(out, out=out)
    out += 1.0
    np.reciprocal(out, out=out)
    return out


@dataclass
class DLAPostprocessor:
    """Decode + NMS for DLA-exported YOLOv7 heads.

    Parameters
    ----------
    nc : int
        Number of classes.
    nm : int
        Number of mask coefficients (0 for pure detection).
    anchors : np.ndarray, optional
        Shape (nl, na, 2), in pixel units on the input image.
    strides : np.ndarray, optional
        Shape (nl,), network stride per detection level (e.g. [8, 16, 32]).
    conf_thres : float
        Objectness * class-confidence threshold.
    iou_thres : float
        NMS IoU threshold.
    max_det : int
        Max detections kept after NMS.
    max_nms : int
        Max candidates fed into NMS (pre-NMS top-k by score).
    """

    nc: int
    nm: int = 0
    anchors: np.ndarray = None  # type: ignore[assignment]
    strides: np.ndarray = None  # type: ignore[assignment]
    conf_thres: float = 0.25
    iou_thres: float = 0.45
    max_det: int = 300
    max_nms: int = 3000

    def __post_init__(self) -> None:
        if self.anchors is None:
            self.anchors = DEFAULT_ANCHORS.copy()
        if self.strides is None:
            self.strides = DEFAULT_STRIDES.copy()
        self.anchors = np.asarray(self.anchors, dtype=np.float32)
        self.strides = np.asarray(self.strides, dtype=np.float32)
        assert self.anchors.ndim == 3 and self.anchors.shape[-1] == 2, (
            f'anchors must be (nl, na, 2), got {self.anchors.shape}')
        self.nl = int(self.anchors.shape[0])
        self.na = int(self.anchors.shape[1])
        self.no = self.nc + 5 + self.nm  # outputs per anchor
        # Precomputed grid cache keyed by (level, ny, nx). Static DLA shapes
        # mean this is populated once on the first frame and reused forever.
        self._grid_cache: dict[tuple[int, int, int], tuple[np.ndarray, np.ndarray]] = {}

    # ------------------------------------------------------------------
    # Decoding
    # ------------------------------------------------------------------
    def _grid(self, i: int, ny: int, nx: int) -> tuple[np.ndarray, np.ndarray]:
        key = (i, ny, nx)
        cached = self._grid_cache.get(key)
        if cached is not None:
            return cached
        yv, xv = np.meshgrid(np.arange(ny, dtype=np.float32),
                             np.arange(nx, dtype=np.float32),
                             indexing='ij')
        # Shape: (1, na, ny, nx, 2). Matches IDetect._make_grid minus the
        # 0.5 offset, which is applied at decode time: (2*sig - 0.5 + grid).
        grid = np.stack((xv, yv), axis=-1)[None, None, ...]
        grid = np.broadcast_to(grid, (1, self.na, ny, nx, 2)).copy()
        anchor_grid = (self.anchors[i] * self.strides[i]).reshape(1, self.na, 1, 1, 2)
        anchor_grid = np.broadcast_to(anchor_grid, (1, self.na, ny, nx, 2)).copy()
        self._grid_cache[key] = (grid, anchor_grid)
        return grid, anchor_grid

    def decode_level(self, raw: np.ndarray, i: int) -> np.ndarray:
        """Decode one detection scale.

        raw : (bs, na*no, ny, nx) float32
        returns (bs, na*ny*nx, no) float32 with xywh in input-image pixels.
        """
        bs, c, ny, nx = raw.shape
        assert c == self.na * self.no, (
            f'level {i}: expected {self.na * self.no} channels, got {c}')
        # (bs, na, no, ny, nx) -> (bs, na, ny, nx, no). This is the same
        # layout IDetect builds, just in NumPy.
        y = raw.reshape(bs, self.na, self.no, ny, nx).transpose(0, 1, 3, 4, 2)
        y = _sigmoid(y)

        grid, anchor_grid = self._grid(i, ny, nx)
        # xy: (2*sig - 0.5 + grid) * stride
        y[..., 0:2] = (y[..., 0:2] * 2.0 - 0.5 + grid) * self.strides[i]
        # wh: (2*sig)^2 * anchor_grid
        wh = y[..., 2:4] * 2.0
        y[..., 2:4] = wh * wh * anchor_grid
        # Mask coeffs live in the tail (indices 5+nc : 5+nc+nm). They were
        # sigmoided above for simplicity; for YOLOv7-seg the original code
        # does NOT sigmoid mask coeffs (they are linear and combined via
        # matmul with proto then sigmoided). Undo the sigmoid here by
        # substituting the raw values back.
        if self.nm > 0:
            raw_mask = raw.reshape(bs, self.na, self.no, ny, nx).transpose(0, 1, 3, 4, 2)
            y[..., 5 + self.nc:] = raw_mask[..., 5 + self.nc:]
        return y.reshape(bs, self.na * ny * nx, self.no)

    def decode(self, raws: Sequence[np.ndarray]) -> np.ndarray:
        """Concat all decoded scales into (bs, N, no)."""
        assert len(raws) == self.nl, f'expected {self.nl} raw outputs, got {len(raws)}'
        parts = [self.decode_level(r, i) for i, r in enumerate(raws)]
        return np.concatenate(parts, axis=1)

    # ------------------------------------------------------------------
    # NMS
    # ------------------------------------------------------------------
    @staticmethod
    def _xywh_to_xyxy(b: np.ndarray) -> np.ndarray:
        out = np.empty_like(b)
        half_w = b[..., 2] * 0.5
        half_h = b[..., 3] * 0.5
        out[..., 0] = b[..., 0] - half_w
        out[..., 1] = b[..., 1] - half_h
        out[..., 2] = b[..., 0] + half_w
        out[..., 3] = b[..., 1] + half_h
        return out

    @staticmethod
    def _nms(boxes: np.ndarray, scores: np.ndarray, iou_thres: float) -> np.ndarray:
        # Standard greedy NMS. For ~1-3k candidates this runs in well under
        # 1 ms on an Orin NX A78AE CPU; a batched-NMS TRT plugin would
        # shave this further but pure NumPy keeps the pipeline portable.
        if boxes.size == 0:
            return np.empty((0,), dtype=np.int64)
        x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        areas = (x2 - x1).clip(min=0) * (y2 - y1).clip(min=0)
        order = scores.argsort()[::-1]
        keep: list[int] = []
        while order.size > 0:
            i = int(order[0])
            keep.append(i)
            if order.size == 1:
                break
            rest = order[1:]
            xx1 = np.maximum(x1[i], x1[rest])
            yy1 = np.maximum(y1[i], y1[rest])
            xx2 = np.minimum(x2[i], x2[rest])
            yy2 = np.minimum(y2[i], y2[rest])
            w = (xx2 - xx1).clip(min=0)
            h = (yy2 - yy1).clip(min=0)
            inter = w * h
            iou = inter / (areas[i] + areas[rest] - inter + 1e-7)
            order = rest[iou <= iou_thres]
        return np.asarray(keep, dtype=np.int64)

    def postprocess(
        self,
        raws: Sequence[np.ndarray],
        proto: np.ndarray | None = None,
    ) -> list[dict]:
        """Run the full CPU decode + NMS pipeline.

        Parameters
        ----------
        raws : list of (bs, na*no, ny, nx) arrays, one per detection level.
        proto : optional (bs, nm, Hp, Wp) mask protos (segmentation only).

        Returns
        -------
        list of per-image dicts with keys:
            - 'boxes':  (k, 4) float32, xyxy in input-image pixels
            - 'scores': (k,)   float32
            - 'classes':(k,)   int64
            - 'masks':  (k, nm) float32 mask coefficients (if nm > 0)
            - 'proto':  (nm, Hp, Wp) if proto was supplied
        """
        decoded = self.decode(raws)  # (bs, N, no)
        bs = decoded.shape[0]
        results: list[dict] = []
        for b in range(bs):
            pred = decoded[b]
            obj = pred[:, 4]
            # Multiply obj * class probs; take best class per anchor. Matches
            # utils.general.non_max_suppression in "best-class" mode which is
            # what YOLOv7 uses at inference.
            cls_scores = pred[:, 5:5 + self.nc] * obj[:, None]
            cls_ids = cls_scores.argmax(axis=1)
            scores = cls_scores[np.arange(cls_scores.shape[0]), cls_ids]

            mask = scores > self.conf_thres
            if not mask.any():
                results.append(self._empty_result(proto, b))
                continue
            pred = pred[mask]
            scores = scores[mask]
            cls_ids = cls_ids[mask]

            if scores.shape[0] > self.max_nms:
                top = np.argpartition(-scores, self.max_nms)[:self.max_nms]
                pred, scores, cls_ids = pred[top], scores[top], cls_ids[top]

            boxes = self._xywh_to_xyxy(pred[:, :4])
            # Class-aware NMS: offset boxes by class id so NMS is per-class
            # in one pass (same trick torchvision.ops.batched_nms uses).
            max_wh = 7680.0
            offsets = cls_ids.astype(np.float32) * max_wh
            nms_boxes = boxes + offsets[:, None]
            keep = self._nms(nms_boxes, scores, self.iou_thres)
            if keep.size > self.max_det:
                keep = keep[:self.max_det]

            out = {
                'boxes': boxes[keep].astype(np.float32, copy=False),
                'scores': scores[keep].astype(np.float32, copy=False),
                'classes': cls_ids[keep].astype(np.int64, copy=False),
            }
            if self.nm > 0:
                out['masks'] = pred[keep, 5 + self.nc:5 + self.nc + self.nm].astype(
                    np.float32, copy=False)
            if proto is not None:
                out['proto'] = proto[b]
            results.append(out)
        return results

    def _empty_result(self, proto: np.ndarray | None, b: int) -> dict:
        out: dict = {
            'boxes': np.zeros((0, 4), dtype=np.float32),
            'scores': np.zeros((0,), dtype=np.float32),
            'classes': np.zeros((0,), dtype=np.int64),
        }
        if self.nm > 0:
            out['masks'] = np.zeros((0, self.nm), dtype=np.float32)
        if proto is not None:
            out['proto'] = proto[b]
        return out

    # ------------------------------------------------------------------
    # Optional: combine mask coeffs with proto to produce HxW masks.
    # Do this only for surviving detections - it is the most expensive
    # step (matmul + sigmoid over Hp*Wp per detection).
    # ------------------------------------------------------------------
    @staticmethod
    def build_masks(mask_coeffs: np.ndarray, proto: np.ndarray) -> np.ndarray:
        """masks = sigmoid(mask_coeffs @ proto_flat).reshape(k, Hp, Wp)."""
        if mask_coeffs.size == 0:
            _, hp, wp = proto.shape
            return np.zeros((0, hp, wp), dtype=np.float32)
        nm, hp, wp = proto.shape
        m = mask_coeffs @ proto.reshape(nm, hp * wp)
        m = _sigmoid(m)
        return m.reshape(-1, hp, wp)


__all__ = ['DLAPostprocessor', 'DEFAULT_ANCHORS', 'DEFAULT_STRIDES']
