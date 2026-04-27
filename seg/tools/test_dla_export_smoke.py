"""
Smoke-test: export a randomly-initialised YOLOv7-seg-DLA model to ONNX and
verify that the resulting graph contains no DLA-incompatible operators.

No .pt checkpoint is needed — weights are random, we only care about graph
structure.  Run from the seg/ directory:

    cd /Users/pavelzhdanov/projects/jenoptic/yolov7/seg
    python check_dla_graph.py

Pass:  "ALL CHECKS PASSED" printed, vvd_dla_smoke.onnx written.
Fail:  list of offending nodes printed with FAIL.

DLA op restrictions checked (Jetson Orin NX, TRT 10.x):
  - No 5D tensors (view/permute that change rank beyond 4)
  - No POW operator
  - No Reshape that changes tensor rank
  - No MaxPool with kernel > 8
  - No Concat with any output dim > 8192
  - No Flatten (implies non-4D)
  - No Gemm / MatMul (FC layers)
  - No Softmax
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import torch
import onnx
from onnx import numpy_helper

from models.yolo import IDetect, ISegment, IRSegment, Detect, Segment
from models.common import SPPCSPC_DLA
from utils.general import check_img_size

# ── config ────────────────────────────────────────────────────────────────────
YAML      = 'models/segment/yolov7-seg-dla.yaml'
IMGSZ     = (640, 832)   # H, W  — matches VVD production input
NC        = 2            # dummy class count (vehicle + person)
OPSET     = 12
OUT_ONNX  = 'vvd_dla_smoke.onnx'

# DLA hard limits
MAX_POOL_KERNEL = 8
MAX_CONCAT_DIM  = 8192

# Operators that are never supported on DLA
BANNED_OPS = {
    'Pow',       # (wh*2)**2 anchor decode
    'Flatten',   # implies rank change
    'Gemm',      # FC layers
    'MatMul',    # FC layers
    'Softmax',   # not supported on DLA
}
# ──────────────────────────────────────────────────────────────────────────────


def build_random_model(yaml_path: str, nc: int):
    """Instantiate model from YAML with random weights. No .pt needed."""
    from models.yolo import DetectionModel  # SegmentationModel inherits this
    # Override nc so the head conv channels are correct for our dummy class count
    model = DetectionModel(yaml_path, nc=nc)
    model.eval()
    return model


def set_dla_export_flags(model):
    for _, m in model.named_modules():
        if isinstance(m, (Detect, IDetect, Segment, ISegment, IRSegment)):
            m.inplace  = False
            m.dynamic  = False
            m.export   = True
            if isinstance(m, (IDetect, ISegment, IRSegment)):
                m.export_dla = True


def export_onnx(model, imgsz, opset, out_path):
    gs = int(max(model.stride))
    h  = check_img_size(imgsz[0], gs)
    w  = check_img_size(imgsz[1], gs)
    im = torch.zeros(1, 3, h, w)

    # Dry run to populate grids
    with torch.no_grad():
        model(im)

    torch.onnx.export(
        model, im, out_path,
        verbose=False,
        opset_version=opset,
        dynamo=False,  # avoid torch.export non-leaf parameter bug on fused heads
        training=torch.onnx.TrainingMode.EVAL,
        do_constant_folding=True,
        input_names=['images'],
        output_names=['proto', 'det_p3', 'det_p4', 'det_p5'],
        dynamic_axes=None,  # static shapes — required for DLA
    )
    print(f'  ONNX written → {out_path}')
    return onnx.load(out_path)


def get_initializer_shapes(model_onnx):
    """Return dict name→shape for all constant initializers."""
    return {
        init.name: list(init.dims)
        for init in model_onnx.graph.initializer
    }


def check_graph(model_onnx):
    failures = []
    init_shapes = get_initializer_shapes(model_onnx)

    # Build a quick name→node map for output shape lookup when needed
    # (we use value_info + graph output for shape inference results)
    shape_map = {}
    for vi in list(model_onnx.graph.value_info) + list(model_onnx.graph.output):
        t = vi.type.tensor_type
        if t.HasField('shape'):
            dims = [d.dim_value for d in t.shape.dim]
            shape_map[vi.name] = dims

    for node in model_onnx.graph.node:
        op = node.op_type

        # 1. Banned ops — never on DLA
        if op in BANNED_OPS:
            failures.append(f'BANNED OP  [{op}]  node: {node.name or node.output[0]}')

        # 2. Reshape — flag any that appear (all output reshapes should be gone)
        if op == 'Reshape':
            failures.append(f'RESHAPE    node: {node.name or node.output[0]}  '
                            f'(decode/flatten should be in postprocessor, not graph)')

        # 3. MaxPool kernel > 8
        if op == 'MaxPool':
            for attr in node.attribute:
                if attr.name == 'kernel_shape':
                    bad = [k for k in attr.ints if k > MAX_POOL_KERNEL]
                    if bad:
                        failures.append(
                            f'MAXPOOL kernel {list(attr.ints)} > {MAX_POOL_KERNEL}  '
                            f'node: {node.name or node.output[0]}')

        # 4. Concat output dimension > 8192
        if op == 'Concat':
            out_name = node.output[0]
            shape = shape_map.get(out_name)
            if shape:
                for dim_idx, dim_val in enumerate(shape):
                    if dim_val > MAX_CONCAT_DIM:
                        failures.append(
                            f'CONCAT dim[{dim_idx}]={dim_val} > {MAX_CONCAT_DIM}  '
                            f'node: {node.name or out_name}')

        # 5. Any tensor with rank > 4 in outputs of this node
        for out_name in node.output:
            shape = shape_map.get(out_name)
            if shape and len(shape) > 4:
                failures.append(
                    f'RANK>4     [{op}] output "{out_name}" shape={shape}  '
                    f'node: {node.name or out_name}')

    return failures


def print_output_shapes(model_onnx):
    print('\n  Graph outputs:')
    for out in model_onnx.graph.output:
        t = out.type.tensor_type
        if t.HasField('shape'):
            dims = [d.dim_value for d in t.shape.dim]
            print(f'    {out.name:12s}  {dims}')
        else:
            print(f'    {out.name:12s}  (shape unknown)')


def main():
    print('── Building random model from YAML ──────────────────────────────')
    print(f'  YAML   : {YAML}')
    print(f'  imgsz  : {IMGSZ}')
    model = build_random_model(YAML, NC)
    model.fuse()          # folds BN into Conv — same as production export
    set_dla_export_flags(model)

    print('\n── Exporting to ONNX ────────────────────────────────────────────')
    with torch.no_grad():
        model_onnx = export_onnx(model, IMGSZ, OPSET, OUT_ONNX)

    print_output_shapes(model_onnx)

    print('\n── Checking graph for DLA-incompatible ops ──────────────────────')
    failures = check_graph(model_onnx)

    # Summary
    # Count op types present for reference
    op_counts: dict[str, int] = {}
    for node in model_onnx.graph.node:
        op_counts[node.op_type] = op_counts.get(node.op_type, 0) + 1
    print(f'\n  Op types in graph: { {k: v for k, v in sorted(op_counts.items())} }')

    if failures:
        print(f'\n  FAIL — {len(failures)} issue(s) found:')
        for f in failures:
            print(f'    ✗ {f}')
        sys.exit(1)
    else:
        print('\n  ALL CHECKS PASSED ✓')
        print('  Graph contains no DLA-incompatible operators.')
        print(f'  ONNX saved to: {OUT_ONNX}')


if __name__ == '__main__':
    main()
