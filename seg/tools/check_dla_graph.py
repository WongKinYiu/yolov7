#!/usr/bin/env python3

import argparse
from collections import Counter, defaultdict

import onnx


FORBIDDEN_OPS = {
    "NonMaxSuppression",
    "TopK",
    "GatherND",
    "GridSample",
    "RoiAlign",
    "ScatterND",
    "Pow",
    "Slice",
    "Split",
    "Transpose",
    "Reshape",
}


def get_attr_ints(node, name):
    for attr in node.attribute:
        if attr.name == name:
            return list(attr.ints)
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("onnx")
    args = parser.parse_args()

    model = onnx.load(args.onnx)

    try:
        model = onnx.shape_inference.infer_shapes(model)
    except Exception as e:
        print(f"WARNING: shape inference failed: {e}")

    blockers = defaultdict(list)
    op_counts = Counter(n.op_type for n in model.graph.node)

    for node in model.graph.node:
        name = node.name or "<unnamed>"

        if node.op_type in FORBIDDEN_OPS:
            blockers[node.op_type].append(name)

        if node.op_type == "MaxPool":
            k = get_attr_ints(node, "kernel_shape")
            if k and any(v > 8 for v in k):
                blockers["MaxPool kernel > 8"].append(f"{name}: kernel={k}")

    print("ONNX DLA graph check")
    print("====================")
    print(f"File: {args.onnx}")

    print("\nOutputs:")
    for out in model.graph.output:
        dims = []
        for d in out.type.tensor_type.shape.dim:
            dims.append(d.dim_value if d.dim_value else d.dim_param or "?")
        print(f"  {out.name}: {dims}")

    print("\nOps:")
    for op, count in sorted(op_counts.items()):
        print(f"  {op}: {count}")

    print("\nBlockers:")
    if not blockers:
        print("  none")
    else:
        for kind, items in blockers.items():
            print(f"  {kind}: {len(items)}")
            for item in items[:20]:
                print(f"    - {item}")
            if len(items) > 20:
                print(f"    ... {len(items) - 20} more")

    raise SystemExit(1 if blockers else 0)


if __name__ == "__main__":
    main()