# Research

## Extending Marian-ONNX-Converter for Sockeye

The [Marian-ONNX-Converter](../Marian-ONNX-Converter) submodule currently converts **Hugging Face Marian** models into split ONNX graphs (`encoder.onnx` + `decoder.onnx`) and runs greedy decoding in Python.

To support **Sockeye models** and deploy on **RKNN** (Rockchip NPU) two adaptations are required:

1. `Sockeye -> ONNX` export (submodule)
2. `ONNX -> RKNN` compilation (this repo)

### Add a Sockeye export path

Create a new exporter module (for example `core/sockeye.py`) that mirrors `core/utils.py` behavior but loads Sockeye checkpoints.

Overall flow:

- Load Sockeye model + vocabulary
- Build lightweight `torch.nn.Module` wrappers for encoder and decoder (same idea as `MarianEncoder` / `MarianDecoder`)
- Export split ONNX graphs with fixed input dtypes (e.g. `int32`) and the same `opset_version=14` as the current implementation

### Add an RKNN runtime backend

In addition to the conversion process, Marian-ONNX-Converter also provides an inference runtime.

We need to add a parallel runtime class (for example `core/marian_rknn.py`) that:

- Loads `encoder.rknn` and `decoder.rknn` with RKNN runtime APIs.
- Reuses the same greedy-search loop and logits postprocessing.
- Preserves `int32` token IDs and attention masks.
- Keeps matrix projection (`F.linear` with `lm_weight.bin` / `lm_bias.bin`) on CPU unless you fuse it into decoder export.

## Extending Local Scripts for Sockeye

Introduce a new script alongside [convert.py](../scripts/convert.py) (e.g. `convert_sockeye_to_rknn.py`) that consumes exported ONNX files.

High-level skeleton:

```python
from rknn.api import RKNN


def build_rknn(onnx_path, out_path, target="rk3588"):
    rknn = RKNN(verbose=True)
    rknn.config(
        target_platform=target,
        quantized_dtype="asymmetric_quantized-8",
        optimization_level=3,
    )

    ret = rknn.load_onnx(model=onnx_path)
    assert ret == 0

    # dataset.txt should point to calibration samples
    ret = rknn.build(do_quantization=True, dataset="dataset.txt")
    assert ret == 0

    ret = rknn.export_rknn(out_path)
    assert ret == 0
    rknn.release()
```

Compile both files:

- `encoder.onnx -> encoder.rknn`
- `decoder.onnx -> decoder.rknn`

### Alternative: Extend CLI and model format

Update `convert.py` to support a backend selector:

- `--source {hf,sockeye}`
- `--rknn-target rk3588`

### RKNN-focused constraints to handle early

- **Static shape strategy**: choose max source/target lengths (e.g., 64/128) for deployment profile.
- **Operator support**: avoid uncommon ONNX ops; validate exported graph with RKNN Toolkit early.
- **Quantization quality**: use domain-specific calibration text in `dataset.txt`.
- **Decoder cache**: start without KV cache (simpler), then optimize later.

### Validation plan

Add tests/checks in stages:

1. Export smoke test for Sockeye -> ONNX.
2. RKNN build smoke test (`load_onnx`, `build`, `export_rknn`).
3. End-to-end translation parity against Sockeye baseline on a small sentence set.
4. Latency benchmark on target board (RK356x/RK3588).

### Minimal incremental implementation order

1. Add Sockeye model loader + ONNX exporter.
2. Verify ONNX inference with existing greedy loop.
3. Add RKNN compiler script.
4. Add RKNN runtime class.
5. Wire CLI flags and docs.
