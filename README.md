# Model Export & Optimization Benchmark CLI

A command-line tool that takes any Hugging Face model, exports it to ONNX with dynamic
shape support, quantizes it to INT8, and benchmarks all three variants — PyTorch-FP32,
ONNX-FP32, ONNX-INT8 — for size, latency, and throughput across multiple batch sizes.

## Why this exists

Before deploying a model to production, an inference team needs a real, *measured* answer
to: how much smaller and faster can we make this model, and what does it cost in accuracy?
This tool automates that entire evaluation — export, quantize, validate, and benchmark —
for any Hugging Face model in one command, instead of a one-off notebook per model.

## How it works

1. **Load** — any Hugging Face model + tokenizer via `AutoModel`/`AutoTokenizer`
2. **Export** — to ONNX using PyTorch's `dynamo=True` exporter, with dynamic batch size
   and sequence length (`torch.export.Dim`) so the graph isn't locked to one input shape
3. **Quantize** — the ONNX graph to INT8 weights via ONNX Runtime's dynamic quantization
4. **Validate** — every exported/quantized variant's output against the original PyTorch
   model with a tolerance check, so speed/size gains are never reported without confirming
   correctness held up
5. **Benchmark** — all three variants across configurable batch sizes, reporting mean/P50/P99
   latency and throughput
6. **Report** — a comparison table, a latency-vs-batch-size chart, and a raw JSON results file

## Usage

\`\`\`bash
uv run benchmark_cli.py --model distilbert-base-uncased --batch-sizes 1,4,8,16
uv run benchmark_cli.py --model google/bert_uncased_L-2_H-128_A-2 --batch-sizes 1,4,8,16
\`\`\`

## Results — distilbert-base-uncased

| Variant       | Size (MB) |
|---------------|----------:|
| PyTorch-FP32  |  253.19   |
| ONNX-FP32     | 253.76    |
| ONNX-INT8     | 64.01     |

| Batch | PyTorch-FP32 (ms) | ONNX-FP32 (ms) | ONNX-INT8 (ms) |
|------:|-------------------:|-----------------:|-----------------:|
| 1     | 22.206              | 11.884            | 3.801             |
| 4     | 30.457              | 24.218            | 11.189            |
| 8     | 43.865              | 30.051            | 16.856            |
| 16    | 77.080              | 58.361            | 32.935            |
| 32    | 135.566             | 122.686           | 82.563            |

![Latency vs Batch Size](benchmark_output/distilbert-base-uncased_latency_chart.png)

## Key findings
- Biggest speedup observed: **82.9% on INT8 BatchSize:1**
- Size reduction from INT8 quantization: *74.77%*
- Something that surprised me: *ONNX-INT8 reduced DistilBERT's size by ~75% and reduced batch-1 latency by ~83% vs PyTorch-FP32. However, throughput peaked around batch 16 (~486 samples/s) and dropped at batch 32, suggesting the workload had reached a CPU/runtime bottleneck. ONNX-FP32 showed a similar throughput drop at batch 32.*

## Tested on
Also validated against `google/bert_uncased_L-2_H-128_A-2` (and initially
`prajjwal1/bert-tiny`, which surfaced a real tokenizer-dependency bug — see Limitations)
to confirm the CLI genuinely generalizes rather than being hardcoded to one model.

## Tech stack
PyTorch · ONNX · ONNX Runtime · Hugging Face Transformers · NumPy · Matplotlib

## What I'd improve next
- GPU execution provider comparison (`CUDAExecutionProvider`)
- Static quantization with a calibration dataset, as a 4th comparison column
- Config-file support to benchmark an entire model zoo in one run
- Unit tests for the validation logic

## Limitations
- ONNX Runtime's dynamic INT8 quantization primarily targets CPU inference — no GPU numbers here
- Not every Hugging Face checkpoint ships a "fast" tokenizer out of the box; some require
  `sentencepiece` for slow-to-fast conversion (discovered while testing on a second model)
- Benchmarks reflect this specific machine (RTX 3050 4GB laptop) — absolute numbers will
  differ on other hardware; relative comparisons should hold directionally