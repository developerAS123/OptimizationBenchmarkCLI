"""
Model Export & Optimization Benchmark CLI

Takes any Hugging Face model name, exports it to ONNX (dynamic shapes),
quantizes it to INT8, and prints a full comparison table:
PyTorch-FP32 vs ONNX-FP32 vs ONNX-INT8 — size, latency, throughput.

Usage:
    python benchmark_cli.py --model distilbert-base-uncased --batch-sizes 1,8,16
"""
import argparse
import os
import time
import json

import numpy as np
import torch
from torch.export import Dim
from transformers import AutoTokenizer, AutoModel
import onnx
import onnxruntime as ort
from onnxruntime.quantization import quantize_dynamic, QuantType


def get_file_size_mb(path):
    return os.path.getsize(path) / (1024 * 1024)


def load_model(model_name):
    print(f"Loading {model_name}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    except ValueError as e:
        if "sentencepiece" in str(e).lower():
            raise RuntimeError(
                f"'{model_name}' needs a slow-to-fast tokenizer conversion. "
                f"Run: uv add sentencepiece"
            ) from e
        raise
    model = AutoModel.from_pretrained(model_name)
    model.eval()
    return tokenizer, model


def export_onnx(model, tokenizer, sample_text, onnx_path, max_batch=64, max_seq=128):
    inputs = tokenizer(sample_text, return_tensors="pt")
    batch_dim = Dim("batch_size", min=1, max=max_batch)
    seq_dim = Dim("seq_len", min=1, max=max_seq)
    onnx_program = torch.onnx.export(
        model,
        (inputs["input_ids"], inputs["attention_mask"]),
        dynamo=True,
        input_names=["input_ids", "attention_mask"],
        output_names=["last_hidden_state"],
        dynamic_shapes=({0: batch_dim, 1: seq_dim}, {0: batch_dim, 1: seq_dim}),
    )
    onnx_program.save(onnx_path)
    onnx.checker.check_model(onnx.load(onnx_path))
    print(f"  Exported and validated: {onnx_path}")


def quantize_onnx(onnx_fp32_path, onnx_int8_path):
    quantize_dynamic(model_input=onnx_fp32_path, model_output=onnx_int8_path, weight_type=QuantType.QInt8)
    print(f"  Quantized: {onnx_int8_path}")


def validate_outputs(pytorch_model, onnx_path, tokenizer, sample_text, tol, label):
    inputs = tokenizer(sample_text, return_tensors="pt")
    with torch.no_grad():
        torch_out = pytorch_model(**inputs).last_hidden_state.numpy()
    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    onnx_out = session.run(None, {
        "input_ids": inputs["input_ids"].numpy(),
        "attention_mask": inputs["attention_mask"].numpy(),
    })[0]
    max_diff = float(np.max(np.abs(torch_out - onnx_out)))
    passed = max_diff < tol
    print(f"  [{label}] max abs diff vs PyTorch-FP32: {max_diff:.5f}  {'PASS' if passed else 'CHECK MANUALLY'}")
    return max_diff


def _percentile_stats(latencies_ms, batch_size):
    latencies_ms = np.array(latencies_ms)
    return {
        "mean_ms": float(latencies_ms.mean()),
        "p50_ms": float(np.percentile(latencies_ms, 50)),
        "p99_ms": float(np.percentile(latencies_ms, 99)),
        "throughput": float(batch_size / (latencies_ms.mean() / 1000)),
    }


def bench_pytorch(model, tokenizer, sample_text, batch_sizes, n_runs, n_warmup):
    results = {}
    with torch.no_grad():
        for bs in batch_sizes:
            inputs = tokenizer([sample_text] * bs, return_tensors="pt", padding=True, truncation=True)
            for _ in range(n_warmup):
                _ = model(**inputs)
            latencies = []
            for _ in range(n_runs):
                start = time.time()
                _ = model(**inputs)
                latencies.append((time.time() - start) * 1000)
            results[bs] = _percentile_stats(latencies, bs)
    return results


def bench_onnxruntime(onnx_path, tokenizer, sample_text, batch_sizes, n_runs, n_warmup):
    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    results = {}
    for bs in batch_sizes:
        inputs = tokenizer([sample_text] * bs, return_tensors="pt", padding=True, truncation=True)
        feed = {"input_ids": inputs["input_ids"].numpy(), "attention_mask": inputs["attention_mask"].numpy()}
        for _ in range(n_warmup):
            _ = session.run(None, feed)
        latencies = []
        for _ in range(n_runs):
            start = time.time()
            _ = session.run(None, feed)
            latencies.append((time.time() - start) * 1000)
        results[bs] = _percentile_stats(latencies, bs)
    return results


def print_comparison_table(all_results, sizes, batch_sizes):
    print("\n" + "=" * 80)
    print("MODEL SIZE")
    print("=" * 80)
    for label, size_mb in sizes.items():
        print(f"  {label:<20}: {size_mb:>8.2f} MB")

    print("\n" + "=" * 80)
    print("LATENCY (mean ms) & THROUGHPUT (samples/sec)")
    print("=" * 80)
    header = f"{'Batch':<8}"
    for label in all_results:
        header += f"{label + ' ms':<16}{label + ' smp/s':<16}"
    print(header)
    for bs in batch_sizes:
        row = f"{bs:<8}"
        for results in all_results.values():
            row += f"{results[bs]['mean_ms']:<16.3f}{results[bs]['throughput']:<16.1f}"
        print(row)

    print("\n" + "=" * 80)
    print("SPEEDUP vs PyTorch-FP32 (mean latency)")
    print("=" * 80)
    baseline = all_results["PyTorch-FP32"]
    for label, results in all_results.items():
        if label == "PyTorch-FP32":
            continue
        print(f"  {label}:")
        for bs in batch_sizes:
            speedup = (baseline[bs]["mean_ms"] - results[bs]["mean_ms"]) / baseline[bs]["mean_ms"] * 100
            print(f"    batch={bs:<4}: {speedup:+.1f}%")


def plot_results(all_results, batch_sizes, output_path):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 6))
    for label, results in all_results.items():
        ys = [results[bs]["mean_ms"] for bs in batch_sizes]
        plt.plot(batch_sizes, ys, marker="o", label=label)
    plt.xlabel("Batch size")
    plt.ylabel("Mean latency (ms)")
    plt.title("Latency vs Batch Size — PyTorch-FP32 vs ONNX-FP32 vs ONNX-INT8")
    plt.legend()
    plt.grid(True)
    plt.savefig(output_path)
    print(f"\nSaved chart: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Model Export & Optimization Benchmark CLI")
    parser.add_argument("--model", type=str, default="distilbert-base-uncased")
    parser.add_argument("--batch-sizes", type=str, default="1,8,16")
    parser.add_argument("--n-runs", type=int, default=50)
    parser.add_argument("--n-warmup", type=int, default=10)
    parser.add_argument("--sample-text", type=str,
                         default="AI inference engineers optimize models to run fast and cheap in production.")
    parser.add_argument("--output-dir", type=str, default="./benchmark_output")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    batch_sizes = [int(b) for b in args.batch_sizes.split(",")]
    safe_name = args.model.replace("/", "_")

    onnx_fp32_path = os.path.join(args.output_dir, f"{safe_name}_fp32.onnx")
    onnx_int8_path = os.path.join(args.output_dir, f"{safe_name}_int8.onnx")
    pt_path = os.path.join(args.output_dir, f"{safe_name}_fp32.pt")

    tokenizer, model = load_model(args.model)

    print("\nExporting to ONNX (dynamic batch + sequence length)...")
    export_onnx(model, tokenizer, args.sample_text, onnx_fp32_path)

    print("\nQuantizing ONNX model to INT8...")
    quantize_onnx(onnx_fp32_path, onnx_int8_path)

    print("\nValidating outputs against PyTorch-FP32 baseline...")
    validate_outputs(model, onnx_fp32_path, tokenizer, args.sample_text, tol=1e-3, label="ONNX-FP32")
    validate_outputs(model, onnx_int8_path, tokenizer, args.sample_text, tol=5e-2, label="ONNX-INT8")

    print("\nMeasuring model sizes...")
    torch.save(model.state_dict(), pt_path)
    sizes = {
        "PyTorch-FP32": get_file_size_mb(pt_path),
        "ONNX-FP32": get_file_size_mb(onnx_fp32_path),
        "ONNX-INT8": get_file_size_mb(onnx_int8_path),
    }

    print(f"\nBenchmarking PyTorch-FP32 across batch sizes {batch_sizes}...")
    pt_results = bench_pytorch(model, tokenizer, args.sample_text, batch_sizes, args.n_runs, args.n_warmup)
    print("Benchmarking ONNX-FP32...")
    onnx_fp32_results = bench_onnxruntime(onnx_fp32_path, tokenizer, args.sample_text, batch_sizes, args.n_runs, args.n_warmup)
    print("Benchmarking ONNX-INT8...")
    onnx_int8_results = bench_onnxruntime(onnx_int8_path, tokenizer, args.sample_text, batch_sizes, args.n_runs, args.n_warmup)

    all_results = {"PyTorch-FP32": pt_results, "ONNX-FP32": onnx_fp32_results, "ONNX-INT8": onnx_int8_results}

    print_comparison_table(all_results, sizes, batch_sizes)
    plot_results(all_results, batch_sizes, os.path.join(args.output_dir, f"{safe_name}_latency_chart.png"))

    with open(os.path.join(args.output_dir, f"{safe_name}_results.json"), "w") as f:
        json.dump({"sizes": sizes, "results": all_results}, f, indent=2)
    print(f"Saved raw results: {args.output_dir}/{safe_name}_results.json")


if __name__ == "__main__":
    main()