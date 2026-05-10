<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:200122,50:6f0000,100:F2994A&height=220&section=header&text=Triton-attention-kernels&fontSize=46&fontColor=ffffff&fontAlignY=38&desc=Fused%20GPU%20kernels%20for%20attention%2C%20RMSNorm%2C%20SwiGLU%2C%20RoPE&descAlignY=62&descSize=16&animation=fadeIn" width="100%"/>

<a href="https://github.com/Umarfarook1/Triton-attention-kernels">
  <img src="https://readme-typing-svg.demolab.com?font=JetBrains+Mono&weight=600&size=22&duration=2800&pause=600&color=F2994A&center=true&vCenter=true&width=900&lines=Hand-written+Triton+kernels+for+the+transformer+hot+path.;FlashAttention-style+tiling+%C2%B7+online+softmax+%C2%B7+KV-cache.;Benchmarked+against+PyTorch+SDPA+on+H100+%2F+A100.;Each+kernel+ships+with+ablations+and+a+writeup." alt="Typing SVG" />
</a>

<br/>

<p>
  <img src="https://img.shields.io/badge/status-WIP-F5A623?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/tier-Advanced-D7263D?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Triton-2.x-FF6F00?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/CUDA-12.x-76B900?logo=nvidia&logoColor=white&style=for-the-badge"/>
  <img src="https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white&style=for-the-badge"/>
  <img src="https://img.shields.io/badge/license-MIT-1f6feb?style=for-the-badge"/>
</p>

<sub><i>If you can't beat <code>F.scaled_dot_product_attention</code>, you don't really understand attention. This repo is the receipts.</i></sub>

</div>

---

## Why this repo exists

PyTorch's fused-attention kernel is fast · but it's a black box. The point of this repo is to rebuild the transformer hot path from first principles in Triton: attention, RMSNorm, SwiGLU, RoPE · each as an independently benchmarked kernel with ablations on tile size, num_warps, and pipeline depth, plus a writeup of *why* each choice matters.

> **Status:** kernel scaffolding + benchmark harness. First milestone: forward attention within 5% of `F.scaled_dot_product_attention` on H100.

## Kernel zoo

| Kernel | Forward | Backward | Status | Notes |
|---|:-:|:-:|---|---|
| Fused causal attention | ✅ planned | ⏳ | scaffolding | online softmax + KV tiling |
| Paged-KV decode attention | ⏳ | n/a | planned | for serving |
| RMSNorm | ✅ planned | ⏳ | scaffolding | fused gain |
| SwiGLU | ✅ planned | ⏳ | scaffolding | fused with linear projection |
| RoPE (apply + cache) | ✅ planned | n/a | scaffolding | dim-split layout |
| Cross-entropy + chunked logits | ⏳ | ⏳ | planned | memory-saver à la Liger |

## Tiling intuition

```mermaid
flowchart LR
    Q[Q tile<br/>BLOCK_M x D] --> M[Q · K^T]
    K[K tile<br/>BLOCK_N x D] --> M
    M --> S[scale + causal mask]
    S --> SM[online softmax<br/>m, l updated]
    V[V tile<br/>BLOCK_N x D] --> AV[A · V]
    SM --> AV
    AV --> O[Output tile<br/>BLOCK_M x D]
    O -.-> NEXTKV[next KV tile]
```

Each Q tile scans all KV tiles, accumulating the softmax statistics `(m, l)` and the partial output online · i.e., never materializing the full `BLOCK_M × N` attention matrix. This is the FlashAttention insight, restated in Triton.

## Benchmark methodology

- **Hardware:** H100 80GB SXM, A100 40GB PCIe (separate tables per device)
- **Baseline:** `torch.nn.functional.scaled_dot_product_attention` (Flash backend), `torch.nn.RMSNorm`, `torch.nn.SiLU + linear` for SwiGLU
- **Workloads:** sweep over `(B, H, S, D)` covering common transformer shapes (S ∈ {512, 2k, 8k, 32k}, D ∈ {64, 128})
- **Metric:** TFLOPS (compute-bound) and ms/iter (memory-bound), median of 100 runs after warmup
- **Numerical check:** max-abs and mean-abs vs reference, must be within fp16 noise

## Quickstart <sub><i>(coming soon)</i></sub>

```bash
# install
uv pip install -e .

# run a benchmark sweep
uv run bench/attention.py --device cuda --shapes configs/h100.yaml --out reports/h100.md

# correctness suite
uv run pytest tests/

# ablation: tile size for attention forward
uv run bench/attention.py --ablate BLOCK_M:64,128,256 --ablate BLOCK_N:64,128
```

## Headline numbers <sub><i>(populated as kernels land)</i></sub>

| Kernel | Shape | torch SDPA (ms) | this repo (ms) | speedup |
|---|---|---|---|---|
| Causal attn fwd | (4, 16, 4096, 128) | TBD | TBD | TBD |
| Causal attn bwd | (4, 16, 4096, 128) | TBD | TBD | TBD |
| RMSNorm fwd | (B=4, S=4096, D=4096) | TBD | TBD | TBD |
| SwiGLU fwd | (B=4, S=4096, D=11008) | TBD | TBD | TBD |

## Roadmap

- [ ] Triton scaffolding + autotune + correctness harness
- [ ] Fused causal attention forward (online softmax)
- [ ] Fused causal attention backward
- [ ] RMSNorm forward + backward
- [ ] SwiGLU fused projection
- [ ] RoPE apply + cache write
- [ ] Cross-entropy with chunked logits (memory saver)
- [ ] H100 + A100 benchmark report (Markdown + plots)
- [ ] Companion blog post on tile-size selection
- [ ] Optional: integrate into a fork of `nanoGPT` for end-to-end speedup numbers

## Inspiration & required reading

- [GPU MODE · lectures & community](https://github.com/gpu-mode/lectures)
- [ScalingIntelligence/KernelBench](https://github.com/ScalingIntelligence/KernelBench) · kernel benchmarking methodology
- [linkedin/Liger-Kernel](https://github.com/linkedin/Liger-Kernel) · production-quality fused training kernels
- [Triton tutorials](https://triton-lang.org/main/getting-started/tutorials/index.html)
- [Dao-AILab/flash-attention](https://github.com/Dao-AILab/flash-attention) · the paper that started it
- [pytorch/ao](https://github.com/pytorch/ao) · quantization + fused-op patterns

---

<div align="center">
<img src="https://capsule-render.vercel.app/api?type=waving&color=0:200122,50:6f0000,100:F2994A&height=110&section=footer&fontColor=ffffff" width="100%"/>
</div>
