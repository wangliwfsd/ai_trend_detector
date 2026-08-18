from __future__ import annotations

from datetime import datetime, timedelta, timezone

from .models import Item

SAMPLE_AS_OF = datetime(2026, 8, 18, 4, 0, tzinfo=timezone.utc)

SAMPLE_ROWS = [
    (1, "arXiv", "paper", "SpecForge: Training Draft Models for Adaptive Speculative Decoding", "Speculative decoding with adaptive draft selection improves LLM inference throughput."),
    (2, "Hugging Face Papers", "paper", "TreeSpec: Parallel Token Verification without Extra KV Cache", "Parallel decoding verifies tree candidates while reducing KV cache pressure."),
    (5, "PyTorch Blog", "blog", "Production speculative decoding with compiled PyTorch kernels", "GPU kernel fusion makes speculative decoding practical for production serving."),
    (12, "arXiv", "paper", "Dynamic Lookahead for Lossless LLM Acceleration", "A draft model changes speculative lookahead based on acceptance rate."),
    (24, "arXiv", "paper", "Survey of Multi-token Prediction for Fast Language Models", "A review of parallel decoding and speculative inference methods."),
    (1, "GitHub Releases", "release", "vllm-project/vllm v0.15 adds disaggregated serving", "The release separates prefill and decode workers for distributed inference."),
    (3, "GitHub Trending", "repository", "sgl-project/sglang", "Fast LLM serving with prefix caching and structured generation."),
    (4, "arXiv", "paper", "FlowServe: Autoscaling Disaggregated LLM Inference", "Serving scheduler balances prefill and decode latency across GPU workers."),
    (9, "PyTorch Blog", "blog", "Reducing LLM serving tail latency with torch.compile", "Production inference benchmarks show higher throughput and lower P99 latency."),
    (20, "arXiv", "paper", "Queue-aware Scheduling for Large Model Serving", "A scheduling system improves serving throughput under bursty traffic."),
    (2, "arXiv", "paper", "CacheWeaver: Tiered KV Cache for Million-token Context", "KV cache pages move across GPU, CPU and NVMe for long-context inference."),
    (4, "Hugging Face Papers", "paper", "QuestKV: Query-aware KV Cache Quantization", "Low-bit KV cache compression preserves long-context quality."),
    (8, "NVIDIA Developer AI", "blog", "Optimizing paged attention for Blackwell GPUs", "CUDA kernels reduce memory bandwidth in paged attention and KV cache access."),
    (18, "arXiv", "paper", "ContextZip: Learned Attention Cache Eviction", "A KV cache eviction policy enables longer context windows."),
    (2, "arXiv", "paper", "BitMoE: Expert-aware FP4 Quantization", "Quantization of sparse MoE experts cuts memory and inference cost."),
    (6, "Hugging Face Papers", "paper", "QServe-MoE: Low-bit Expert Parallel Inference", "Distributed inference combines quantization and expert routing."),
    (11, "GitHub Releases", "release", "pytorch/ao v0.16 adds float8 MoE recipes", "Training efficiency and inference quantization recipes for mixture of experts."),
    (26, "arXiv", "paper", "SparseRoute: Communication-efficient MoE Training", "Expert routing reduces all-to-all communication in distributed training."),
    (1, "arXiv", "paper", "AgentMesh: Durable Runtime for Long-running Tool Agents", "Agent infrastructure adds checkpoints, permissions and tool execution tracing."),
    (3, "GitHub Trending", "repository", "agent-runtime/mesh", "A production agent runtime with sandboxed tools and durable workflows."),
    (7, "Hugging Face Papers", "paper", "TraceRL: Learning Reliable Agent Tool Use", "Reinforcement learning uses execution traces to improve agent reliability."),
    (21, "arXiv", "paper", "Memory Protocols for Multi-agent Systems", "A protocol for shared memory and coordination among agents."),
]


def sample_items() -> list[Item]:
    result: list[Item] = []
    for index, (age, source, kind, title, summary) in enumerate(SAMPLE_ROWS, 1):
        slug = title.lower().replace(" ", "-")[:50]
        metrics = {"upvotes": float(60 - age)} if source == "Hugging Face Papers" else {}
        result.append(
            Item(
                uid=f"sample:{index}",
                source=source,
                kind=kind,
                title=title,
                url=f"https://example.com/{index}/{slug}",
                published_at=SAMPLE_AS_OF - timedelta(days=age),
                summary=summary,
                metrics=metrics,
            )
        )
    return result

