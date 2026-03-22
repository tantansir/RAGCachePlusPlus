"""Configuration for RAGCache++."""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class CacheConfig:
    """Cache system configuration."""

    # Multi-tier capacity (number of KV cache blocks)
    gpu_cache_capacity: int = 1000
    host_cache_capacity: int = 5000

    # PGDSF policy parameters
    pgdsf_age_decay: float = 0.95  # per-access decay for age penalty
    pgdsf_size_weight: float = 1.0  # weight for log(size) term

    # Spatial-aware extensions
    spatial_lambda: float = 0.3  # weight for spatial locality bonus in PGDSF
    admission_tau: float = 0.1  # region utility threshold for admission
    admission_k_hot: int = 3  # top-k rank threshold (admit regardless of region)

    # Prefetch parameters
    prefetch_budget: int = 5  # max chunks to prefetch per query
    geohash_precision: int = 6  # geohash precision level (~1.2km cells)

    # Spatial utility decay
    region_decay_rate: float = 0.99  # slower decay than individual docs
    doc_decay_rate: float = 0.95

    # Selective recomputation
    recompute_budget: float = 0.15  # max fraction of tokens recomputed per layer
    recompute_deviation_threshold: float = 0.1  # KV deviation threshold delta


@dataclass
class ServingConfig:
    """LLM serving configuration."""

    model_name: str = "meta-llama/Llama-2-7b-hf"
    max_model_len: int = 4096
    gpu_memory_utilization: float = 0.9
    block_size: int = 16  # vLLM block size for PagedAttention
    num_gpu_blocks: Optional[int] = None  # auto-detect if None
    num_cpu_blocks: Optional[int] = None


@dataclass
class RetrievalConfig:
    """Retrieval system configuration."""

    embedding_model: str = "text-embedding-3-small"
    embedding_dim: int = 1536
    top_k: int = 5
    chunk_size: int = 512  # tokens per chunk
    index_type: str = "IVF"  # IVF or HNSW
    nprobe: int = 32  # IVF search parameter
    ef_search: int = 64  # HNSW search parameter

    # Spatial retrieval
    spatial_filter_enabled: bool = True
    default_spatial_radius_m: float = 1000.0  # 1km default


@dataclass
class BenchmarkConfig:
    """Benchmark configuration."""

    # Workload
    num_queries: int = 1000
    concurrency_levels: list[int] = field(default_factory=lambda: [1, 4, 8, 16, 32])
    arrival_process: str = "poisson"  # poisson or closed-loop
    arrival_rate: float = 10.0  # requests/sec for poisson

    # Trace protocol
    hotspot_fraction: float = 0.6  # 60% from hotspot regions
    num_hotspot_regions: int = 3
    phase_length: int = 1000  # queries per demand phase
    top_k_choices: list[int] = field(default_factory=lambda: [3, 5, 10])
    chunk_length_range: tuple[int, int] = (64, 512)
    retrieval_noise_fraction: float = 0.1

    # Metrics
    warmup_queries: int = 100
    report_percentiles: list[float] = field(default_factory=lambda: [50, 95, 99])


@dataclass
class RAGCachePPConfig:
    """Top-level configuration."""

    cache: CacheConfig = field(default_factory=CacheConfig)
    serving: ServingConfig = field(default_factory=ServingConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    benchmark: BenchmarkConfig = field(default_factory=BenchmarkConfig)

    # System mode
    enable_spatial_policies: bool = True
    enable_selective_recompute: bool = True
    enable_speculative_pipelining: bool = True
