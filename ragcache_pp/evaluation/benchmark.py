"""Benchmark runner for RAGCache++ evaluation.

Runs the full evaluation suite:
1. Workload characterization (spatial locality analysis)
2. Factorial comparison (2x2: spatial x recompute)
3. Baseline comparison
4. Systems profiling
"""

from __future__ import annotations

import json
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional

from ragcache_pp.cache.cache_manager import CacheManager
from ragcache_pp.cache.spatial_index import SpatialIndex, encode_geohash, haversine_distance
from ragcache_pp.config import CacheConfig, RAGCachePPConfig
from ragcache_pp.evaluation.workload_generator import (
    GeoSpatialWorkloadGenerator,
    POI,
    SyntheticPOIGenerator,
    WorkloadConfig,
)
from ragcache_pp.serving.rag_controller import RAGController, RAGRequest


def run_workload_characterization(
    pois: list[POI],
    trace: list[RAGRequest],
    precision_levels: list[int] | None = None,
) -> dict:
    """Run upfront workload characterization to verify spatial locality.

    Measures:
    1. Query-query spatial autocorrelation (Jaccard overlap vs distance)
    2. Retrieved-document overlap vs geohash distance
    3. Reuse-distance distributions by region
    """
    if precision_levels is None:
        precision_levels = [4, 5, 6, 7]

    results: dict = {}

    # 1. Jaccard overlap vs spatial distance
    print("  [1/3] Computing query-query overlap vs distance...")
    sample_size = min(200, len(trace))
    sample_indices = list(range(0, len(trace), max(1, len(trace) // sample_size)))[:sample_size]
    sample = [trace[i] for i in sample_indices]

    distance_overlap_pairs: list[tuple[float, float]] = []
    for i in range(len(sample)):
        for j in range(i + 1, min(i + 20, len(sample))):
            qi, qj = sample[i], sample[j]
            if qi.latitude is None or qj.latitude is None:
                continue
            d_spatial = haversine_distance(qi.latitude, qi.longitude, qj.latitude, qj.longitude)
            set_i = set(qi.retrieved_doc_ids)
            set_j = set(qj.retrieved_doc_ids)
            union = set_i | set_j
            jaccard = len(set_i & set_j) / len(union) if union else 0.0
            distance_overlap_pairs.append((d_spatial, jaccard))

    # Bin by distance
    bins = [(0, 200), (200, 500), (500, 1000), (1000, 2000), (2000, 5000), (5000, float("inf"))]
    binned: dict[str, list[float]] = {}
    for label_lo, label_hi in bins:
        label = f"{label_lo}-{label_hi}m"
        overlaps = [j for d, j in distance_overlap_pairs if label_lo <= d < label_hi]
        binned[label] = overlaps

    results["jaccard_vs_distance"] = {
        k: {"count": len(v), "mean_jaccard": sum(v) / len(v) if v else 0.0}
        for k, v in binned.items()
    }

    # 2. Per-precision geohash overlap analysis
    print("  [2/3] Computing geohash overlap at different precisions...")
    precision_results: dict[int, dict] = {}
    for prec in precision_levels:
        same_cell_overlaps: list[float] = []
        adjacent_cell_overlaps: list[float] = []
        distant_overlaps: list[float] = []

        for i in range(len(sample)):
            for j in range(i + 1, min(i + 10, len(sample))):
                qi, qj = sample[i], sample[j]
                if qi.latitude is None or qj.latitude is None:
                    continue
                gh_i = encode_geohash(qi.latitude, qi.longitude, prec)
                gh_j = encode_geohash(qj.latitude, qj.longitude, prec)

                set_i = set(qi.retrieved_doc_ids)
                set_j = set(qj.retrieved_doc_ids)
                union = set_i | set_j
                jaccard = len(set_i & set_j) / len(union) if union else 0.0

                if gh_i == gh_j:
                    same_cell_overlaps.append(jaccard)
                elif gh_i[:prec - 1] == gh_j[:prec - 1]:
                    adjacent_cell_overlaps.append(jaccard)
                else:
                    distant_overlaps.append(jaccard)

        precision_results[prec] = {
            "same_cell": {
                "count": len(same_cell_overlaps),
                "mean_overlap": sum(same_cell_overlaps) / len(same_cell_overlaps) if same_cell_overlaps else 0.0,
            },
            "adjacent_cell": {
                "count": len(adjacent_cell_overlaps),
                "mean_overlap": sum(adjacent_cell_overlaps) / len(adjacent_cell_overlaps) if adjacent_cell_overlaps else 0.0,
            },
            "distant": {
                "count": len(distant_overlaps),
                "mean_overlap": sum(distant_overlaps) / len(distant_overlaps) if distant_overlaps else 0.0,
            },
        }

    results["geohash_precision_analysis"] = precision_results

    # 3. Reuse distance by region
    print("  [3/3] Computing reuse distances by region...")
    doc_access_times: dict[str, list[int]] = defaultdict(list)
    for t, req in enumerate(trace):
        for doc_id in req.retrieved_doc_ids:
            doc_access_times[doc_id].append(t)

    reuse_distances: list[int] = []
    for doc_id, times in doc_access_times.items():
        for i in range(1, len(times)):
            reuse_distances.append(times[i] - times[i - 1])

    results["reuse_distance"] = {
        "total_documents_accessed": len(doc_access_times),
        "documents_accessed_once": sum(1 for v in doc_access_times.values() if len(v) == 1),
        "documents_accessed_multiple": sum(1 for v in doc_access_times.values() if len(v) > 1),
        "mean_reuse_distance": sum(reuse_distances) / len(reuse_distances) if reuse_distances else float("inf"),
        "median_reuse_distance": sorted(reuse_distances)[len(reuse_distances) // 2] if reuse_distances else float("inf"),
        "reuse_distance_p90": sorted(reuse_distances)[int(len(reuse_distances) * 0.9)] if reuse_distances else float("inf"),
    }

    return results


def run_single_config(
    trace: list[RAGRequest],
    pois: list[POI],
    enable_spatial: bool,
    enable_recompute: bool,
    cache_config: Optional[CacheConfig] = None,
    label: str = "",
) -> dict:
    """Run a single configuration and return metrics.

    Used for factorial comparison and baseline experiments.
    """
    if cache_config is None:
        cache_config = CacheConfig()

    config = RAGCachePPConfig(cache=cache_config)
    config.enable_spatial_policies = enable_spatial
    config.enable_selective_recompute = enable_recompute

    # Create spatial index and register POIs
    spatial_index = SpatialIndex(
        precision=cache_config.geohash_precision,
        region_decay_rate=cache_config.region_decay_rate,
    )
    for poi in pois:
        spatial_index.register_document(poi.doc_id, poi.latitude, poi.longitude)

    # Create cache manager
    cache_manager = CacheManager(
        config=cache_config,
        spatial_index=spatial_index if enable_spatial else None,
        enable_spatial=enable_spatial,
        enable_non_prefix_reuse=enable_recompute,
    )

    # Create controller
    controller = RAGController(
        config=config,
        cache_manager=cache_manager,
        spatial_index=spatial_index if enable_spatial else None,
        simulation_mode=True,
    )

    # Process all requests
    print(f"  Running config: {label} ({len(trace)} queries)...")
    t_start = time.time()
    for request in trace:
        controller.process_request(request)
    elapsed = time.time() - t_start

    # Collect results
    agg = controller.get_aggregate_metrics()
    agg["config_label"] = label
    agg["enable_spatial"] = enable_spatial
    agg["enable_recompute"] = enable_recompute
    agg["wall_clock_sec"] = elapsed

    return agg


def run_factorial_comparison(
    trace: list[RAGRequest],
    pois: list[POI],
    cache_config: Optional[CacheConfig] = None,
) -> dict:
    """Run the 2x2 factorial comparison.

    Configs:
    1. RAGCache (reproduced) — no spatial, no recompute
    2. RAGCache + Recompute — no spatial, yes recompute
    3. RAGCache + Spatial — yes spatial, no recompute
    4. RAGCache++ (full) — yes spatial, yes recompute
    """
    print("\n=== 2x2 Factorial Comparison ===")
    results: dict[str, dict] = {}

    configs = [
        (False, False, "RAGCache (baseline)"),
        (False, True, "RAGCache + Recompute"),
        (True, False, "RAGCache + Spatial"),
        (True, True, "RAGCache++ (full)"),
    ]

    for enable_spatial, enable_recompute, label in configs:
        result = run_single_config(
            trace, pois, enable_spatial, enable_recompute, cache_config, label
        )
        results[label] = result

    return results


def run_full_benchmark(output_path: str = "results/benchmark_results.json") -> dict:
    """Run the complete benchmark suite."""
    print("=" * 60)
    print("RAGCache++ Benchmark Suite")
    print("=" * 60)

    # Step 1: Generate POIs and workload
    print("\n[Step 1] Generating synthetic POIs and workload trace...")
    wl_config = WorkloadConfig(num_queries=2000, num_pois=5000, seed=42)
    poi_gen = SyntheticPOIGenerator(wl_config)
    pois = poi_gen.generate_pois()
    print(f"  Generated {len(pois)} POIs")

    wl_gen = GeoSpatialWorkloadGenerator(wl_config, pois)
    trace = wl_gen.generate_trace()
    print(f"  Generated {len(trace)} queries")

    all_results: dict = {"timestamp": time.strftime("%Y-%m-%d %H:%M:%S")}

    # Step 2: Workload characterization
    print("\n[Step 2] Workload Characterization...")
    char_results = run_workload_characterization(pois, trace)
    all_results["workload_characterization"] = char_results
    print(f"  Jaccard overlap (0-200m): {char_results['jaccard_vs_distance'].get('0-200m', {}).get('mean_jaccard', 'N/A'):.3f}")
    print(f"  Jaccard overlap (2000-5000m): {char_results['jaccard_vs_distance'].get('2000-5000m', {}).get('mean_jaccard', 'N/A'):.3f}")
    print(f"  Mean reuse distance: {char_results['reuse_distance']['mean_reuse_distance']:.1f} queries")

    # Step 3: Factorial comparison
    print("\n[Step 3] Factorial Comparison...")
    cache_config = CacheConfig(gpu_cache_capacity=5000, host_cache_capacity=20000)
    factorial_results = run_factorial_comparison(trace, pois, cache_config)
    all_results["factorial_comparison"] = factorial_results

    # Print factorial summary
    print("\n  === Factorial Results Summary ===")
    print(f"  {'Config':<30} {'TTFT p50':>10} {'TTFT p99':>10} {'Hit Rate':>10} {'FLOPs Saved':>12}")
    print(f"  {'-'*30} {'-'*10} {'-'*10} {'-'*10} {'-'*12}")
    for label, res in factorial_results.items():
        print(
            f"  {label:<30} "
            f"{res.get('ttft_p50_ms', 0):>10.1f} "
            f"{res.get('ttft_p99_ms', 0):>10.1f} "
            f"{res.get('cache_stats', {}).get('hit_rate', 0):>10.3f} "
            f"{res.get('avg_flops_saved', 0):>12.3f}"
        )

    # Step 4: True no-locality trace (negative control)
    # Uses random doc retrieval (not nearest-neighbor) to destroy spatial locality.
    # Compares spatial on/off with recompute held fixed to isolate spatial overhead.
    print("\n[Step 4] No-Locality Trace (Negative Control)...")
    noloc_trace = wl_gen.generate_no_locality_trace(500)

    noloc_spatial_on = run_single_config(
        noloc_trace, pois, True, True, cache_config, "RAGCache++ (no-locality)"
    )
    noloc_spatial_off = run_single_config(
        noloc_trace, pois, False, True, cache_config, "Recompute only (no-locality)"
    )

    all_results["negative_control"] = {
        "spatial_on": noloc_spatial_on,
        "spatial_off": noloc_spatial_off,
    }

    overhead_pct = 0.0
    base_ttft = noloc_spatial_off.get("ttft_p50_ms", 1)
    spatial_ttft = noloc_spatial_on.get("ttft_p50_ms", 1)
    if base_ttft > 0:
        overhead_pct = (spatial_ttft - base_ttft) / base_ttft * 100
    print(f"  Spatial overhead under no-locality workload: {overhead_pct:+.1f}%")
    print(f"  Hit rate (spatial on): {noloc_spatial_on.get('cache_stats', {}).get('hit_rate', 0):.4f}")
    print(f"  Hit rate (spatial off): {noloc_spatial_off.get('cache_stats', {}).get('hit_rate', 0):.4f}")

    # Step 5: Systems profiling
    print("\n[Step 5] Memory Profile (from last factorial run)...")
    for label, res in factorial_results.items():
        mem = res.get("memory_profile", {})
        print(
            f"  {label:<30} "
            f"GPU: {mem.get('gpu_utilization', 0):.1%}  "
            f"Host: {mem.get('host_utilization', 0):.1%}  "
            f"Tree nodes: {mem.get('tree_node_count', 0)}"
        )

    # Step 6: Cache-size sensitivity sweep
    print("\n[Step 6] Cache-Size Sensitivity Sweep...")
    size_sweep_results: dict[str, dict] = {}
    for gpu_cap, host_cap in [(1000, 4000), (2500, 10000), (5000, 20000), (10000, 40000)]:
        sweep_config = CacheConfig(gpu_cache_capacity=gpu_cap, host_cache_capacity=host_cap)
        label = f"GPU={gpu_cap} Host={host_cap}"
        res = run_single_config(trace, pois, True, True, sweep_config, label)
        size_sweep_results[label] = res
        cs = res.get("cache_stats", {})
        print(
            f"  {label:<25} "
            f"Hit: {cs.get('hit_rate', 0):.3f}  "
            f"TTFT p50: {res.get('ttft_p50_ms', 0):.1f}ms  "
            f"FLOPs: {res.get('avg_flops_saved', 0):.3f}"
        )
    all_results["cache_size_sweep"] = size_sweep_results

    # Step 7: Recompute-budget sensitivity sweep
    print("\n[Step 7] Recompute-Budget Sensitivity Sweep...")
    budget_sweep_results: dict[str, dict] = {}
    for budget in [0.0, 0.05, 0.15, 0.30, 0.50, 1.0]:
        sweep_config = CacheConfig(
            gpu_cache_capacity=5000, host_cache_capacity=20000,
            recompute_budget=budget,
        )
        label = f"Budget={budget:.0%}"
        res = run_single_config(trace, pois, True, True, sweep_config, label)
        budget_sweep_results[label] = res
        cs = res.get("cache_stats", {})
        print(
            f"  {label:<15} "
            f"Hit: {cs.get('hit_rate', 0):.3f}  "
            f"TTFT p50: {res.get('ttft_p50_ms', 0):.1f}ms  "
            f"FLOPs: {res.get('avg_flops_saved', 0):.3f}  "
            f"Recomp tokens: {res.get('avg_tokens_recomputed', 0):.0f}"
        )
    all_results["recompute_budget_sweep"] = budget_sweep_results

    # Step 8: Time-series hit rate (warmup vs steady state)
    print("\n[Step 8] Time-Series Analysis (warmup vs steady state)...")
    ts_config = RAGCachePPConfig(cache=cache_config)
    ts_config.enable_spatial_policies = True
    ts_config.enable_selective_recompute = True
    ts_spatial = SpatialIndex(precision=cache_config.geohash_precision, region_decay_rate=cache_config.region_decay_rate)
    for poi in pois:
        ts_spatial.register_document(poi.doc_id, poi.latitude, poi.longitude)
    ts_cm = CacheManager(config=cache_config, spatial_index=ts_spatial, enable_spatial=True, enable_non_prefix_reuse=True)
    ts_ctrl = RAGController(config=ts_config, cache_manager=ts_cm, spatial_index=ts_spatial, simulation_mode=True)

    window = 100
    ts_hit_rates: list[dict] = []
    for q_idx, request in enumerate(trace):
        ts_ctrl.process_request(request)
        if (q_idx + 1) % window == 0:
            cs = ts_cm.stats
            lookups = cs.total_lookups if cs.total_lookups > 0 else 1
            ts_hit_rates.append({
                "query": q_idx + 1,
                "hit_rate": cs.hit_rate,
                "prefix_hits": cs.prefix_hits,
                "non_prefix_hits": cs.non_prefix_hits,
                "gpu_util": ts_cm.gpu_allocator.utilization,
                "host_util": ts_cm.host_allocator.utilization,
                "evictions_gpu": cs.evictions_gpu,
                "evictions_host": cs.evictions_host,
                "admissions": cs.admissions,
            })
    all_results["time_series"] = ts_hit_rates

    warmup_n = 200  # first 200 queries as warmup
    warmup_end_idx = warmup_n // window - 1
    if warmup_end_idx >= 0 and len(ts_hit_rates) > warmup_end_idx + 1:
        warmup_hr = ts_hit_rates[warmup_end_idx]["hit_rate"]
        steady_hr = ts_hit_rates[-1]["hit_rate"]
        print(f"  Warmup ({warmup_n} queries): hit rate = {warmup_hr:.4f}")
        print(f"  Steady state (all {len(trace)}): hit rate = {steady_hr:.4f}")
        print(f"  Time-series snapshots: {len(ts_hit_rates)} data points")

    # Save results
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {output_path}")

    return all_results


if __name__ == "__main__":
    run_full_benchmark()
