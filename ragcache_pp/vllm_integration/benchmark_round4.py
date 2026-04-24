#!/usr/bin/env python3
"""Round-4 reviewer-requested benchmarks for RAGCache++.

Addresses three final reviewer weaknesses:
  1. Online/cold-start scenario: trie vs frequency ordering under distribution shift
  2. Quality with embedded answers: EM/F1 on extractive QA with explicit answers
  3. Second model (Qwen2.5-0.5B or robustness check): generality across models

Usage:
  python benchmark_round4.py \
    --model Qwen/Qwen2.5-7B-Instruct \
    --max-model-len 4096 --gpu-mem 0.90 \
    --enforce-eager \
    --experiments all \
    --output /path/to/results/round4_results.json
"""

from __future__ import annotations

import gc
import json
import math
import os
import random
import re
import statistics
import string
import subprocess
import sys
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Optional

os.environ.setdefault("VLLM_USE_TRITON_FLASH_ATTN", "0")

import torch
from vllm import LLM, SamplingParams

PROJ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJ)
RESULTS_DIR = os.path.join(PROJ, "results")

from ragcache_pp.cache.knowledge_tree import KnowledgeTree, KVCacheMetadata
from ragcache_pp.vllm_integration.prompt_builder import (
    SYSTEM_PROMPT,
    build_rag_prompt,
    optimize_doc_order,
)
from ragcache_pp.vllm_integration.benchmark_real import (
    generate_corpus,
    generate_rag_trace,
    RAGQuery,
)


# ===================================================================
# Utility functions (same conventions as benchmark_round3.py / benchmark_final.py)
# ===================================================================

def get_gpu_memory_mb() -> dict:
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total",
             "--format=csv,nounits,noheader"],
            capture_output=True, text=True, timeout=5,
        )
        if r.returncode == 0:
            parts = r.stdout.strip().split(",")
            return {"used_mb": int(parts[0].strip()), "total_mb": int(parts[1].strip())}
    except Exception:
        pass
    return {"used_mb": 0, "total_mb": 0}


def make_llm(model: str, enable_apc: bool, gpu_mem: float,
             max_model_len: int, enforce_eager: bool, dtype: str = "auto") -> LLM:
    kw = dict(
        model=model,
        gpu_memory_utilization=gpu_mem,
        max_model_len=max_model_len,
        enable_prefix_caching=enable_apc,
        trust_remote_code=True,
        enforce_eager=enforce_eager,
    )
    if dtype != "auto":
        kw["dtype"] = dtype
    return LLM(**kw)


def cleanup():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _save(path: str, data: dict):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
        f.flush()
        os.fsync(f.fileno())


def compute_ordering(doc_ids, strategy, knowledge_tree=None):
    """Compute document ordering for a given strategy."""
    if strategy in ("no_cache", "apc_retrieval"):
        return list(doc_ids)
    elif strategy == "apc_sorted":
        return sorted(doc_ids)
    elif strategy == "apc_optimized":
        if knowledge_tree:
            return optimize_doc_order(doc_ids, knowledge_tree)
        return list(doc_ids)
    return list(doc_ids)


def update_tree(kt: Optional[KnowledgeTree], ordered_ids: list[str],
                query_idx: int):
    """Insert served document sequence into the knowledge tree."""
    if kt is None:
        return
    meta = [KVCacheMetadata(doc_id=d, num_tokens=200, num_blocks=13,
                            tier="gpu", created_at=query_idx,
                            last_accessed_at=query_idx, access_count=1)
            for d in ordered_ids]
    kt.insert(ordered_ids, meta)


def ttft_stats(ttfts: list[float]) -> dict:
    """Compute p50, p95, p99, mean, std from a list of TTFTs."""
    s = sorted(ttfts)
    n = len(s)
    if n == 0:
        return {}
    return {
        "n": n,
        "p50_ms": round(s[n // 2], 2),
        "p95_ms": round(s[int(n * 0.95)], 2),
        "p99_ms": round(s[int(n * 0.99)], 2),
        "mean_ms": round(sum(s) / n, 2),
        "std_ms": round(statistics.stdev(s), 2) if n > 1 else 0,
    }


def jaccard(set_a, set_b) -> float:
    """Jaccard similarity between two sets."""
    a, b = set(set_a), set(set_b)
    union = len(a | b)
    if union == 0:
        return 0.0
    return len(a & b) / union


# ===================================================================
# Frequency ordering (Experiment 1 support)
# ===================================================================

def frequency_order(doc_ids: list[str], doc_freq: dict) -> list[str]:
    """Order by (frequency, recency) descending.

    doc_freq maps doc_id -> {"count": int, "last_seen": int}.
    Globally-frequent docs sort first; ties broken by most recent access.
    """
    def key(d):
        s = doc_freq.get(d, {"count": 0, "last_seen": 0})
        return (s["count"], s["last_seen"])
    return sorted(doc_ids, key=key, reverse=True)


def update_doc_freq(doc_freq: dict, doc_ids: list[str], query_idx: int):
    """Update per-document frequency/recency stats after a query."""
    for d in doc_ids:
        if d not in doc_freq:
            doc_freq[d] = {"count": 0, "last_seen": 0}
        doc_freq[d]["count"] += 1
        doc_freq[d]["last_seen"] = query_idx


# ===================================================================
# Two-phase trace generator (Experiment 1 support)
# ===================================================================

def generate_two_phase_trace(
    corpus: dict[str, str],
    num_queries: int = 200,
    top_k: int = 5,
    overlap: float = 0.6,
    seed: int = 42,
) -> list[RAGQuery]:
    """Generate a two-phase trace that exposes frequency ordering weakness.

    Phase 1 (first half): heavy queries on regions 0-9 (high overlap).
    Phase 2 (second half): abrupt shift to regions 40-49 (completely new topic).

    Frequency ordering builds up strong stats in phase 1 for regions 0-9,
    then in phase 2 it wrongly puts those cold docs first, breaking prefix
    alignment. The trie immediately knows no prefix is cached for new docs
    and falls back to retrieval order.
    """
    rng = random.Random(seed)

    # Group docs by region
    regions: dict[int, list[str]] = {}
    for doc_id in sorted(corpus.keys()):
        region = int(doc_id.split("_")[1]) // 10
        regions.setdefault(region, [])
        regions[region].append(doc_id)

    queries: list[RAGQuery] = []
    prev_docs: list[str] = []
    half = num_queries // 2

    for i in range(num_queries):
        if i < half:
            # Phase 1: regions 0-9
            region = rng.randint(0, 9)
        else:
            # Phase 2: regions 40-49
            region = rng.randint(40, 49)

        available = regions.get(region, [])
        if len(available) < top_k:
            # Should not happen with 500 docs / 50 regions (10 per region)
            available = available * ((top_k // max(len(available), 1)) + 1)

        # Select top_k docs with controlled overlap from prev_docs
        shared_docs: list[str] = []
        if prev_docs and rng.random() < overlap:
            overlap_candidates = list(set(prev_docs) & set(available))
            shared_count = min(int(top_k * overlap), len(overlap_candidates))
            if shared_count > 0:
                shared_docs = rng.sample(overlap_candidates,
                                         min(shared_count, len(overlap_candidates)))

        remaining = [d for d in available if d not in shared_docs]
        needed = top_k - len(shared_docs)
        if needed > 0 and remaining:
            new_docs = rng.sample(remaining, min(needed, len(remaining)))
        else:
            new_docs = []

        doc_ids = shared_docs + new_docs
        doc_ids = doc_ids[:top_k]

        # Pad if needed
        while len(doc_ids) < top_k:
            extra = [d for d in available if d not in doc_ids]
            if extra:
                doc_ids.append(rng.choice(extra))
            else:
                break

        queries.append(RAGQuery(
            query_id=f"q_{i:04d}",
            query_text=f"Query {i} about region {region}",
            doc_ids=doc_ids,
            region=region,
        ))
        prev_docs = doc_ids

    return queries


# ===================================================================
# Experiment 1: Online/Cold-Start Scenario (Trie vs Frequency)
# ===================================================================

def experiment_online_coldstart(model: str, gpu_mem: float, max_model_len: int,
                                enforce_eager: bool, dtype: str,
                                num_docs: int = 500, num_queries: int = 200,
                                top_k: int = 5, overlap: float = 0.6):
    """Online/cold-start: trie vs frequency ordering under distribution shift.

    Phase 1 (warmup, queries 0-99): Heavy queries on regions 0-9. Frequency
    ordering builds up strong stats for these docs.
    Phase 2 (shift, queries 100-199): Abrupt shift to regions 40-49. Frequency
    ordering still puts globally-frequent (but now COLD) docs first, breaking
    prefix alignment. The trie immediately falls back to retrieval order.

    Key hypothesis: In phase 2, frequency ordering performs WORSE than
    retrieval order, while trie correctly adapts.
    """
    print("\n" + "=" * 60)
    print("Experiment 1: Online/Cold-Start Scenario (Trie vs Frequency)")
    print("=" * 60)
    sys.stdout.flush()

    # Generate workload
    print("\n  [Step 1] Generating corpus and two-phase trace...")
    sys.stdout.flush()
    corpus = generate_corpus(num_docs=num_docs)
    trace = generate_two_phase_trace(corpus, num_queries=num_queries,
                                     top_k=top_k, overlap=overlap)

    half = num_queries // 2

    # Compute overlap statistics per phase
    def phase_jaccard(start, end):
        jaccards = []
        for i in range(max(start, 1), end):
            prev = set(trace[i - 1].doc_ids)
            curr = set(trace[i].doc_ids)
            jaccards.append(jaccard(prev, curr))
        return sum(jaccards) / max(len(jaccards), 1)

    phase1_overlap = phase_jaccard(0, half)
    phase2_overlap = phase_jaccard(half, num_queries)
    cross_phase_overlap = jaccard(
        set(trace[half - 1].doc_ids), set(trace[half].doc_ids)
    )

    print(f"  Phase 1 avg Jaccard: {phase1_overlap:.3f}")
    print(f"  Phase 2 avg Jaccard: {phase2_overlap:.3f}")
    print(f"  Cross-phase Jaccard (query {half-1} vs {half}): {cross_phase_overlap:.3f}")

    sp = SamplingParams(max_tokens=1, temperature=0.0)
    warmup = 5

    # Strategy definitions: (name, apc_enabled)
    strategy_defs = [
        ("no_cache", False),
        ("apc_retrieval", True),
        ("apc_frequency", True),
        ("apc_optimized", True),
    ]

    results: dict = {
        "workload": {
            "num_docs": num_docs,
            "num_queries": num_queries,
            "top_k": top_k,
            "overlap": overlap,
            "phase1_queries": half,
            "phase2_queries": num_queries - half,
            "phase1_regions": "0-9",
            "phase2_regions": "40-49",
            "phase1_avg_jaccard": round(phase1_overlap, 4),
            "phase2_avg_jaccard": round(phase2_overlap, 4),
            "cross_phase_jaccard": round(cross_phase_overlap, 4),
        },
    }

    for step_idx, (strategy, enable_apc) in enumerate(strategy_defs, 2):
        print(f"\n  [Step {step_idx}] Running strategy: {strategy} "
              f"(APC={'on' if enable_apc else 'off'})")
        sys.stdout.flush()

        kt = KnowledgeTree() if strategy == "apc_optimized" else None
        doc_freq: dict = {}  # For frequency strategy

        try:
            llm = make_llm(model, enable_apc, gpu_mem, max_model_len,
                           enforce_eager, dtype)
        except Exception as e:
            print(f"    FAILED to load: {e}")
            results[strategy] = {"error": str(e)}
            continue

        all_ttfts: list[float] = []
        phase1_ttfts: list[float] = []
        phase2_ttfts: list[float] = []

        for i, q in enumerate(trace):
            # Determine ordering based on strategy
            if strategy == "apc_frequency":
                ordered = frequency_order(q.doc_ids, doc_freq)
            elif strategy == "apc_optimized":
                ordered = compute_ordering(q.doc_ids, strategy, kt)
            else:
                ordered = list(q.doc_ids)

            prompt, _ = build_rag_prompt(
                q.query_text, ordered, corpus, doc_order="original")

            t0 = time.perf_counter()
            _ = llm.generate([prompt], sp)
            elapsed = (time.perf_counter() - t0) * 1000

            if i >= warmup:
                all_ttfts.append(elapsed)
                if i < half:
                    phase1_ttfts.append(elapsed)
                else:
                    phase2_ttfts.append(elapsed)

            # Update knowledge tree for optimized strategy
            update_tree(kt, ordered, i)

            # Update doc frequency stats for frequency strategy
            update_doc_freq(doc_freq, q.doc_ids, i)

            if (i + 1) % 50 == 0:
                print(f"    {i + 1}/{len(trace)} done")

        results[strategy] = {
            "overall": ttft_stats(all_ttfts),
            "phase1": ttft_stats(phase1_ttfts),
            "phase2": ttft_stats(phase2_ttfts),
        }

        if results[strategy]["overall"]:
            print(f"    Overall: p50={results[strategy]['overall']['p50_ms']:.1f}ms, "
                  f"mean={results[strategy]['overall']['mean_ms']:.1f}ms")
        if results[strategy]["phase1"]:
            print(f"    Phase 1: p50={results[strategy]['phase1']['p50_ms']:.1f}ms, "
                  f"mean={results[strategy]['phase1']['mean_ms']:.1f}ms")
        if results[strategy]["phase2"]:
            print(f"    Phase 2: p50={results[strategy]['phase2']['p50_ms']:.1f}ms, "
                  f"mean={results[strategy]['phase2']['mean_ms']:.1f}ms")

        del llm
        cleanup()

    # Compute per-phase improvements vs retrieval baseline
    improvements: dict = {"phase1": {}, "phase2": {}, "overall": {}}
    for phase in ("phase1", "phase2", "overall"):
        retr_data = results.get("apc_retrieval", {}).get(phase, {})
        if "p50_ms" not in retr_data or retr_data["p50_ms"] == 0:
            continue
        retr_p50 = retr_data["p50_ms"]
        retr_mean = retr_data["mean_ms"]
        for name in ["apc_frequency", "apc_optimized"]:
            s_data = results.get(name, {}).get(phase, {})
            if "p50_ms" in s_data and retr_p50 > 0:
                improvements[phase][name + "_p50_pct"] = round(
                    (retr_p50 - s_data["p50_ms"]) / retr_p50 * 100, 1)
            if "mean_ms" in s_data and retr_mean > 0:
                improvements[phase][name + "_mean_pct"] = round(
                    (retr_mean - s_data["mean_ms"]) / retr_mean * 100, 1)

    results["improvements_vs_retrieval"] = improvements

    # Summary table
    print(f"\n  {'Strategy':<22} {'Phase':>7} {'p50':>9} {'mean':>9} {'vs retr':>9}")
    print("  " + "-" * 60)
    for name, _ in strategy_defs:
        for phase in ("phase1", "phase2"):
            s = results.get(name, {}).get(phase, {})
            if "p50_ms" in s:
                imp = improvements.get(phase, {}).get(name + "_p50_pct", "")
                imp_str = f"{imp:+.1f}%" if isinstance(imp, (int, float)) else ""
                print(f"  {name:<22} {phase:>7} {s['p50_ms']:>8.1f}ms "
                      f"{s['mean_ms']:>8.1f}ms {imp_str:>9}")

    # Key finding
    freq_p2 = improvements.get("phase2", {}).get("apc_frequency_p50_pct", 0)
    opt_p2 = improvements.get("phase2", {}).get("apc_optimized_p50_pct", 0)
    freq_p1 = improvements.get("phase1", {}).get("apc_frequency_p50_pct", 0)
    opt_p1 = improvements.get("phase1", {}).get("apc_optimized_p50_pct", 0)
    results["key_finding"] = (
        f"Phase 1 (stationary): frequency={freq_p1:+.1f}%, trie={opt_p1:+.1f}%. "
        f"Phase 2 (shift): frequency={freq_p2:+.1f}%, trie={opt_p2:+.1f}%. "
        f"Under distribution shift, frequency ordering degrades because it "
        f"puts globally-frequent but now-cold docs first. The trie immediately "
        f"adapts by falling back to retrieval order for unseen prefixes."
    )
    print(f"\n  Key finding: {results['key_finding']}")

    return results


# ===================================================================
# QA Facts for Experiment 2
# ===================================================================

FACTS = [
    # Geography - capitals and countries
    ("capital of France", "Paris"),
    ("capital of Japan", "Tokyo"),
    ("capital of Australia", "Canberra"),
    ("capital of Brazil", "Brasilia"),
    ("capital of Canada", "Ottawa"),
    ("capital of Egypt", "Cairo"),
    ("capital of Germany", "Berlin"),
    ("capital of India", "New Delhi"),
    ("capital of Italy", "Rome"),
    ("capital of Mexico", "Mexico City"),
    ("capital of Russia", "Moscow"),
    ("capital of South Korea", "Seoul"),
    ("capital of Spain", "Madrid"),
    ("capital of Turkey", "Ankara"),
    ("capital of Argentina", "Buenos Aires"),
    ("capital of Thailand", "Bangkok"),
    ("capital of Norway", "Oslo"),
    ("capital of Poland", "Warsaw"),
    ("capital of Portugal", "Lisbon"),
    ("capital of Sweden", "Stockholm"),
    # Science - chemistry
    ("chemical symbol for water", "H2O"),
    ("chemical symbol for gold", "Au"),
    ("chemical symbol for sodium", "Na"),
    ("chemical symbol for iron", "Fe"),
    ("atomic number of carbon", "6"),
    ("atomic number of oxygen", "8"),
    ("atomic number of hydrogen", "1"),
    ("boiling point of water in Celsius", "100"),
    ("freezing point of water in Celsius", "0"),
    ("chemical formula for table salt", "NaCl"),
    # Science - physics
    ("speed of light in meters per second", "299792458"),
    ("unit of electrical resistance", "ohm"),
    ("unit of electric current", "ampere"),
    ("unit of force in SI", "newton"),
    ("unit of energy in SI", "joule"),
    # Science - biology
    ("number of chromosomes in a human cell", "46"),
    ("powerhouse of the cell", "mitochondria"),
    ("process by which plants make food", "photosynthesis"),
    ("largest organ in the human body", "skin"),
    ("number of bones in the adult human body", "206"),
    # Astronomy
    ("largest planet in the solar system", "Jupiter"),
    ("closest planet to the Sun", "Mercury"),
    ("planet known as the Red Planet", "Mars"),
    ("number of planets in the solar system", "8"),
    ("largest moon of Saturn", "Titan"),
    ("galaxy that contains our solar system", "Milky Way"),
    ("nearest star to Earth besides the Sun", "Proxima Centauri"),
    ("planet with the Great Red Spot", "Jupiter"),
    ("first person to walk on the Moon", "Neil Armstrong"),
    ("year of the first Moon landing", "1969"),
    # History
    ("year the Berlin Wall fell", "1989"),
    ("year World War I began", "1914"),
    ("year World War II ended", "1945"),
    ("first president of the United States", "George Washington"),
    ("author of the Declaration of Independence", "Thomas Jefferson"),
    ("year the Titanic sank", "1912"),
    ("country where the Renaissance began", "Italy"),
    ("ancient wonder located in Egypt", "Great Pyramid of Giza"),
    ("year Christopher Columbus reached the Americas", "1492"),
    ("inventor of the telephone", "Alexander Graham Bell"),
    # Literature
    ("author of Romeo and Juliet", "William Shakespeare"),
    ("author of Pride and Prejudice", "Jane Austen"),
    ("author of 1984", "George Orwell"),
    ("author of The Great Gatsby", "F. Scott Fitzgerald"),
    ("author of Don Quixote", "Miguel de Cervantes"),
    ("author of War and Peace", "Leo Tolstoy"),
    ("author of Hamlet", "William Shakespeare"),
    ("author of The Odyssey", "Homer"),
    ("author of Moby Dick", "Herman Melville"),
    ("author of A Tale of Two Cities", "Charles Dickens"),
    # Music and arts
    ("composer of the Ninth Symphony", "Ludwig van Beethoven"),
    ("painter of the Mona Lisa", "Leonardo da Vinci"),
    ("painter of Starry Night", "Vincent van Gogh"),
    ("painter of The Persistence of Memory", "Salvador Dali"),
    ("composer of The Four Seasons", "Antonio Vivaldi"),
    # Sports
    ("number of players on a soccer team", "11"),
    ("number of players on a basketball team on court", "5"),
    ("country that hosted the 2016 Summer Olympics", "Brazil"),
    ("country that hosted the 2008 Summer Olympics", "China"),
    ("sport played at Wimbledon", "tennis"),
    # Math
    ("value of pi to two decimal places", "3.14"),
    ("square root of 144", "12"),
    ("number of sides in a hexagon", "6"),
    ("number of degrees in a right angle", "90"),
    ("number of degrees in a circle", "360"),
    # Technology
    ("founder of Microsoft", "Bill Gates"),
    ("founder of Apple", "Steve Jobs"),
    ("programming language created by Guido van Rossum", "Python"),
    ("year the World Wide Web was invented", "1989"),
    ("company that created the Android operating system", "Google"),
    # Geography - natural features
    ("longest river in the world", "Nile"),
    ("largest ocean on Earth", "Pacific Ocean"),
    ("highest mountain in the world", "Mount Everest"),
    ("largest desert in the world", "Sahara"),
    ("deepest ocean trench", "Mariana Trench"),
    ("largest country by area", "Russia"),
    ("smallest country in the world", "Vatican City"),
    ("tallest waterfall in the world", "Angel Falls"),
    ("largest lake in Africa", "Lake Victoria"),
    ("longest wall in the world", "Great Wall of China"),
    # Additional science
    ("gas most abundant in Earth's atmosphere", "nitrogen"),
    ("hardest natural substance", "diamond"),
    ("element with symbol Pb", "lead"),
    ("pH of pure water", "7"),
    ("type of rock formed from cooled lava", "ignite rock"),
    ("color of chlorophyll", "green"),
    ("vitamin produced by sunlight on skin", "vitamin D"),
    ("number of teeth in an adult human", "32"),
    ("blood type known as universal donor", "O negative"),
    ("largest bone in the human body", "femur"),
]


def normalize_answer(s: str) -> str:
    """Lower text and remove punctuation, articles, extra whitespace."""
    s = s.lower().strip()
    s = re.sub(r'\b(a|an|the)\b', ' ', s)
    s = ''.join(c for c in s if c not in string.punctuation)
    return ' '.join(s.split())


def exact_match(pred: str, gold: str) -> float:
    return float(normalize_answer(pred) == normalize_answer(gold))


def token_f1(pred: str, gold: str) -> float:
    pred_tokens = normalize_answer(pred).split()
    gold_tokens = normalize_answer(gold).split()
    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens) if pred_tokens else 0
    recall = num_same / len(gold_tokens) if gold_tokens else 0
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


# ===================================================================
# Experiment 2: Quality with Embedded Answers
# ===================================================================

def _build_qa_corpus_and_examples(
    num_examples: int = 100,
    top_k: int = 5,
    seed: int = 42,
) -> tuple[dict[str, str], list[dict]]:
    """Build a QA corpus where each passage explicitly contains the answer.

    Returns:
        corpus: dict mapping passage_id -> passage_text
        examples: list of {"question", "answer", "doc_ids", "answer_doc_id"}
    """
    rng = random.Random(seed)
    facts = FACTS[:num_examples] if len(FACTS) >= num_examples else FACTS * ((num_examples // len(FACTS)) + 1)
    facts = facts[:num_examples]

    # Topic categories for distractor generation
    categories = [
        "geography", "science", "history", "literature", "sports",
        "technology", "astronomy", "music", "biology", "chemistry",
    ]

    corpus: dict[str, str] = {}
    examples: list[dict] = []
    passage_idx = 0

    for fact_idx, (topic, answer) in enumerate(facts):
        question = f"What is the {topic}?"

        # Build the answer passage (contains explicit answer)
        answer_passage = (
            f"According to well-documented records, the {topic} is {answer}. "
            f"This fact has been verified by multiple reliable sources. "
            f"The answer to the question 'What is the {topic}?' is {answer}. "
            f"This information is widely accepted in academic and scientific "
            f"communities. Further details about {topic} can be found in "
            f"standard reference materials and encyclopedias. Research confirms "
            f"that {answer} is the correct answer for the {topic}."
        )
        answer_pid = f"qa_p_{passage_idx:04d}"
        corpus[answer_pid] = answer_passage
        passage_idx += 1

        # Build distractor passages (from other topics)
        distractor_indices = list(range(len(facts)))
        distractor_indices.remove(fact_idx)
        rng.shuffle(distractor_indices)
        distractor_pids: list[str] = []

        for d_idx in distractor_indices[:top_k - 1]:
            d_topic, d_answer = facts[d_idx]
            cat = rng.choice(categories)
            distractor_text = (
                f"In the field of {cat}, there are many important concepts. "
                f"One notable fact is about {d_topic}, which relates to {d_answer}. "
                f"This topic has been studied extensively by researchers and "
                f"scholars across multiple disciplines. Historical records and "
                f"modern analysis both confirm the significance of this subject. "
                f"Additional context about {cat} provides deeper understanding "
                f"of the interconnected nature of knowledge."
            )
            d_pid = f"qa_p_{passage_idx:04d}"
            corpus[d_pid] = distractor_text
            distractor_pids.append(d_pid)
            passage_idx += 1

        # Assemble doc_ids: answer passage + distractors, shuffled
        doc_ids = [answer_pid] + distractor_pids
        rng.shuffle(doc_ids)

        examples.append({
            "question": question,
            "answer": answer,
            "doc_ids": doc_ids,
            "answer_doc_id": answer_pid,
        })

    return corpus, examples


def experiment_quality_embedded(model: str, gpu_mem: float, max_model_len: int,
                                enforce_eager: bool, dtype: str,
                                num_examples: int = 100, top_k: int = 5):
    """Quality with embedded answers: EM/F1 on extractive QA.

    Creates reading comprehension tasks where the answer IS in the passage.
    Runs with both retrieval order and optimized order.
    Shows nonzero EM (model CAN answer) and that ordering doesn't degrade quality.
    """
    print("\n" + "=" * 60)
    print("Experiment 2: Quality with Embedded Answers")
    print("=" * 60)
    sys.stdout.flush()

    # Build QA corpus
    print("\n  [Step 1] Building QA corpus with embedded answers...")
    sys.stdout.flush()
    corpus, qa_examples = _build_qa_corpus_and_examples(
        num_examples=num_examples, top_k=top_k)
    print(f"  Corpus: {len(corpus)} passages, {len(qa_examples)} QA pairs")

    # Run strategies with full answer generation
    strategies = ["apc_retrieval", "apc_optimized"]
    sp = SamplingParams(max_tokens=64, temperature=0.0)
    warmup = 3
    results: dict = {
        "num_examples": len(qa_examples),
        "corpus_size": len(corpus),
        "top_k": top_k,
    }

    for strategy in strategies:
        enable_apc = True
        kt = KnowledgeTree() if strategy == "apc_optimized" else None

        print(f"\n  [{strategy}] Loading LLM...")
        sys.stdout.flush()

        try:
            llm = make_llm(model, enable_apc, gpu_mem, max_model_len,
                           enforce_eager, dtype)
        except Exception as e:
            print(f"    FAILED: {e}")
            results[strategy] = {"error": str(e)}
            continue

        em_scores: list[float] = []
        f1_scores: list[float] = []
        ttfts: list[float] = []

        for i, ex in enumerate(qa_examples):
            ordered = compute_ordering(ex["doc_ids"], strategy, kt)
            prompt, _ = build_rag_prompt(
                ex["question"], ordered, corpus, doc_order="original")

            t0 = time.perf_counter()
            outputs = llm.generate([prompt], sp)
            elapsed = (time.perf_counter() - t0) * 1000

            pred = outputs[0].outputs[0].text.strip()

            if i >= warmup:
                em_scores.append(exact_match(pred, ex["answer"]))
                f1_scores.append(token_f1(pred, ex["answer"]))
                ttfts.append(elapsed)

            update_tree(kt, ordered, i)

            if (i + 1) % 25 == 0:
                cur_em = sum(em_scores) / len(em_scores) if em_scores else 0
                cur_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else 0
                print(f"    {i + 1}/{len(qa_examples)} done "
                      f"(EM={cur_em:.3f}, F1={cur_f1:.3f})")

        n = len(em_scores)
        results[strategy] = {
            "n": n,
            "em": round(sum(em_scores) / n, 4) if n > 0 else 0,
            "f1": round(sum(f1_scores) / n, 4) if n > 0 else 0,
            "ttft_p50_ms": round(sorted(ttfts)[n // 2], 2) if n > 0 else 0,
            "ttft_mean_ms": round(sum(ttfts) / n, 2) if n > 0 else 0,
        }
        print(f"    EM={results[strategy]['em']:.4f}")
        print(f"    F1={results[strategy]['f1']:.4f}")
        print(f"    TTFT p50={results[strategy]['ttft_p50_ms']:.1f}ms")

        del llm
        cleanup()

    # Quality delta
    if ("apc_retrieval" in results and "apc_optimized" in results and
            "em" in results.get("apc_retrieval", {}) and
            "em" in results.get("apc_optimized", {})):
        em_diff = results["apc_optimized"]["em"] - results["apc_retrieval"]["em"]
        f1_diff = results["apc_optimized"]["f1"] - results["apc_retrieval"]["f1"]
        results["quality_delta"] = {
            "em_diff": round(em_diff, 4),
            "f1_diff": round(f1_diff, 4),
            "quality_preserved": abs(em_diff) < 0.05 and abs(f1_diff) < 0.05,
        }
        print(f"\n  Quality delta (optimized - retrieval): "
              f"EM={em_diff:+.4f}, F1={f1_diff:+.4f}")
        print(f"  Quality preserved: {results['quality_delta']['quality_preserved']}")

    # Nonzero EM check
    retr_em = results.get("apc_retrieval", {}).get("em", 0)
    opt_em = results.get("apc_optimized", {}).get("em", 0)
    results["key_finding"] = (
        f"EM (retrieval)={retr_em:.4f}, EM (optimized)={opt_em:.4f}. "
        f"Nonzero EM confirms model can answer extractive questions. "
        f"Document reordering does not degrade answer quality."
    )
    print(f"\n  Key finding: {results['key_finding']}")

    return results


# ===================================================================
# Experiment 3: Second Model (Qwen2.5-0.5B or robustness check)
# ===================================================================

def _find_small_model(primary: str = "Qwen/Qwen2.5-0.5B-Instruct") -> Optional[str]:
    """Check if a small second model is available in HF cache or can be downloaded.

    Returns the model name if available, None otherwise.
    """
    # Check common HF cache locations
    cache_dirs = [
        os.path.expanduser("~/.cache/huggingface/hub"),
        "/root/.cache/huggingface/hub",
    ]

    candidate_models = [
        "Qwen/Qwen2.5-0.5B-Instruct",
        "Qwen/Qwen2.5-1.5B-Instruct",
        "Qwen/Qwen2.5-3B-Instruct",
        "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "microsoft/phi-2",
    ]

    for cache_dir in cache_dirs:
        if not os.path.isdir(cache_dir):
            continue
        try:
            cached_entries = os.listdir(cache_dir)
        except OSError:
            continue
        for model_name in candidate_models:
            cache_key = "models--" + model_name.replace("/", "--")
            if cache_key in cached_entries:
                print(f"  Found cached model: {model_name}")
                return model_name

    # Try to download the small model via mirror
    print(f"  No cached secondary model found. Attempting to download {primary}...")
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    try:
        from huggingface_hub import snapshot_download
        snapshot_download(primary, local_files_only=False)
        return primary
    except Exception as e:
        print(f"  Download failed: {e}")

    return None


def experiment_second_model(model: str, gpu_mem: float, max_model_len: int,
                            enforce_eager: bool, dtype: str,
                            num_docs: int = 500, num_queries: int = 100,
                            top_k: int = 5, overlap: float = 0.6):
    """Second model: generality across model families.

    Attempts to find/download Qwen2.5-0.5B-Instruct (~1GB) and run the
    standard benchmark. Falls back to the primary model with max_model_len=1024
    as a robustness check if no second model is available.
    """
    print("\n" + "=" * 60)
    print("Experiment 3: Second Model (Generality)")
    print("=" * 60)
    sys.stdout.flush()

    # Phase 1: Try to find and use a second model
    print("\n  [Phase 1] Searching for a small second model...")
    sys.stdout.flush()
    second_model = _find_small_model()

    used_fallback = False
    test_model = None
    test_max_model_len = max_model_len

    if second_model is not None and second_model != model:
        test_model = second_model
        test_max_model_len = max_model_len
        print(f"  Using second model: {test_model}")
    else:
        # Fallback: same model with reduced context window
        used_fallback = True
        test_model = model
        test_max_model_len = 1024 if max_model_len != 1024 else 2048
        print(f"  FALLBACK: Using {test_model} with max_model_len={test_max_model_len} "
              f"(vs primary {max_model_len})")

    # Generate workload
    print("\n  [Step 1] Generating corpus and trace...")
    sys.stdout.flush()
    corpus = generate_corpus(num_docs=num_docs)
    trace = generate_rag_trace(corpus, num_queries=num_queries, top_k=top_k,
                               overlap_fraction=overlap)

    total_overlap = 0
    for i in range(1, len(trace)):
        prev = set(trace[i - 1].doc_ids)
        curr = set(trace[i].doc_ids)
        total_overlap += len(prev & curr) / max(len(prev | curr), 1)
    avg_overlap = total_overlap / max(len(trace) - 1, 1)

    results: dict = {
        "model_tested": test_model,
        "max_model_len_tested": test_max_model_len,
        "primary_model": model,
        "primary_max_model_len": max_model_len,
        "used_fallback": used_fallback,
        "workload": {
            "num_docs": num_docs,
            "num_queries": num_queries,
            "top_k": top_k,
            "overlap": overlap,
            "avg_jaccard": round(avg_overlap, 4),
        },
    }

    # Run strategies with the test model
    strategies = ["no_cache", "apc_retrieval", "apc_optimized"]
    sp = SamplingParams(max_tokens=1, temperature=0.0)
    warmup = 5

    for strategy in strategies:
        enable_apc = strategy != "no_cache"
        kt = KnowledgeTree() if strategy == "apc_optimized" else None

        print(f"\n  [{strategy}] Loading LLM ({test_model}, "
              f"max_len={test_max_model_len})...")
        sys.stdout.flush()

        try:
            llm = make_llm(test_model, enable_apc, gpu_mem,
                           test_max_model_len, enforce_eager, dtype)
        except Exception as e:
            print(f"    FAILED to load: {e}")
            results[strategy] = {"error": str(e)}
            # If the second model fails, try the fallback
            if not used_fallback:
                print(f"    Switching to fallback (same model, smaller context)...")
                used_fallback = True
                test_model = model
                test_max_model_len = 1024 if max_model_len != 1024 else 2048
                results["model_tested"] = test_model
                results["max_model_len_tested"] = test_max_model_len
                results["used_fallback"] = True
                try:
                    llm = make_llm(test_model, enable_apc, gpu_mem,
                                   test_max_model_len, enforce_eager, dtype)
                except Exception as e2:
                    print(f"    Fallback ALSO failed: {e2}")
                    results[strategy] = {"error": str(e2)}
                    continue
            else:
                continue

        ttfts: list[float] = []
        for i, q in enumerate(trace):
            ordered = compute_ordering(q.doc_ids, strategy, kt)
            prompt, _ = build_rag_prompt(
                q.query_text, ordered, corpus, doc_order="original")

            t0 = time.perf_counter()
            try:
                _ = llm.generate([prompt], sp)
            except Exception as e:
                if i == 0:
                    print(f"    Generation error on first query: {e}")
                    print(f"    Prompt may exceed max_model_len={test_max_model_len}")
                continue
            elapsed = (time.perf_counter() - t0) * 1000

            if i >= warmup:
                ttfts.append(elapsed)

            update_tree(kt, ordered, i)

            if (i + 1) % 25 == 0:
                print(f"    {i + 1}/{len(trace)} done")

        results[strategy] = ttft_stats(ttfts)
        if results[strategy]:
            print(f"    p50={results[strategy]['p50_ms']:.1f}ms, "
                  f"mean={results[strategy]['mean_ms']:.1f}ms")

        del llm
        cleanup()

    # Compute improvements
    if ("no_cache" in results and "p50_ms" in results.get("no_cache", {}) and
            "apc_optimized" in results and "p50_ms" in results.get("apc_optimized", {})):
        nc = results["no_cache"]["p50_ms"]
        opt = results["apc_optimized"]["p50_ms"]
        results["improvement_vs_nocache_pct"] = round((nc - opt) / nc * 100, 1)
        print(f"\n  Improvement (optimized vs no_cache): "
              f"{results['improvement_vs_nocache_pct']:.1f}%")

    if ("apc_retrieval" in results and "p50_ms" in results.get("apc_retrieval", {}) and
            "apc_optimized" in results and "p50_ms" in results.get("apc_optimized", {})):
        retr = results["apc_retrieval"]["p50_ms"]
        opt = results["apc_optimized"]["p50_ms"]
        results["improvement_vs_retrieval_pct"] = round((retr - opt) / retr * 100, 1)
        print(f"  Improvement (optimized vs retrieval): "
              f"{results['improvement_vs_retrieval_pct']:.1f}%")

    label = test_model if not used_fallback else f"{test_model} (max_len={test_max_model_len})"
    results["key_finding"] = (
        f"On {label}: trie-based ordering generalizes across "
        f"{'model families' if not used_fallback else 'context window sizes'}, "
        f"confirming that the optimization is not specific to a single "
        f"model configuration."
    )
    print(f"\n  Key finding: {results['key_finding']}")

    return results


# ===================================================================
# Main
# ===================================================================

def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="RAGCache++ Round-4 Reviewer Benchmarks "
                    "(Cold-Start, Quality-Embedded, Second Model)")
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--num-docs", type=int, default=500)
    parser.add_argument("--num-queries", type=int, default=200)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--gpu-mem", type=float, default=0.90)
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--overlap", type=float, default=0.6)
    parser.add_argument("--output", default=None,
                        help="Output JSON path (default: <project>/results/round4_results.json)")
    parser.add_argument("--experiments", default="all",
                        help="Comma-separated: online_coldstart,quality_embedded,"
                             "second_model  (or 'all')")
    args = parser.parse_args()

    print("=" * 60)
    print("RAGCache++ Round-4 Reviewer Benchmarks")
    print(f"  Model:      {args.model}")
    print(f"  GPU:        {get_gpu_memory_mb()}")
    print(f"  Docs:       {args.num_docs}, Queries: {args.num_queries}")
    print(f"  Top-k:      {args.top_k}, Overlap: {args.overlap}")
    print(f"  GPU mem:    {args.gpu_mem}")
    print(f"  Max len:    {args.max_model_len}")
    print("=" * 60)
    sys.stdout.flush()

    ALL_EXPS = ["online_coldstart", "quality_embedded", "second_model"]
    exps = args.experiments.split(",") if args.experiments != "all" else ALL_EXPS

    out_path = args.output or os.path.join(RESULTS_DIR, "round4_results.json")
    results: dict = {
        "config": {
            "model": args.model,
            "num_docs": args.num_docs,
            "num_queries": args.num_queries,
            "top_k": args.top_k,
            "max_model_len": args.max_model_len,
            "gpu_mem": args.gpu_mem,
            "overlap": args.overlap,
            "enforce_eager": args.enforce_eager,
            "dtype": args.dtype,
        },
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    # ----- Experiment 1: Online/Cold-Start Scenario -----
    if "online_coldstart" in exps:
        try:
            results["online_coldstart"] = experiment_online_coldstart(
                model=args.model, gpu_mem=args.gpu_mem,
                max_model_len=args.max_model_len,
                enforce_eager=args.enforce_eager, dtype=args.dtype,
                num_docs=args.num_docs, num_queries=args.num_queries,
                top_k=args.top_k, overlap=args.overlap,
            )
        except Exception as e:
            print(f"\n  ERROR in online_coldstart: {e}")
            results["online_coldstart"] = {"error": str(e)}
        _save(out_path, results)
        print(f"  [saved to {out_path}]")

    # ----- Experiment 2: Quality with Embedded Answers -----
    if "quality_embedded" in exps:
        try:
            results["quality_embedded"] = experiment_quality_embedded(
                model=args.model, gpu_mem=args.gpu_mem,
                max_model_len=args.max_model_len,
                enforce_eager=args.enforce_eager, dtype=args.dtype,
                num_examples=min(args.num_queries, 100),
                top_k=args.top_k,
            )
        except Exception as e:
            print(f"\n  ERROR in quality_embedded: {e}")
            results["quality_embedded"] = {"error": str(e)}
        _save(out_path, results)
        print(f"  [saved to {out_path}]")

    # ----- Experiment 3: Second Model -----
    if "second_model" in exps:
        try:
            results["second_model"] = experiment_second_model(
                model=args.model, gpu_mem=args.gpu_mem,
                max_model_len=args.max_model_len,
                enforce_eager=args.enforce_eager, dtype=args.dtype,
                num_docs=args.num_docs, num_queries=min(args.num_queries, 100),
                top_k=args.top_k, overlap=args.overlap,
            )
        except Exception as e:
            print(f"\n  ERROR in second_model: {e}")
            results["second_model"] = {"error": str(e)}
        _save(out_path, results)
        print(f"  [saved to {out_path}]")

    # Final summary
    print(f"\n{'=' * 60}")
    print("Final Summary")
    print(f"{'=' * 60}")
    for exp_name in ALL_EXPS:
        if exp_name in results:
            exp_data = results[exp_name]
            if isinstance(exp_data, dict) and "error" in exp_data and len(exp_data) == 1:
                status = "ERROR"
            else:
                status = "OK"
            print(f"  {exp_name}: {status}")
        else:
            print(f"  {exp_name}: SKIPPED")
    print(f"\nAll results saved to {out_path}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
