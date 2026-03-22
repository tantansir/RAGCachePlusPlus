"""Real QA benchmark using HotpotQA to validate:
1. Document ordering doesn't hurt answer quality (EM/F1)
2. TTFT improvement holds on real retrieval workloads
"""
from __future__ import annotations

import gc
import json
import re
import string
import time
from collections import Counter
from dataclasses import dataclass, field

from datasets import load_dataset
from vllm import LLM, SamplingParams

from ragcache_pp.cache.knowledge_tree import KnowledgeTree
from ragcache_pp.vllm_integration.prompt_builder import build_rag_prompt


# ---------------------------------------------------------------------------
# QA Metrics (from SQuAD evaluation script)
# ---------------------------------------------------------------------------

def normalize_answer(s: str) -> str:
    """Lower text and remove punctuation, articles, extra whitespace."""
    s = s.lower()
    # Remove articles
    s = re.sub(r'\b(a|an|the)\b', ' ', s)
    # Remove punctuation
    s = s.translate(str.maketrans('', '', string.punctuation))
    # Remove extra whitespace
    s = ' '.join(s.split())
    return s


def exact_match(prediction: str, ground_truth: str) -> float:
    return float(normalize_answer(prediction) == normalize_answer(ground_truth))


def f1_score(prediction: str, ground_truth: str) -> float:
    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(ground_truth).split()
    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------

@dataclass
class HotpotQAExample:
    qid: str
    question: str
    answer: str
    doc_ids: list[str]
    doc_contents: dict[str, str]


def load_hotpotqa(num_examples: int = 100, max_docs: int = 5) -> list[HotpotQAExample]:
    """Load HotpotQA validation set and format as RAG queries."""
    ds = load_dataset("hotpot_qa", "fullwiki", split=f"validation[:{num_examples}]")
    examples = []
    for item in ds:
        ctx = item["context"]
        doc_ids = []
        doc_contents = {}
        for title, sentences in zip(ctx["title"][:max_docs], ctx["sentences"][:max_docs]):
            doc_id = title.replace(" ", "_")[:50]
            # Deduplicate doc IDs
            if doc_id in doc_contents:
                doc_id = f"{doc_id}_{len(doc_ids)}"
            doc_ids.append(doc_id)
            # Truncate to ~120 words to keep prompt within context window
            text = f"{title}\n" + " ".join(sentences)
            words = text.split()
            if len(words) > 120:
                text = " ".join(words[:120]) + "..."
            doc_contents[doc_id] = text

        examples.append(HotpotQAExample(
            qid=item["id"],
            question=item["question"],
            answer=item["answer"],
            doc_ids=doc_ids,
            doc_contents=doc_contents,
        ))
    return examples


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------

@dataclass
class QAResult:
    strategy: str
    em_scores: list[float] = field(default_factory=list)
    f1_scores: list[float] = field(default_factory=list)
    ttft_ms: list[float] = field(default_factory=list)

    def summary(self) -> dict:
        n = len(self.em_scores)
        if n == 0:
            return {"strategy": self.strategy, "n": 0}
        ttfts = sorted(self.ttft_ms)
        return {
            "strategy": self.strategy,
            "n": n,
            "em": round(sum(self.em_scores) / n, 4),
            "f1": round(sum(self.f1_scores) / n, 4),
            "ttft_p50_ms": round(ttfts[len(ttfts) // 2], 2),
            "ttft_mean_ms": round(sum(ttfts) / n, 2),
        }


def run_qa_strategy(
    llm: LLM,
    examples: list[HotpotQAExample],
    strategy: str,
    knowledge_tree: KnowledgeTree | None = None,
    warmup: int = 3,
) -> QAResult:
    """Run QA evaluation with a specific document ordering strategy."""
    sampling_params = SamplingParams(max_tokens=50, temperature=0.0)
    result = QAResult(strategy=strategy)

    doc_order_map = {
        "apc_retrieval": "original",
        "apc_random": "random",
        "apc_sorted": "sorted",
        "apc_optimized": "optimized",
        "no_cache": "original",
    }
    doc_order = doc_order_map.get(strategy, "original")

    for i, ex in enumerate(examples):
        prompt, ordered_ids = build_rag_prompt(
            ex.question, ex.doc_ids, ex.doc_contents,
            doc_order=doc_order,
            knowledge_tree=knowledge_tree,
        )

        t0 = time.perf_counter()
        outputs = llm.generate([prompt], sampling_params)
        t1 = time.perf_counter()

        prediction = outputs[0].outputs[0].text.strip()
        elapsed_ms = (t1 - t0) * 1000.0

        # Update knowledge tree
        if knowledge_tree is not None:
            from ragcache_pp.cache.knowledge_tree import KVCacheMetadata
            metadata_list = []
            for doc_id in ordered_ids:
                meta = KVCacheMetadata(
                    doc_id=doc_id, num_tokens=200, num_blocks=13,
                    tier="gpu", created_at=i, last_accessed_at=i, access_count=1,
                )
                metadata_list.append(meta)
            knowledge_tree.insert(ordered_ids, metadata_list)

        if i >= warmup:
            result.em_scores.append(exact_match(prediction, ex.answer))
            result.f1_scores.append(f1_score(prediction, ex.answer))
            result.ttft_ms.append(elapsed_ms)

        if (i + 1) % 25 == 0:
            print(f"  [{strategy}] {i + 1}/{len(examples)} done")

    return result


def run_hotpotqa_benchmark(
    model_name: str = "Qwen/Qwen2.5-1.5B-Instruct",
    num_examples: int = 100,
    output_path: str = "hotpotqa_results.json",
    gpu_mem: float = 0.85,
    max_model_len: int = 2048,
):
    """Run HotpotQA benchmark comparing document ordering strategies."""
    print("=" * 60)
    print("RAGCache++ HotpotQA Quality + Latency Benchmark")
    print("=" * 60)

    print(f"\n[Step 1] Loading HotpotQA ({num_examples} examples)...")
    examples = load_hotpotqa(num_examples)
    print(f"  Loaded {len(examples)} examples")

    # Compute document overlap between consecutive examples
    total_ov = 0
    for i in range(1, len(examples)):
        prev = set(examples[i - 1].doc_ids)
        curr = set(examples[i].doc_ids)
        union = len(prev | curr)
        if union > 0:
            total_ov += len(prev & curr) / union
    avg_ov = total_ov / max(len(examples) - 1, 1)
    print(f"  Avg consecutive Jaccard overlap: {avg_ov:.3f}")

    results = {}

    strategies = [
        ("no_cache", False, None),
        ("apc_retrieval", True, None),
        ("apc_random", True, None),
        ("apc_sorted", True, None),
        ("apc_optimized", True, KnowledgeTree()),
    ]

    import torch
    for step_idx, (strategy, enable_apc, kt) in enumerate(strategies, 2):
        print(f"\n[Step {step_idx}] Running: {strategy}")
        llm = LLM(
            model=model_name,
            gpu_memory_utilization=gpu_mem,
            max_model_len=max_model_len,
            enable_prefix_caching=enable_apc,
            trust_remote_code=True,
            enforce_eager=True,
        )

        r = run_qa_strategy(llm, examples, strategy, knowledge_tree=kt)
        results[strategy] = r.summary()
        s = r.summary()
        print(f"  EM={s['em']:.3f}, F1={s['f1']:.3f}, TTFT p50={s['ttft_p50_ms']:.1f}ms")

        del llm
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Summary
    print("\n" + "=" * 60)
    print("Results Summary")
    print("=" * 60)
    print(f"{'Strategy':<20} {'EM':>8} {'F1':>8} {'TTFT p50':>10} {'TTFT mean':>10}")
    print("-" * 60)
    for name, r in results.items():
        print(f"{name:<20} {r['em']:>7.3f} {r['f1']:>7.3f} "
              f"{r['ttft_p50_ms']:>9.1f}ms {r['ttft_mean_ms']:>9.1f}ms")

    # Quality comparison
    if "apc_retrieval" in results and "apc_optimized" in results:
        ret_em = results["apc_retrieval"]["em"]
        opt_em = results["apc_optimized"]["em"]
        ret_f1 = results["apc_retrieval"]["f1"]
        opt_f1 = results["apc_optimized"]["f1"]
        print(f"\nQuality delta (optimized - retrieval): EM={opt_em - ret_em:+.3f}, F1={opt_f1 - ret_f1:+.3f}")

    output = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "config": {
            "model_name": model_name,
            "num_examples": num_examples,
            "dataset": "hotpot_qa/fullwiki/validation",
        },
        "workload": {"avg_consecutive_overlap": round(avg_ov, 3)},
        "results": results,
    }

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {output_path}")
    return output


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--num-examples", type=int, default=100)
    parser.add_argument("--output", default="hotpotqa_results.json")
    parser.add_argument("--gpu-mem", type=float, default=0.85)
    parser.add_argument("--max-model-len", type=int, default=2048)
    args = parser.parse_args()

    run_hotpotqa_benchmark(
        model_name=args.model,
        num_examples=args.num_examples,
        output_path=args.output,
        gpu_mem=args.gpu_mem,
        max_model_len=args.max_model_len,
    )
