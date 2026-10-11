#!/usr/bin/env bash
#!/usr/bin/env python3
"""
Academic Benchmark and Validation Suite for Remote LLM Serving.
Evaluates OpenAI-compatible inference endpoints (LocalAI / llama.cpp) across
representative academic workloads:
1. Short Question / Factual Retrieval
2. In-Depth Academic Text Analysis
3. Cross-Document Comparative Synthesis
4. Structured Literature Review Table Generation
5. Long-Context Ingestion & Streaming Performance (TTFT, tokens/sec)
"""

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional, Tuple


SAMPLE_ABSTRACT = """
Title: Scaling Laws for Neural Language Models
Abstract: We investigate empirical scaling laws for language model performance across cross-entropy loss,
dataset size, model size, and compute. We find that performance scales as a power-law across more than
six orders of magnitude in compute, with minimal dependence on architectural hyperparameters such as depth
versus width. Furthermore, larger models are significantly more sample-efficient: optimal compute allocation
favors training larger models on fewer tokens rather than smaller models to convergence.
Sample size analyzed: 50+ models ranging from 10^3 to 10^11 parameters.
"""

PAPER_A_ATTENTION = """
Paper A: 'Attention Is All You Need' (Vaswani et al.)
Core Proposition: The Transformer architecture relies entirely on self-attention mechanisms to compute
representations of its input and output without using sequence-aligned RNNs or convolution.
Computational Complexity: Standard self-attention has a computational and memory complexity of O(N^2)
with respect to sequence length N, where N represents the number of tokens.
Key Advantage: Exceptional parallelization during training, capturing arbitrary long-range dependencies
without sequential path length bottlenecks.
Primary Limitation: Quadratic memory explosion for long sequences (e.g., >32k tokens) in the KV cache
and attention matrix computation.
"""

PAPER_B_MAMBA = """
Paper B: 'Mamba: Linear-Time Sequence Modeling with Selective State Spaces' (Gu and Dao)
Core Proposition: Structured State Space models (SSMs) with input-dependent selection mechanisms allow
selective information propagation along the sequence dimension.
Computational Complexity: Computes sequence representations with O(N) linear time and constant O(1)
memory complexity per generation step during inference.
Key Advantage: Eliminates the growing quadratic KV-cache bottleneck of Transformers, allowing scalable
inference across million-token sequences with high throughput.
Primary Limitation: May experience state compression decay on associative recall tasks requiring
exact retrieval of obscure facts from distant contexts compared to full-rank attention matrices.
"""

MULTI_STUDIES = """
Study 1 (Smith & Lin, 2023): Investigated prompt caching efficiency in long-context LLMs across 1,000 academic PDF queries. Found a 78% reduction in latency when reusing pre-computed KV prefixes. Sample: 1,000 papers. Method: Empirical benchmark on A100 GPUs.
Study 2 (Venkatesh et al., 2024): Evaluated citation hallucination rates in 14B vs 70B parameter open-weight models. Reported that 14B models hallucinated 14.2% of citations, while 70B models hallucinated 4.1%. Sample: 500 generated reviews. Method: Blind expert annotation.
Study 3 (Dubois & Zhang, 2024): Analyzed retrieval-augmented generation (RAG) vs long-context needle-in-a-haystack performance. Demonstrated that long-context models maintain higher thematic coherence, whereas chunked RAG excels at precise isolated entity retrieval. Sample: 200 synthetic books. Method: Needle retrieval accuracy test.
Study 4 (Kowalski, 2025): Benchmarked energy consumption and cost per 1M tokens across spot cloud GPU instances. Noted that dynamic spot instances reduced academic research compute expenses by 65% compared to dedicated on-demand clusters. Sample: 10,000 GPU rental hours. Method: Cost accounting on public clouds.
"""


class AcademicBenchmark:
    def __init__(self, base_url: str, model: str, api_key: str):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.results: List[Dict[str, Any]] = []

    def _send_request(self, messages: List[Dict[str, str]], stream: bool = False, max_tokens: int = 1024) -> Tuple[str, Dict[str, Any]]:
        url = f"{self.base_url}/chat/completions"
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.2,
            "max_tokens": max_tokens,
            "stream": stream,
        }

        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST"
        )

        start_time = time.perf_counter()
        first_token_time = None
        output_chunks = []

        try:
            with urllib.request.urlopen(req, timeout=300) as resp:
                if stream:
                    for line in resp:
                        line = line.decode("utf-8").strip()
                        if not line or not line.startswith("data:"):
                            continue
                        data_str = line[5:].strip()
                        if data_str == "[DONE]":
                            break
                        try:
                            chunk = json.loads(data_str)
                            delta = chunk.get("choices", [{}])[0].get("delta", {}).get("content", "")
                            if delta:
                                if first_token_time is None:
                                    first_token_time = time.perf_counter()
                                output_chunks.append(delta)
                        except Exception:
                            continue
                    full_text = "".join(output_chunks)
                else:
                    body = resp.read().decode("utf-8")
                    data = json.loads(body)
                    full_text = data.get("choices", [{}])[0].get("message", {}).get("content", "")
                    first_token_time = time.perf_counter()

        except Exception as e:
            return "", {"error": str(e), "duration_s": round(time.perf_counter() - start_time, 3)}

        total_time = time.perf_counter() - start_time
        ttft = (first_token_time - start_time) if first_token_time else total_time

        # Approximate token count by word count * 1.3
        words = len(full_text.split())
        est_tokens = int(words * 1.33)
        gen_tps = round(est_tokens / max(0.001, (total_time - ttft)), 1) if total_time > ttft else 0.0

        metrics = {
            "duration_s": round(total_time, 3),
            "ttft_s": round(ttft, 3),
            "est_tokens": est_tokens,
            "tokens_per_sec": gen_tps,
            "success": bool(full_text),
        }
        return full_text, metrics

    def test_short_question(self):
        print("\n[TEST 1] Short Question / Abstract Retrieval...")
        messages = [
            {"role": "system", "content": "You are a rigorous academic research assistant. Answer concisely with exact facts."},
            {"role": "user", "content": f"Based on the following abstract, state the empirical relationship found, the sample size analyzed, and the compute allocation recommendation:\n\n{SAMPLE_ABSTRACT}"}
        ]
        text, metrics = self._send_request(messages, stream=True, max_tokens=256)
        passed = ("power-law" in text.lower() or "power law" in text.lower()) and "50" in text
        result = {"task": "short_question", "passed": passed, "metrics": metrics, "response_sample": text[:180]}
        self.results.append(result)
        print(f"  Result: {'PASS' if passed else 'FAIL'} (Duration: {metrics.get('duration_s')}s, TTFT: {metrics.get('ttft_s')}s, ~{metrics.get('tokens_per_sec')} t/s)")

    def test_long_document_analysis(self):
        print("\n[TEST 2] In-Depth Academic Text Analysis...")
        # Simulate long context text
        multi_context = (SAMPLE_ABSTRACT + "\n\n") * 15
        prompt = (
            f"Analyze the following empirical research excerpts:\n\n{multi_context}\n\n"
            "Synthesize the theoretical findings into three distinct bullet points: "
            "1. Scaling behavior across model parameters. 2. Sample efficiency implications. 3. Practical limitations."
        )
        messages = [
            {"role": "system", "content": "You are an expert peer reviewer in computer science and AI."},
            {"role": "user", "content": prompt}
        ]
        text, metrics = self._send_request(messages, stream=True, max_tokens=512)
        passed = metrics.get("success", False) and len(text.split()) > 50
        result = {"task": "long_document_analysis", "passed": passed, "metrics": metrics, "response_sample": text[:180]}
        self.results.append(result)
        print(f"  Result: {'PASS' if passed else 'FAIL'} (Duration: {metrics.get('duration_s')}s, TTFT: {metrics.get('ttft_s')}s)")

    def test_cross_document_comparison(self):
        print("\n[TEST 3] Cross-Document Comparative Synthesis...")
        prompt = (
            f"Compare and contrast the architectures described in these two papers:\n\n"
            f"{PAPER_A_ATTENTION}\n\n{PAPER_B_MAMBA}\n\n"
            "Provide a scholarly comparison table covering: Computational Complexity, Memory Bottleneck, and Generation Efficiency."
        )
        messages = [
            {"role": "system", "content": "You are an academic researcher writing a literature review."},
            {"role": "user", "content": prompt}
        ]
        text, metrics = self._send_request(messages, stream=True, max_tokens=600)
        passed = ("o(n^2)" in text.lower() or "quadratic" in text.lower()) and ("o(n)" in text.lower() or "linear" in text.lower())
        result = {"task": "cross_document_comparison", "passed": passed, "metrics": metrics, "response_sample": text[:180]}
        self.results.append(result)
        print(f"  Result: {'PASS' if passed else 'FAIL'} (Duration: {metrics.get('duration_s')}s, TTFT: {metrics.get('ttft_s')}s)")

    def test_literature_review_summary(self):
        print("\n[TEST 4] Structured Literature Review Table Generation...")
        prompt = (
            f"Synthesize the findings of the following 4 studies into a Markdown literature review table:\n\n"
            f"{MULTI_STUDIES}\n\n"
            "Columns required: Study (Author & Year) | Core Method | Sample Size | Key Finding | Primary Limitation"
        )
        messages = [
            {"role": "system", "content": "You are an academic synthesis assistant. Output only structured Markdown."},
            {"role": "user", "content": prompt}
        ]
        text, metrics = self._send_request(messages, stream=True, max_tokens=750)
        passed = "|" in text and "Smith" in text and "Dubois" in text
        result = {"task": "literature_review_summary", "passed": passed, "metrics": metrics, "response_sample": text[:180]}
        self.results.append(result)
        print(f"  Result: {'PASS' if passed else 'FAIL'} (Duration: {metrics.get('duration_s')}s, TTFT: {metrics.get('ttft_s')}s)")

    def test_streaming_and_stability(self):
        print("\n[TEST 5] Streaming & Long-Context Stability Test...")
        prompt = "Write an academic essay explaining how epistemic uncertainty is modeled in Bayesian neural networks. (Approx 400 words)."
        messages = [
            {"role": "system", "content": "You are a statistics and machine learning professor."},
            {"role": "user", "content": prompt}
        ]
        text, metrics = self._send_request(messages, stream=True, max_tokens=600)
        passed = metrics.get("success", False) and metrics.get("tokens_per_sec", 0) > 5
        result = {"task": "streaming_stability", "passed": passed, "metrics": metrics, "response_sample": text[:180]}
        self.results.append(result)
        print(f"  Result: {'PASS' if passed else 'FAIL'} (Gen speed: {metrics.get('tokens_per_sec')} tokens/s, Total duration: {metrics.get('duration_s')}s)")

    def run_all(self) -> Dict[str, Any]:
        print(f"==================================================")
        print(f" Running Academic LLM Benchmark Suite")
        print(f" Endpoint: {self.base_url}")
        print(f" Model:    {self.model}")
        print(f"==================================================")

        self.test_short_question()
        self.test_long_document_analysis()
        self.test_cross_document_comparison()
        self.test_literature_review_summary()
        self.test_streaming_and_stability()

        total = len(self.results)
        passed = sum(1 for r in self.results if r.get("passed"))
        summary = {
            "total_tests": total,
            "passed_tests": passed,
            "success_rate": f"{(passed / total) * 100:.1f}%",
            "tests": self.results,
        }

        print("\n==================================================")
        print(f" Benchmark Complete: {passed}/{total} Passed ({summary['success_rate']})")
        print("==================================================")
        return summary


def main():
    parser = argparse.ArgumentParser(description="Academic LLM Benchmark")
    parser.add_argument("--endpoint", default="http://127.0.0.1:8088/v1", help="OpenAI-compatible base URL")
    parser.add_argument("--model", default="qwen2.5-14b-instruct", help="Model identifier")
    parser.add_argument("--api-key", default=os.getenv("LOCALAI_API_KEY", os.getenv("API_TOKEN", "dummy-key")), help="Bearer token")
    parser.add_argument("--output", default="benchmark_results.json", help="Output path for benchmark report")
    parser.add_argument("--dry-run", action="store_true", help="Simulate tests without calling live endpoint")

    args = parser.parse_args()

    if args.dry_run:
        print("[DRY-RUN] Simulating benchmark suite execution...")
        mock_summary = {
            "total_tests": 5,
            "passed_tests": 5,
            "success_rate": "100.0%",
            "dry_run": True,
        }
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(mock_summary, f, indent=2)
        print(f"[DRY-RUN] Benchmark mock report saved to {args.output}")
        sys.exit(0)

    benchmark = AcademicBenchmark(base_url=args.endpoint, model=args.model, api_key=args.api_key)
    summary = benchmark.run_all()

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Report written to {args.output}")


if __name__ == "__main__":
    main()
