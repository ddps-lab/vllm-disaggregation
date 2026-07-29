# SPDX-License-Identifier: Apache-2.0
"""Minimal EP online-serving check: concurrent chat completions + latency.

With DP=N every request lands on one engine (DP load balancing); concurrent
requests exercise multiple engines, and every single request exercises EP
all-to-all across all nodes inside each MoE layer.
"""

import argparse
import statistics
import time
from concurrent.futures import ThreadPoolExecutor

from openai import OpenAI

PROMPTS = [
    "Mixture-of-Experts LLM이 무엇인지 두 문장으로 설명해줘.",
    "Expert parallelism이 뭐야? 짧게 답해줘.",
    "한국의 도시 세 곳을 말해줘.",
    "Transformer의 KV cache는 무엇을 저장해? 한 문장으로 답해줘.",
    "GPU를 주제로 짧은 시를 써줘.",
    "17 * 23은 얼마야? 숫자만 답해줘.",
    "'distributed inference'를 한국어로 번역해줘.",
    "MoE 모델은 왜 router를 사용해? 한 문장으로 답해줘.",
]


def parse_args():
    parser = argparse.ArgumentParser(description="EP online-serving client")
    parser.add_argument("--base-url", default="http://localhost:8000/v1")
    parser.add_argument("-n", "--num-requests", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=128)
    return parser.parse_args()


def run_one(client: OpenAI, model: str, prompt: str, max_tokens: int):
    start = time.perf_counter()
    completion = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
    )
    elapsed = time.perf_counter() - start
    return prompt, completion.choices[0].message.content, elapsed


def main():
    args = parse_args()
    client = OpenAI(api_key="EMPTY", base_url=args.base_url)

    model = client.models.list().data[0].id
    print(f"serving model: {model}\n")

    prompts = [PROMPTS[i % len(PROMPTS)] for i in range(args.num_requests)]
    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.num_requests) as pool:
        results = list(
            pool.map(
                lambda p: run_one(client, model, p, args.max_tokens), prompts
            )
        )
    wall = time.perf_counter() - start

    latencies = []
    for prompt, answer, elapsed in results:
        latencies.append(elapsed)
        answer = " ".join(answer.split())
        print(f"[{elapsed:6.2f}s] {prompt}")
        print(f"          -> {answer[:120]}\n")

    print(
        f"{len(results)} requests in {wall:.2f}s | "
        f"latency p50={statistics.median(latencies):.2f}s "
        f"max={max(latencies):.2f}s"
    )


if __name__ == "__main__":
    main()
