#!/usr/bin/env python3
"""
Evaluation Harness for RAG Chatbot

- Reads experiments/queries.csv with columns [id,query]
- For each row:
  * start a timer
  * call run_query(query, params)
  * stop the timer
- Writes CSV to experiments/results/{timestamp}_{run_id}.csv with columns:
  [timestamp,run_id,query_id,query_text,top_k,temperature,max_tokens,latency_ms,
   retrieved_titles,answer_text]
- Argparse flags: --run-id, --top-k, --temperature, --max-tokens
- Defaults: top_k=5, temperature=0.1, max_tokens=1000

Notes:
- Uses the existing run_query() from rag/query_engine.py
- Keeps interactive app behavior unchanged
"""

import argparse
import csv
import os
from pathlib import Path
from datetime import datetime
import sys
import time

# Add project paths
ROOT = Path(__file__).resolve().parents[1]
RAG_DIR = ROOT / "rag"
if str(RAG_DIR) not in sys.path:
    sys.path.append(str(RAG_DIR))

from query_engine import run_query  # type: ignore


def read_queries(csv_path: Path):
    queries = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            qid = row.get("id")
            qtext = row.get("query")
            if qid is None or qtext is None:
                continue
            qtext = qtext.strip()
            if not qtext:
                continue
            queries.append({"id": qid, "query": qtext})
    return queries


def ensure_results_dir(results_dir: Path):
    results_dir.mkdir(parents=True, exist_ok=True)


def main():
    parser = argparse.ArgumentParser(description="Evaluation harness for RAG chatbot")
    parser.add_argument("--run-id", type=str, default="eval_v1", help="Run identifier for result file name")
    parser.add_argument("--top-k", type=int, default=5, help="Number of chunks to retrieve")
    parser.add_argument("--temperature", type=float, default=0.1, help="Generation temperature")
    parser.add_argument("--max-tokens", type=int, default=1000, help="Max tokens for generation")
    parser.add_argument("--queries-csv", type=str, default=str(ROOT / "experiments" / "queries.csv"), help="Path to input CSV with [id,query]")
    parser.add_argument("--results-dir", type=str, default=str(ROOT / "experiments" / "results"), help="Directory to store results CSV")
    args = parser.parse_args()

    queries_csv = Path(args.queries_csv)
    results_dir = Path(args.results_dir)

    if not queries_csv.exists():
        print(f"Input CSV not found: {queries_csv}")
        sys.exit(1)

    ensure_results_dir(results_dir)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_path = results_dir / f"{timestamp}_{args.run_id}.csv"

    queries = read_queries(queries_csv)
    if not queries:
        print("No queries found in input CSV.")
        sys.exit(1)

    # Prepare header and open CSV for writing
    fieldnames = [
        "timestamp",
        "run_id",
        "query_id",
        "query_text",
        "top_k",
        "temperature",
        "max_tokens",
        "latency_ms",
        "retrieved_titles",
        "answer_text",
    ]

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for item in queries:
            qid = item["id"]
            qtext = item["query"]

            params = {
                "top_k": args.top_k,
                "temperature": args.temperature,
                "max_tokens": args.max_tokens,
            }

            start = time.perf_counter()
            try:
                result = run_query(qtext, params)
            except Exception as e:
                # On failure, write an error row with minimal details
                latency_ms = int((time.perf_counter() - start) * 1000)
                writer.writerow({
                    "timestamp": timestamp,
                    "run_id": args.run_id,
                    "query_id": qid,
                    "query_text": qtext,
                    "top_k": args.top_k,
                    "temperature": args.temperature,
                    "max_tokens": args.max_tokens,
                    "latency_ms": latency_ms,
                    "retrieved_titles": "",
                    "answer_text": f"ERROR: {e}",
                })
                continue

            latency_ms = int((time.perf_counter() - start) * 1000)

            retrieved_titles = result.get("retrieved_titles", [])
            answer_text = result.get("answer_text", "")

            writer.writerow({
                "timestamp": timestamp,
                "run_id": args.run_id,
                "query_id": qid,
                "query_text": qtext,
                "top_k": args.top_k,
                "temperature": args.temperature,
                "max_tokens": args.max_tokens,
                "latency_ms": latency_ms,
                "retrieved_titles": " | ".join(retrieved_titles),
                "answer_text": answer_text,
            })

    print(f"Saved results to: {out_path}")


if __name__ == "__main__":
    main()
