#!/usr/bin/env python3
"""
QA Pair Evaluation Harness for RAG Chatbot

- Loads experiments/qa.csv with columns [id,question,reference_answer,reference_section]
- For each row:
    * run_query(question, params)
    * collect: id, question, reference_section, answer_text, retrieved_titles, latency_ms
- Adds two scores:
    contains_reference = 1 if any title in retrieved_titles contains reference_section (case-insensitive), else 0
    rough_match = 1 if answer_text shares >= 6 uncommon words with reference_answer (lowercased; stopwords removed), else 0
- Writes experiments/results/qa_{timestamp}_{run_id}.csv including the scores.
- Prints summary: mean contains_reference, mean rough_match, average latency_ms
- Args: --run-id, --top-k, --temperature, --max-tokens (defaults: 5, 0.1, 1000)
"""

import argparse
import csv
from datetime import datetime
from pathlib import Path
import re
import sys
import time
from typing import Dict, List

# Project paths
ROOT = Path(__file__).resolve().parents[1]
RAG_DIR = ROOT / "rag"
if str(RAG_DIR) not in sys.path:
    sys.path.append(str(RAG_DIR))

from query_engine import run_query  # type: ignore

STOPWORDS = {
    "a","an","the","and","or","but","if","then","than","that","this","those","these","to","of","in","on","for","by","with","as","at","from","is","are","was","were","be","being","been","it","its","their","there","which","who","whom","what","when","where","why","how","into","about","over","under","between","through","during","before","after","above","below","up","down","out","off","again","further","more","most","some","such","no","nor","not","only","own","same","so","too","very","can","will","just","don","should","now"
}

WORD_RE = re.compile(r"[a-z0-9]+")

def tokenize_uncommon(text: str) -> List[str]:
    words = [w for w in WORD_RE.findall(text.lower()) if w not in STOPWORDS and len(w) > 2]
    return words


def read_qa(csv_path: Path) -> List[Dict[str, str]]:
    items: List[Dict[str, str]] = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if not row:
                continue
            qid = (row.get("id") or "").strip()
            question = (row.get("question") or "").strip()
            ref_ans = (row.get("reference_answer") or "").strip()
            ref_sec = (row.get("reference_section") or "").strip()
            if not qid or not question:
                continue
            items.append({
                "id": qid,
                "question": question,
                "reference_answer": ref_ans,
                "reference_section": ref_sec,
            })
    return items


def main():
    parser = argparse.ArgumentParser(description="Run QA pairs against the RAG chatbot and score results.")
    parser.add_argument("--run-id", type=str, default="qa_v1", help="Run identifier (used in output filename)")
    parser.add_argument("--top-k", type=int, default=5, help="Number of chunks to retrieve")
    parser.add_argument("--temperature", type=float, default=0.1, help="Generation temperature")
    parser.add_argument("--max-tokens", type=int, default=1000, help="Max tokens for generation")
    parser.add_argument("--qa-csv", type=str, default=str(ROOT / "experiments" / "qa.csv"), help="Path to QA CSV file")
    parser.add_argument("--results-dir", type=str, default=str(ROOT / "experiments" / "results"), help="Directory to store results")
    args = parser.parse_args()

    qa_csv = Path(args.qa_csv)
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    if not qa_csv.exists():
        print(f"QA CSV not found: {qa_csv}")
        sys.exit(1)

    qa_items = read_qa(qa_csv)
    if not qa_items:
        print("No QA rows found.")
        sys.exit(1)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_path = results_dir / f"qa_{timestamp}_{args.run_id}.csv"

    fieldnames = [
        "timestamp","run_id","id","question","reference_section","latency_ms",
        "contains_reference","rough_match","retrieved_titles","answer_text"
    ]

    contains_scores: List[int] = []
    rough_scores: List[int] = []
    latencies: List[int] = []

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for row in qa_items:
            qid = row["id"]
            question = row["question"]
            ref_answer = row["reference_answer"]
            ref_section = row["reference_section"]

            params = {
                "top_k": args.top_k,
                "temperature": args.temperature,
                "max_tokens": args.max_tokens,
            }

            start = time.perf_counter()
            try:
                result = run_query(question, params)
                answer_text = result.get("answer_text", "")
                retrieved_titles = result.get("retrieved_titles", [])
            except Exception as e:
                answer_text = f"ERROR: {e}"
                retrieved_titles = []
            latency_ms = int((time.perf_counter() - start) * 1000)

            # contains_reference score
            contains_reference = 0
            if ref_section:
                rs = ref_section.lower()
                for t in retrieved_titles:
                    if rs in (t or "").lower():
                        contains_reference = 1
                        break

            # rough_match score
            ref_tokens = set(tokenize_uncommon(ref_answer))
            ans_tokens = set(tokenize_uncommon(answer_text))
            shared = ref_tokens.intersection(ans_tokens)
            rough_match = 1 if len(shared) >= 6 else 0

            contains_scores.append(contains_reference)
            rough_scores.append(rough_match)
            latencies.append(latency_ms)

            writer.writerow({
                "timestamp": timestamp,
                "run_id": args.run_id,
                "id": qid,
                "question": question,
                "reference_section": ref_section,
                "latency_ms": latency_ms,
                "contains_reference": contains_reference,
                "rough_match": rough_match,
                "retrieved_titles": " | ".join(retrieved_titles),
                "answer_text": answer_text,
            })

    # Summary
    def mean(xs: List[int]) -> float:
        return (sum(xs) / len(xs)) if xs else 0.0

    print("\n=== QA EVAL SUMMARY ===")
    print(f"Run ID: {args.run_id}")
    print(f"Results: {out_path}")
    print(f"Mean contains_reference: {mean(contains_scores):.3f}")
    print(f"Mean rough_match: {mean(rough_scores):.3f}")
    print(f"Average latency_ms: {mean(latencies):.1f}")


if __name__ == "__main__":
    main()
