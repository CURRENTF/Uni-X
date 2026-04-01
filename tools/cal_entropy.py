import argparse
import csv
import json
import os
from pathlib import Path
from typing import Iterable, Iterator, List, Optional

import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer


def _resolve_device(device: Optional[str]) -> str:
    if device is None:
        return "cuda:0" if torch.cuda.is_available() else "cpu"
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is not available in this environment.")
    return device


def _prepare_ngram_views(
    token_sequences: List[torch.LongTensor],
    n: int,
    device: str,
    concatenate_sequences: bool,
):
    if concatenate_sequences:
        long_seq = torch.cat(token_sequences, dim=0).to(device)
        if long_seq.numel() < n:
            return None, None
        return long_seq.unfold(0, n, 1), long_seq.unfold(0, n - 1, 1)

    all_ngrams = []
    all_prefixes = []
    for seq in token_sequences:
        if seq.numel() < n:
            continue
        seq = seq.to(device)
        all_ngrams.append(seq.unfold(0, n, 1))
        all_prefixes.append(seq.unfold(0, n - 1, 1))

    if not all_ngrams:
        return None, None

    return torch.cat(all_ngrams, dim=0), torch.cat(all_prefixes, dim=0)


def calculate_ngram_conditional_entropy(
    token_sequences: List[torch.LongTensor],
    n: int = 2,
    device: str = "cpu",
    concatenate_sequences: bool = False,
) -> float:
    """
    Compute the n-gram conditional entropy for a list of token sequences.

    When `concatenate_sequences` is enabled, all sequences are concatenated
    before extracting n-grams. This matches the original high-throughput code
    path used during the paper experiments.
    """
    if not isinstance(n, int) or n < 1:
        raise ValueError("n must be an integer greater than or equal to 1.")
    if not token_sequences:
        return 0.0

    if n == 1:
        all_tokens = torch.cat(token_sequences, dim=0).to(device)
        total_tokens = all_tokens.numel()
        if total_tokens == 0:
            return 0.0

        _, counts = torch.unique(all_tokens, return_counts=True)
        probabilities = counts.to(dtype=torch.float64) / total_tokens
        entropy = -torch.sum(probabilities * torch.log2(probabilities))
        return entropy.item()

    all_ngrams, all_prefixes = _prepare_ngram_views(
        token_sequences=token_sequences,
        n=n,
        device=device,
        concatenate_sequences=concatenate_sequences,
    )
    if all_ngrams is None or all_prefixes is None:
        return 0.0

    total_ngrams = all_ngrams.shape[0]
    unique_ngrams, ngram_counts = torch.unique(all_ngrams, dim=0, return_counts=True)
    unique_prefixes, prefix_counts = torch.unique(all_prefixes, dim=0, return_counts=True)

    prefix_counts_map = {
        tuple(prefix.tolist()): count.item()
        for prefix, count in zip(unique_prefixes.cpu(), prefix_counts.cpu())
    }
    ngram_prefixes = unique_ngrams[:, :-1].cpu()
    corresponding_prefix_counts = torch.tensor(
        [prefix_counts_map[tuple(prefix.tolist())] for prefix in ngram_prefixes],
        device=device,
        dtype=torch.float64,
    )

    ngram_counts = ngram_counts.to(device=device, dtype=torch.float64)
    p_ngram = ngram_counts / total_ngrams
    p_conditional = ngram_counts / corresponding_prefix_counts
    valid = p_conditional > 0

    entropy = -torch.sum(p_ngram[valid] * torch.log2(p_conditional[valid]))
    return entropy.item()


def _worker_calculate_entropy_chunk(args):
    chunk, total_ngrams = args
    chunk_prefixes = chunk[:, :-1]

    unique_ngrams, ngram_inverse, ngram_counts = torch.unique(
        chunk,
        dim=0,
        return_inverse=True,
        return_counts=True,
    )
    _, prefix_inverse, prefix_counts = torch.unique(
        chunk_prefixes,
        dim=0,
        return_inverse=True,
        return_counts=True,
    )

    map_ngram_id_to_prefix_id = torch.empty(unique_ngrams.shape[0], dtype=torch.long)
    map_ngram_id_to_prefix_id.scatter_(0, ngram_inverse, prefix_inverse)
    corresponding_prefix_counts = prefix_counts[map_ngram_id_to_prefix_id]

    ngram_counts = ngram_counts.to(dtype=torch.float64)
    p_ngram = ngram_counts / total_ngrams
    p_conditional = ngram_counts / corresponding_prefix_counts.to(dtype=torch.float64)
    valid = p_conditional > 0

    chunk_entropy = -torch.sum(p_ngram[valid] * torch.log2(p_conditional[valid]))
    return chunk_entropy.item()


def calculate_ngram_conditional_entropy_chunked(
    token_sequences: List[torch.LongTensor],
    n: int = 2,
    num_workers: Optional[int] = None,
) -> float:
    """
    Approximate large-scale entropy with CPU chunking.

    This mirrors the original experimental fallback for very large token streams.
    It requires the sequences to be treated as one concatenated stream.
    """
    if not isinstance(n, int) or n < 1:
        raise ValueError("n must be an integer greater than or equal to 1.")
    if not token_sequences:
        return 0.0
    if n == 1:
        return calculate_ngram_conditional_entropy(token_sequences, n=1, device="cpu")

    if num_workers is None:
        num_workers = min(os.cpu_count() or 1, 16)
    num_workers = max(1, num_workers)

    long_seq = torch.cat(token_sequences, dim=0)
    if long_seq.numel() < n:
        return 0.0

    all_ngrams = long_seq.unfold(0, n, 1)
    total_ngrams = all_ngrams.shape[0]
    if total_ngrams == 0:
        return 0.0

    indices = torch.arange(total_ngrams)
    for col in range(n - 1, -1, -1):
        indices = indices[torch.argsort(all_ngrams[indices, col], stable=True)]
    all_ngrams = all_ngrams[indices]

    chunks = torch.tensor_split(all_ngrams, num_workers, dim=0)
    tasks = [(chunk, total_ngrams) for chunk in chunks if chunk.numel() > 0]
    if not tasks:
        return 0.0

    if num_workers == 1:
        return sum(_worker_calculate_entropy_chunk(task) for task in tasks)

    import multiprocessing as mp

    total_entropy = 0.0
    with mp.Pool(processes=num_workers) as pool:
        progress = tqdm(
            pool.imap_unordered(_worker_calculate_entropy_chunk, tasks),
            total=len(tasks),
            desc="Processing entropy chunks",
        )
        for chunk_entropy in progress:
            total_entropy += chunk_entropy
    return total_entropy


def _iter_jsonl(path: Path) -> Iterator[dict]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _load_samples(args) -> Iterable[dict]:
    if args.input_format == "hf":
        if not args.dataset:
            raise ValueError("--dataset is required when --input-format hf is used.")
        return load_dataset(
            args.dataset,
            args.dataset_config,
            split=args.split,
            streaming=args.streaming,
        )

    if not args.input_jsonl:
        raise ValueError("--input-jsonl is required when --input-format jsonl is used.")
    return _iter_jsonl(Path(args.input_jsonl))


def _parse_token_value(value) -> Optional[torch.LongTensor]:
    if value is None:
        return None
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return None
        value = json.loads(value)
    if not isinstance(value, list):
        raise TypeError(f"Expected a list or JSON string of tokens, got {type(value)}.")

    tensor = torch.tensor(value, dtype=torch.long)
    if tensor.numel() == 0:
        return None
    return tensor.reshape(-1)


def collect_token_sequences(args) -> List[torch.LongTensor]:
    tokenizer = None
    if args.mode == "text":
        if not args.tokenizer:
            raise ValueError("--tokenizer is required when --mode text is used.")
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    token_sequences: List[torch.LongTensor] = []
    progress = tqdm(desc=f"Loading {args.source_name}", total=args.max_samples)

    for sample in _load_samples(args):
        if args.mode == "text":
            text = sample.get(args.text_key)
            if not isinstance(text, str) or not text.strip():
                continue
            token_ids = tokenizer(
                text,
                add_special_tokens=False,
                return_tensors="pt",
            )["input_ids"][0]
        else:
            token_ids = _parse_token_value(sample.get(args.token_key))
            if token_ids is None:
                continue

        if token_ids.numel() == 0:
            continue

        token_sequences.append(token_ids.cpu())
        progress.update(1)

        if args.max_samples and len(token_sequences) >= args.max_samples:
            break

    progress.close()

    if not token_sequences:
        raise ValueError("No valid token sequences were loaded from the provided input.")
    return token_sequences


def write_results_csv(output_csv: Path, rows: List[dict], append: bool) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    file_exists = output_csv.exists()

    with output_csv.open("a" if append else "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "source",
                "n",
                "entropy",
                "mode",
                "method",
                "concatenate_sequences",
            ],
        )
        if not append or not file_exists:
            writer.writeheader()
        writer.writerows(rows)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute n-gram conditional entropy for text or token sequences.",
    )
    parser.add_argument("--source-name", required=True, help="Label used in printed logs and CSV output.")
    parser.add_argument(
        "--input-format",
        choices=["hf", "jsonl"],
        required=True,
        help="Read from a Hugging Face dataset or a local JSONL file.",
    )
    parser.add_argument("--dataset", help="Dataset name or local dataset path for --input-format hf.")
    parser.add_argument("--dataset-config", help="Optional dataset config name.")
    parser.add_argument("--split", default="train", help="Dataset split for --input-format hf.")
    parser.add_argument("--streaming", action="store_true", help="Enable Hugging Face streaming mode.")
    parser.add_argument("--input-jsonl", help="Path to a local JSONL file.")

    parser.add_argument(
        "--mode",
        choices=["text", "tokens"],
        required=True,
        help="Use raw text with a tokenizer or pre-tokenized sequences.",
    )
    parser.add_argument("--text-key", default="text", help="Field name containing raw text.")
    parser.add_argument("--token-key", default="tokens", help="Field name containing token sequences.")
    parser.add_argument("--tokenizer", help="Tokenizer path or model name used when --mode text is selected.")

    parser.add_argument(
        "--n-values",
        type=int,
        nargs="+",
        default=[1, 2, 3, 4],
        help="List of n values to evaluate.",
    )
    parser.add_argument(
        "--method",
        choices=["exact", "chunked"],
        default="exact",
        help="Exact computation or chunked CPU approximation for very large streams.",
    )
    parser.add_argument("--device", help="Torch device for exact mode. Defaults to cuda:0 when available.")
    parser.add_argument(
        "--concatenate-sequences",
        action="store_true",
        help="Concatenate all sequences before extracting n-grams. This matches the original paper code path.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=min(os.cpu_count() or 1, 16),
        help="Worker count used by the chunked CPU approximation.",
    )
    parser.add_argument("--max-samples", type=int, help="Optional cap on the number of loaded sequences.")
    parser.add_argument("--output-csv", help="Optional CSV path for saving entropy results.")
    parser.add_argument("--append", action="store_true", help="Append to an existing CSV instead of overwriting it.")
    return parser.parse_args()


def main():
    args = parse_args()
    device = _resolve_device(args.device)
    token_sequences = collect_token_sequences(args)

    rows = []
    for n in args.n_values:
        if args.method == "chunked":
            if not args.concatenate_sequences:
                raise ValueError(
                    "--method chunked requires --concatenate-sequences because it operates on one token stream."
                )
            entropy = calculate_ngram_conditional_entropy_chunked(
                token_sequences,
                n=n,
                num_workers=args.num_workers,
            )
        else:
            entropy = calculate_ngram_conditional_entropy(
                token_sequences,
                n=n,
                device=device,
                concatenate_sequences=args.concatenate_sequences,
            )

        print(f"{args.source_name}, n={n} entropy = {entropy:.4f} bits")
        rows.append(
            {
                "source": args.source_name,
                "n": n,
                "entropy": round(entropy, 6),
                "mode": args.mode,
                "method": args.method,
                "concatenate_sequences": args.concatenate_sequences,
            }
        )

    if args.output_csv:
        write_results_csv(Path(args.output_csv), rows, append=args.append)


if __name__ == "__main__":
    main()
