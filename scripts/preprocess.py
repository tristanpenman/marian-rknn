#!/usr/bin/env python3

import argparse
import json
import sys
from pathlib import Path

import sentencepiece as spm

DEFAULT_SEQ_LEN = 32


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Convert raw text into fixed-length token-ID lines suitable for "
            "RKNN quantization datasets."
        )
    )
    parser.add_argument(
        "input_path",
        nargs="?",
        help="Optional path to an input text file. If omitted, read from stdin.",
    )
    parser.add_argument(
        "--spm-model",
        required=True,
        help="Path to SentencePiece model file.",
    )
    parser.add_argument(
        "--vocab",
        required=True,
        help="Path to vocab.json containing piece -> id mappings.",
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to config.json containing pad_token_id.",
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=DEFAULT_SEQ_LEN,
        help=f"Output sequence length (default: {DEFAULT_SEQ_LEN}).",
    )
    parser.add_argument(
        "--keep-empty",
        action="store_true",
        help="Emit padded rows for empty/whitespace-only lines.",
    )
    args = parser.parse_args()

    if args.seq_len <= 0:
        parser.error("--seq-len must be a positive integer")

    return args


def load_sentencepiece_model(path):
    """Load a SentencePiece model from disk."""
    processor = spm.SentencePieceProcessor()
    if not processor.load(path):
        raise RuntimeError(f"Failed to load SentencePiece model: {path}")
    return processor


def load_vocab(path):
    """Load and validate vocab mapping (piece -> token id)."""
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)

    if not isinstance(data, dict):
        raise ValueError(f"Vocab file must contain a JSON object: {path}")

    vocab = {}
    for piece, token_id in data.items():
        if not isinstance(token_id, int):
            raise ValueError(
                f"Vocab token ID must be int for piece {piece!r}, got {type(token_id).__name__}"
            )
        vocab[piece] = token_id
    return vocab


def load_config(path):
    """Load model config JSON file."""
    with open(path, "r", encoding="utf-8") as handle:
        config = json.load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Config file must contain a JSON object: {path}")
    return config


def resolve_pad_token_id(config):
    """Read and validate pad_token_id from config."""
    pad_token_id = config.get("pad_token_id")
    if not isinstance(pad_token_id, int):
        raise ValueError("config.json must contain an integer 'pad_token_id'")
    return pad_token_id


def iter_lines(input_path):
    """Iterate over lines from file or stdin, preserving spaces except trailing newline."""
    if input_path is None:
        for line in sys.stdin:
            yield line.rstrip("\n")
        return

    with open(input_path, "r", encoding="utf-8") as handle:
        for line in handle:
            yield line.rstrip("\n")


def pieces_to_ids(pieces, vocab, unk_id):
    """Convert sentencepiece tokens to vocab IDs."""
    ids = []
    for piece in pieces:
        token_id = vocab.get(piece)
        if token_id is None:
            if unk_id is None:
                raise KeyError(piece)
            token_id = unk_id
        ids.append(token_id)
    return ids


def shape_ids(ids, seq_len, pad_token_id):
    """Truncate/pad token IDs to fixed length."""
    clipped = ids[:seq_len]
    if len(clipped) < seq_len:
        clipped.extend([pad_token_id] * (seq_len - len(clipped)))
    return clipped


def main():
    """Entrypoint for quantization dataset preprocessing."""
    args = parse_args()

    input_path = Path(args.input_path) if args.input_path else None
    spm_model_path = Path(args.spm_model)
    vocab_path = Path(args.vocab)
    config_path = Path(args.config)

    try:
        sp = load_sentencepiece_model(str(spm_model_path))
        vocab = load_vocab(str(vocab_path))
        config = load_config(str(config_path))
        pad_token_id = resolve_pad_token_id(config)
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(1)

    unk_id = vocab.get("<unk>")

    try:
        for line_index, text in enumerate(iter_lines(str(input_path) if input_path else None), start=1):
            if not text.strip() and not args.keep_empty:
                continue

            pieces = sp.encode(text, out_type=str)
            try:
                token_ids = pieces_to_ids(pieces, vocab, unk_id)
            except KeyError as missing_piece:
                print(
                    (
                        "error: missing vocab entry for piece "
                        f"{missing_piece.args[0]!r} at line {line_index}; "
                        "<unk> token is not available in vocab.json"
                    ),
                    file=sys.stderr,
                )
                sys.exit(1)

            output_ids = shape_ids(token_ids, args.seq_len, pad_token_id)
            print(" ".join(map(str, output_ids)))
    except OSError as exc:
        print(f"error: failed to read input: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
