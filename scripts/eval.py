#!/usr/bin/env python3

import argparse
import time

import sacrebleu
import sentencepiece as spm

from counters import Counters
from rknn_infer import (
    DEFAULT_DEC_LEN,
    DEFAULT_ENC_LEN,
    load_config,
    load_lm_weights,
    load_rknn_model,
    load_vocab,
    translate_line,
)


SUPPORTED_METRICS = {"bleu", "chrf"}


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate RKNN Marian translation (BLEU/chrF).")
    parser.add_argument("model_path", help="Path to the directory containing the model files.")
    parser.add_argument("input_file", help="Path to text file containing input lines.")
    parser.add_argument("reference_files", nargs="+", help="Reference translation file(s).")
    parser.add_argument(
        "--metrics",
        default="bleu,chrf",
        help="Comma-separated list of metrics to compute (bleu,chrf).",
    )
    parser.add_argument(
        "--bleu-tokenize",
        default="13a",
        help="Tokenizer to use for BLEU (sacrebleu).",
    )
    parser.add_argument("--beam-search", action="store_true", help="Use beam search decoding instead of greedy.")
    parser.add_argument("--beam-depth", type=int, default=None, help="Maximum decoding depth for beam search.")
    parser.add_argument("--beam-width", type=int, default=4, help="Beam width for beam search decoding.")
    parser.add_argument("--enc-len", type=int, default=DEFAULT_ENC_LEN, help="Encoder sequence length.")
    parser.add_argument("--dec-len", type=int, default=DEFAULT_DEC_LEN, help="Decoder sequence length.")
    return parser.parse_args()


def read_lines(path):
    with open(path, "r", encoding="utf-8") as handle:
        return [line.rstrip("\n") for line in handle]


def validate_reference_lengths(reference_sets, expected_length):
    for index, reference in enumerate(reference_sets, start=1):
        if len(reference) != expected_length:
            raise ValueError(
                f"Reference file {index} has {len(reference)} lines; expected {expected_length}."
            )


def parse_metrics(metrics_value):
    metrics = {entry.strip().lower() for entry in metrics_value.split(",") if entry.strip()}
    unknown = metrics - SUPPORTED_METRICS
    if unknown:
        raise ValueError(f"Unsupported metrics: {', '.join(sorted(unknown))}")
    if not metrics:
        raise ValueError("No metrics specified.")
    return metrics


def compute_metrics(translations, references, metrics, bleu_tokenize):
    results = []
    if "bleu" in metrics:
        bleu = sacrebleu.corpus_bleu(translations, references, tokenize=bleu_tokenize)
        results.append(("BLEU", bleu.score, bleu.signature))
    if "chrf" in metrics:
        chrf = sacrebleu.corpus_chrf(translations, references)
        results.append(("chrF", chrf.score, chrf.signature))
    return results


def main():
    args = parse_args()
    metrics = parse_metrics(args.metrics)

    input_lines = read_lines(args.input_file)
    if not input_lines:
        raise ValueError(f"No input lines found in: {args.input_file}")

    reference_sets = [read_lines(path) for path in args.reference_files]
    validate_reference_lengths(reference_sets, len(input_lines))

    config = load_config(f"{args.model_path}/config.json")
    decoder_start_token_id = config.get("decoder_start_token_id", 59513)
    pad_token_id = config.get("pad_token_id", decoder_start_token_id)
    eos_token_id = config.get("eos_token_id", 0)
    unk_token_id = config.get("unk_token_id", 0)

    model_dim = config.get("d_model")
    if model_dim is None:
        raise ValueError("Missing 'd_model' in config.json")

    vocab_size = config.get("vocab_size")
    if vocab_size is None:
        raise ValueError("Missing 'vocab_size' in config.json")

    vocab, vocab_inv = load_vocab(f"{args.model_path}/vocab.json", vocab_size)

    lm_weight, lm_bias = load_lm_weights(args.model_path)
    if lm_weight.size % vocab_size != 0:
        raise ValueError("LM weight size is not divisible by vocab size.")
    if lm_bias.size % vocab_size != 0:
        raise ValueError("LM bias size is not divisible by vocab size.")

    hidden_size = lm_weight.size // vocab_size
    if hidden_size != model_dim:
        raise ValueError("LM weight size not compatible with hidden size.")

    lm_weight = lm_weight.reshape(vocab_size, hidden_size)

    rknn_enc = load_rknn_model(f"{args.model_path}/encoder.rknn")
    rknn_dec = load_rknn_model(f"{args.model_path}/decoder.rknn")

    spm_src = spm.SentencePieceProcessor(model_file=f"{args.model_path}/source.spm")
    spm_tgt = spm.SentencePieceProcessor(model_file=f"{args.model_path}/target.spm")

    beam_search = (args.beam_width, args.beam_depth) if args.beam_search else None

    total_counters = Counters()
    total_counters.reset()
    translations = []
    start_time = time.perf_counter()

    try:
        for line in input_lines:
            counters = Counters()
            counters.reset()
            translated = translate_line(
                line,
                rknn_enc,
                rknn_dec,
                lm_weight,
                lm_bias,
                spm_src,
                spm_tgt,
                vocab,
                vocab_inv,
                args.enc_len,
                args.dec_len,
                decoder_start_token_id,
                pad_token_id,
                eos_token_id,
                unk_token_id,
                beam_search,
                counters=counters,
            )
            translations.append(translated)
            total_counters.accumulate(counters)
    finally:
        rknn_enc.release()
        rknn_dec.release()

    elapsed_s = time.perf_counter() - start_time

    results = compute_metrics(translations, reference_sets, metrics, args.bleu_tokenize)

    print("Evaluation complete")
    print(f"Elapsed: {elapsed_s:.3f} s")
    print(f"Sentences: {len(input_lines)}")
    if elapsed_s > 0:
        print(f"Sentences/sec: {len(input_lines) / elapsed_s:.3f}")

    for name, score, signature in results:
        print(f"{name}: {score:.2f} (signature: {signature})")


if __name__ == "__main__":
    main()
