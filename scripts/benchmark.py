#!/usr/bin/env python3

import argparse
import time

import sentencepiece as spm

from counters import Counters
from rknn_infer import (
    DEFAULT_DEC_LEN,
    DEFAULT_ENC_LEN,
    load_config,
    load_lm_weights,
    load_rknn_model,
    load_vocab,
    translate_line
)

def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark RKNN Marian translation (Python).")
    parser.add_argument("model_path", help="Path to the directory containing the model files.")
    parser.add_argument("input_file", help="Path to text file containing input lines.")
    parser.add_argument("max_seconds", type=float, help="Maximum time to run the benchmark.")
    parser.add_argument("--beam-search", action="store_true", help="Use beam search decoding instead of greedy.")
    parser.add_argument("--beam-depth", type=int, default=None, help="Maximum decoding depth for beam search.")
    parser.add_argument("--beam-width", type=int, default=4, help="Beam width for beam search decoding.")
    parser.add_argument("--enc-len", type=int, default=DEFAULT_ENC_LEN, help="Encoder sequence length.")
    parser.add_argument("--dec-len", type=int, default=DEFAULT_DEC_LEN, help="Decoder sequence length.")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.max_seconds <= 0:
        raise ValueError("max_seconds must be greater than 0")

    with open(args.input_file, "r", encoding="utf-8") as handle:
        input_lines = [line.strip() for line in handle if line.strip()]

    if not input_lines:
        raise ValueError(f"No non-empty input lines found in: {args.input_file}")

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
    total_sentences = 0
    index = 0
    start_time = time.perf_counter()

    try:
        while True:
            if (time.perf_counter() - start_time) >= args.max_seconds:
                break

            counters = Counters()
            counters.reset()
            translate_line(
                input_lines[index],
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
            total_counters.accumulate(counters)
            total_sentences += 1
            index = (index + 1) % len(input_lines)
    finally:
        rknn_enc.release()
        rknn_dec.release()

    elapsed_s = time.perf_counter() - start_time
    print("Benchmark complete")
    print(f"Elapsed: {elapsed_s:.3f} s")
    print(f"Sentences: {total_sentences}")
    if elapsed_s > 0:
        print(f"Sentences/sec: {total_sentences / elapsed_s:.3f}")

    if total_sentences > 0:
        print(f"Total time: {total_counters.total_ms:.3f} ms")
        print(f"Encoder time: {total_counters.encoder_ms:.3f} ms")
        print(f"Decoder time: {total_counters.decoder_ms:.3f} ms")
        print(f"LM head time: {total_counters.lm_head_ms:.3f} ms")

        print(f"Avg total time per sentence: {total_counters.total_ms / total_sentences:.3f} ms")
        print(f"Avg encoder time per sentence: {total_counters.encoder_ms / total_sentences:.3f} ms")
        print(f"Avg decoder time per sentence: {total_counters.decoder_ms / total_sentences:.3f} ms")
        print(f"Avg LM head time per sentence: {total_counters.lm_head_ms / total_sentences:.3f} ms")

        print(f"Input tokens: {total_counters.input_tokens}")
        print(f"Output tokens: {total_counters.output_tokens}")
        print(f"Decoder iterations: {total_counters.decoder_iterations}")
        if elapsed_s > 0:
            print(f"Input tokens/sec: {total_counters.input_tokens / elapsed_s:.3f}")
            print(f"Output tokens/sec: {total_counters.output_tokens / elapsed_s:.3f}")


if __name__ == "__main__":
    main()
