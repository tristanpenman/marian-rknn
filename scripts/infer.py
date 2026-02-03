#!/usr/bin/env python3

import argparse
import json
import os
import numpy as np
import sentencepiece as spm

DEFAULT_ENC_LEN = 32
DEFAULT_DEC_LEN = 32

MODEL_DIM = 512

ENC_IN_INPUT_IDS_IDX = 0
ENC_IN_ATTENTION_MASK_IDX = 1

DEC_IN_INPUT_IDS_IDX = 0
DEC_IN_ATTENTION_MASK_IDX = 1

def load_config(config_path):
    if not os.path.exists(config_path):
        return {}
    with open(config_path, "r", encoding="utf-8") as handle:
        return json.load(handle)

def load_vocab(vocab_path):
    with open(vocab_path, "r", encoding="utf-8") as handle:
        vocab = json.load(handle)
    vocab_inv = {}
    for token, token_id in vocab.items():
        if token_id in vocab_inv:
            raise ValueError(f"Duplicate vocab id detected: {token_id}")
        vocab_inv[token_id] = token
    return vocab, vocab_inv

def load_rknn_model(model_path):
    # pylint: disable-next=import-error
    from rknnlite.api import RKNNLite

    rknn = RKNNLite()
    ret = rknn.load_rknn(model_path)
    if ret != 0:
        raise RuntimeError(f"Failed to load RKNN model: {model_path}")
    if rknn.init_runtime() != 0:
        raise RuntimeError(f"Failed to initialize RKNN runtime: {model_path}")
    return rknn

def load_lm_weights(model_path, vocab_size):
    weight_path = f"{model_path}/lm_weight.raw"
    bias_path = f"{model_path}/lm_bias.raw"
    weight = np.fromfile(weight_path, dtype=np.float32)
    bias = np.fromfile(bias_path, dtype=np.float32)
    if weight.size % vocab_size != 0:
        raise ValueError("LM weight size is not divisible by vocab size.")
    hidden_size = weight.size // vocab_size
    weight = weight.reshape(vocab_size, hidden_size)
    if bias.size != vocab_size:
        raise ValueError("LM bias size does not match vocab size.")
    return weight, bias

def build_attention_mask(input_ids, eos_token_id):
    attention = np.ones_like(input_ids, dtype=np.int32)
    eos_positions = np.where(input_ids == eos_token_id)[1]
    if eos_positions.size > 0:
        eos_index = eos_positions[0]
        attention[:, eos_index + 1 :] = 0
    return attention

def prepare_encoder_inputs(tokens, enc_len, pad_token_id, eos_token_id):
    input_ids = np.full((1, enc_len), pad_token_id, dtype=np.int32)
    length = min(len(tokens), enc_len)
    if length:
        input_ids[0, :length] = tokens[:length]
    if length < enc_len:
        input_ids[0, length] = eos_token_id
    attention_mask = build_attention_mask(input_ids, eos_token_id)
    return input_ids, attention_mask

def beam_search_decode(
    rknn_dec,
    attention_mask,
    encoder_hidden_states,
    weight,
    bias,
    decoder_start_token_id,
    pad_token_id,
    eos_token_id,
    dec_len,
    beam_width,
    beam_depth
):
    max_steps = min(dec_len - 1, beam_depth) if beam_depth else dec_len - 1
    beams = [([], 0.0, False)]

    for step in range(max_steps):
        candidates = []
        for tokens, score, finished in beams:
            if finished:
                candidates.append((tokens, score, True))
                continue

            decoder_input_ids = np.full((1, dec_len), pad_token_id, dtype=np.int32)
            decoder_input_ids[0, 0] = decoder_start_token_id
            for index, token in enumerate(tokens):
                position = index + 1
                if position < dec_len:
                    decoder_input_ids[0, position] = token

            outputs = rknn_dec.inference(
                inputs=[decoder_input_ids, attention_mask, encoder_hidden_states]
            )
            decoder_output = outputs[0]
            hidden = decoder_output[0, step, :].astype(np.float32)
            logits = hidden @ weight.T + bias

            logits -= np.max(logits)
            exp_logits = np.exp(logits)
            probs = exp_logits / np.sum(exp_logits)

            top_indices = np.argsort(probs)[-beam_width:][::-1]
            for token_id in top_indices:
                token_id = int(token_id)
                new_tokens = tokens + [token_id]
                new_score = score + float(np.log(probs[token_id] + 1e-9))
                candidates.append((new_tokens, new_score, token_id == eos_token_id))

        candidates.sort(key=lambda item: item[1], reverse=True)
        beams = candidates[:beam_width]
        if all(finished for _, _, finished in beams):
            break

    best_tokens = max(beams, key=lambda item: item[1])[0] if beams else []
    return best_tokens

def greedy_decode(
    rknn_dec,
    attention_mask,
    encoder_hidden_states,
    weight,
    bias,
    decoder_start_token_id,
    pad_token_id,
    eos_token_id,
    dec_len,
):
    decoder_input_ids = np.full((1, dec_len), pad_token_id, dtype=np.int32)
    decoder_input_ids[0, 0] = decoder_start_token_id
    output_tokens = []

    for step in range(dec_len - 1):
        outputs = rknn_dec.inference(
            inputs=[decoder_input_ids, attention_mask, encoder_hidden_states]
        )
        decoder_output = outputs[0]
        hidden = decoder_output[0, step, :].astype(np.float32)
        logits = hidden @ weight.T + bias
        next_token = int(np.argmax(logits))
        output_tokens.append(next_token)
        if step + 1 < dec_len:
            decoder_input_ids[0, step + 1] = next_token
        if next_token == eos_token_id:
            break

    return output_tokens

def translate_line(
    line,
    rknn_enc,
    rknn_dec,
    lm_weight,
    lm_bias,
    spm_src,
    spm_tgt,
    vocab,
    vocab_inv,
    enc_len,
    dec_len,
    decoder_start_token_id,
    pad_token_id,
    eos_token_id,
    unk_token_id,
    beam_search
):
    pieces = spm_src.encode(line, out_type=str)
    tokens = [vocab.get(piece, unk_token_id) for piece in pieces]

    encoder_input_ids, attention_mask = prepare_encoder_inputs(
        tokens, enc_len, pad_token_id, eos_token_id
    )

    encoder_outputs = rknn_enc.inference(
        inputs=[encoder_input_ids, attention_mask]
    )
    encoder_hidden_states = encoder_outputs[0]

    if beam_search is None:
        output_tokens = greedy_decode(
            rknn_dec,
            attention_mask,
            encoder_hidden_states,
            lm_weight,
            lm_bias,
            decoder_start_token_id,
            pad_token_id,
            eos_token_id,
            dec_len,
        )
    else:
        beam_width, beam_depth = beam_search
        output_tokens = beam_search_decode(
            rknn_dec,
            attention_mask,
            encoder_hidden_states,
            lm_weight,
            lm_bias,
            decoder_start_token_id,
            pad_token_id,
            eos_token_id,
            dec_len,
            beam_width,
            beam_depth
        )

    decoded_pieces = []
    for token_id in output_tokens:
        if token_id in (eos_token_id, pad_token_id) or token_id <= 0:
            break
        piece = vocab_inv.get(token_id)
        if piece is not None:
            decoded_pieces.append(piece)

    return spm_tgt.decode(decoded_pieces)

def inference(rknn_enc, rknn_dec, lm_weight, lm_bias, input_path, enc_len=None, dec_len=None, beam_search=None, lines=None):
    config = load_config(f"{input_path}/config.json")
    d_model = config.get("d_model", 0)
    if d_model != MODEL_DIM:
        raise ValueError(f"Model dimension mismatch: expected {MODEL_DIM}, got {d_model}")

    decoder_start_token_id = config.get("decoder_start_token_id", 59513)
    pad_token_id = config.get("pad_token_id", decoder_start_token_id)
    eos_token_id = config.get("eos_token_id", 0)
    unk_token_id = config.get("unk_token_id", 0)

    vocab, vocab_inv = load_vocab(f"{input_path}/vocab.json")

    spm_src = spm.SentencePieceProcessor(model_file=f"{input_path}/source.spm")
    spm_tgt = spm.SentencePieceProcessor(model_file=f"{input_path}/target.spm")

    def do_translate(line):
        return translate_line(
            line,
            rknn_enc,
            rknn_dec,
            lm_weight,
            lm_bias,
            spm_src,
            spm_tgt,
            vocab,
            vocab_inv,
            enc_len,
            dec_len,
            decoder_start_token_id,
            pad_token_id,
            eos_token_id,
            unk_token_id,
            beam_search
        )

    if lines is not None:
        for line in lines:
            if not line:
                continue
            print(do_translate(line))
        return

    print("Enter text to translate (empty line to quit):")
    while True:
        line = input("> ").strip()
        if not line:
            break
        print(do_translate(line))

def parse_args():
    parser = argparse.ArgumentParser(description="Run RKNN Marian translation.")
    parser.add_argument("model_path", help="Path to the directory containing the model files.")
    parser.add_argument("--beam-search", action="store_true", help="Use beam search decoding instead of greedy decoding.")
    parser.add_argument("--beam-depth", type=int, default=None, help="Maximum decoding depth for beam search.")
    parser.add_argument("--beam-width", type=int, default=4, help="Beam width for beam search decoding.")
    parser.add_argument("--enc-len", type=int, default=DEFAULT_ENC_LEN, help="Encoder sequence length (default: 32).")
    parser.add_argument("--dec-len", type=int, default=DEFAULT_DEC_LEN, help="Decoder sequence length (default: 32).")
    parser.add_argument(
        "inputs",
        nargs="*",
        help="Optional text strings to translate (quote to preserve spaces).",
    )
    return parser.parse_args()

def main():
    args = parse_args()

    # vocab
    vocab, _ = load_vocab(f"{args.model_path}/vocab.json")

    # lm head
    lm_weight, lm_bias = load_lm_weights(args.model_path, len(vocab))

    # models
    rknn_enc = load_rknn_model(f"{args.model_path}/encoder.rknn")
    rknn_dec = load_rknn_model(f"{args.model_path}/decoder.rknn")

    # optionals
    beam_search = (args.beam_width, args.beam_depth) if args.beam_search else None
    lines = args.inputs if args.inputs else None

    try:
        inference(
            rknn_enc,
            rknn_dec,
            lm_weight,
            lm_bias,
            args.model_path,
            enc_len=args.enc_len,
            dec_len=args.dec_len,
            beam_search=beam_search,
            lines=lines
        )
    finally:
        rknn_enc.release()
        rknn_dec.release()

if __name__ == "__main__":
    main()
