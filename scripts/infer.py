#!/usr/bin/env python3

import json
import os
import numpy as np
import sentencepiece as spm

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

def inference(rknn_enc, rknn_dec, lm_weight, lm_bias, input_path, enc_len, dec_len):
    config = load_config(f"{input_path}/config.json")
    decoder_start_token_id = config.get("decoder_start_token_id", 59513)
    pad_token_id = config.get("pad_token_id", decoder_start_token_id)
    eos_token_id = config.get("eos_token_id", 0)
    unk_token_id = config.get("unk_token_id", 0)

    vocab, vocab_inv = load_vocab(f"{input_path}/vocab.json")
    spm_src = spm.SentencePieceProcessor(model_file=f"{input_path}/source.spm")
    spm_tgt = spm.SentencePieceProcessor(model_file=f"{input_path}/target.spm")

    print("Enter text to translate (empty line to quit):")
    while True:
        line = input("> ").strip()
        if not line:
            break
        pieces = spm_src.encode(line, out_type=str)
        tokens = [vocab.get(piece, unk_token_id) for piece in pieces]

        encoder_input_ids, attention_mask = prepare_encoder_inputs(
            tokens, enc_len, pad_token_id, eos_token_id
        )

        encoder_outputs = rknn_enc.inference(
            inputs=[encoder_input_ids, attention_mask]
        )
        encoder_hidden_states = encoder_outputs[0]

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

        decoded_pieces = []
        for token_id in output_tokens:
            if token_id in (eos_token_id, pad_token_id) or token_id <= 0:
                break
            piece = vocab_inv.get(token_id)
            if piece is not None:
                decoded_pieces.append(piece)

        translated = spm_tgt.decode(decoded_pieces)
        print(translated)
