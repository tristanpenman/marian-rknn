#!/usr/bin/env python3

import argparse
from pathlib import Path
import numpy as np

from rknn.api import RKNN

from .preprocess import iter_lines, process_calibration_data, resolve_special_token_ids, load_sentencepiece_model
from .rknn_infer import inference, load_config, load_vocab, DEFAULT_DEC_LEN, DEFAULT_ENC_LEN, prepare_encoder_inputs, build_attention_mask
from .rknn_convert import convert_model, convert_weights

DEFAULT_CALIBRATION_CACHE = "calibration"

DECODER_INPUTS = ['input_ids', 'attention_mask', 'encoder_hidden_states']
ENCODER_INPUTS = ['input_ids', 'attention_mask']

def parse_arg():
    """Parse command line arguments, returning them as a tuple."""
    parser = argparse.ArgumentParser(
        description="Convert Marian ONNX models to RKNN format with quantization.",
    )
    parser.add_argument("input_path", help="Path to the directory containing the ONNX files.")
    parser.add_argument("--calibration-cache", default=DEFAULT_CALIBRATION_CACHE, help="Path to calibration cache directory.")
    parser.add_argument("--calibration-data", help="Path to text file containing calibration sentences (one per line).")
    parser.add_argument("--dynamic-input", action="store_true", default=False, help="Export model using dynamic inputs (default=off).")
    parser.add_argument("--enc-len", type=int, default=DEFAULT_ENC_LEN, help="Encoder sequence length (default: 32).")
    parser.add_argument("--dec-len", type=int, default=DEFAULT_DEC_LEN, help="Decoder sequence length (default: 32).")
    parser.add_argument(
        "platform",
        choices=[
            "rk3562",
            "rk3566",
            "rk3568",
            "rk3576",
            "rk3588",
            "rv1126b",
            "rv1109",
            "rv1126",
            "rk1808",
        ],
        help="Target platform.",
    )
    parser.add_argument(
        "dtype",
        nargs="?",
        choices=["fp"],
        help="Optional dtype (only 'fp' supported).",
    )
    parser.add_argument(
        "output_path",
        nargs="?",
        help="Optional output directory for generated RKNN files (defaults to input_path).",
    )

    args = parser.parse_args()

    if args.enc_len <= 0 or args.dec_len <= 0:
        parser.error("Encoder and decoder lengths must be positive.")

    output_path = args.output_path or args.input_path

    return args.input_path, args.platform, \
        output_path, args.dynamic_input, args.enc_len, args.dec_len, \
        args.calibration_cache, args.calibration_data


def initial_conversion(input_path, output_path, platform, dynamic_input, encoder_input_size_list, decoder_input_size_list):
    print('Converting encoder...')
    rknn_enc = convert_model(f"{input_path}/encoder.onnx", platform, dynamic_input,
                             f"{output_path}/encoder.rknn", ENCODER_INPUTS, encoder_input_size_list)

    print('Converting decoder...')
    rknn_dec = convert_model(f"{input_path}/decoder.onnx", platform, dynamic_input,
                             f"{output_path}/decoder.rknn", DECODER_INPUTS, decoder_input_size_list)

    print('Converting LM weights...')
    lm_weight = convert_weights(f"{input_path}/lm_weight.bin",
                                f"{output_path}/lm_weight.raw")

    print('Converting LM biases...')
    lm_bias = convert_weights(f"{input_path}/lm_bias.bin",
                              f"{output_path}/lm_bias.raw")

    return rknn_enc, rknn_dec, lm_weight, lm_bias


def quantize_model(
    model_path,
    platform,
    dynamic_input,
    output_path,
    inputs,
    input_size_list,
    quantization_dataset
):
    """Configure, compile, and export an RKNN model for the target platform.

    When dynamic_input is enabled, we still pass concrete sizes so the RKNN
    compiler can validate and infer shapes for the encoder/decoder interfaces.

    Warning: The C++ rknn_api fails with std::out_of_range exceptions when
    loading models with `dynamic_input` enabled.
    """
    rknn = RKNN(verbose=False)

    print('--> Config model')
    if dynamic_input:
        # Warning: The C++ rknn_api fails with std::out_of_range exceptions
        # when loading models with `dynamic_input`
        rknn.config(target_platform=platform, dynamic_input=[input_size_list])
    else:
        rknn.config(target_platform=platform)

    print('--> Loading model')
    ret = rknn.load_onnx(model=model_path,
                         inputs=inputs,
                         input_size_list=input_size_list)
    if ret != 0:
        raise RuntimeError(f"Failed to load model: {model_path}")

    print('--> Building quantized model')
    ret = rknn.build(do_quantization=True, dataset=quantization_dataset)
    if ret != 0:
        raise RuntimeError(f"Failed to build quantized model: {model_path}")

    print('--> Export rknn model')
    ret = rknn.export_rknn(output_path)
    if ret != 0:
        raise RuntimeError(f"Failed to export rknn model: {output_path}")

    return rknn


def generate_encoder_calibration_data(
    sp,
    vocab,
    unk_id,
    pad_token_id,
    eos_token_id,
    calibration_cache,
    calibration_data,
    seq_len
):
    """Prepare calibration data for quantization of the encoder."""

    output_dir = Path(calibration_cache) / "encoder"
    output_dir.mkdir(parents=True, exist_ok=True)

    txt_path = Path(calibration_cache) / "encoder.txt"

    with txt_path.open(mode='w', encoding='utf-8') as f:
        for line_index, text in enumerate(iter_lines(str(calibration_data)), start=1):
            if not text.strip():
                continue

            input_ids_npy, attention_mask_npy = process_calibration_data(
                text=text,
                line_index=line_index,
                sp=sp,
                vocab=vocab,
                unk_id=unk_id,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
                seq_len=seq_len,
                output_dir=output_dir
            )

            f.write(f"encoder/{input_ids_npy} encoder/{attention_mask_npy}\n")


def generate_decoder_calibration_data(
    sp,
    vocab,
    unk_id,
    pad_token_id,
    eos_token_id,
    decoder_start_token_id,
    calibration_cache,
    calibration_data,
    enc_len,
    dec_len,
    rknn_enc,
    rknn_dec,
    weight,
    bias
):
    """Run unquantized inference to capture encoder hidden state ranges for quantization of the decoder."""

    # This has to run all inputs through the encoder to capture hidden state ranges for the decoder quantization,
    # then we save the hidden states to disk

    output_dir = Path(calibration_cache) / "decoder"
    output_dir.mkdir(parents=True, exist_ok=True)

    txt_path = Path(calibration_cache) / "decoder.txt"

    with txt_path.open(mode='w', encoding='utf-8') as f:
        for line_index, text in enumerate(iter_lines(str(calibration_data)), start=1):
            if not text.strip():
                continue

            # tokenize the input text
            pieces = sp.encode(text, out_type=str)
            tokens = [vocab.get(piece, unk_id) for piece in pieces]

            # prepare encoder inputs and attention mask
            encoder_input_ids = prepare_encoder_inputs(tokens, enc_len, pad_token_id, eos_token_id)
            attention_mask = build_attention_mask(encoder_input_ids, eos_token_id)

            # invoke encoder
            encoder_outputs = rknn_enc.inference(inputs=[encoder_input_ids, attention_mask])
            encoder_hidden_state = encoder_outputs[0]

            # write encode hidden state in numpy format to disk
            encoder_hidden_state_npy = f"encoder_hidden_state_{line_index}.npy"
            encoder_hidden_state_path = output_dir / encoder_hidden_state_npy
            np.save(encoder_hidden_state_path, encoder_hidden_state)

            # decoder input ids should mimic inference-time autoregressive inputs:
            # start token followed by previously generated tokens.
            decoder_input_ids = np.full((1, dec_len), pad_token_id, dtype=np.int64)
            decoder_input_ids[0, 0] = decoder_start_token_id

            for step in range(dec_len - 1):
                dec_outputs = rknn_dec.inference(
                    inputs=[decoder_input_ids, attention_mask, encoder_hidden_state]
                )

                # extract hidden state for the current step
                decoder_output = dec_outputs[0]
                hidden = decoder_output[0, step, :].astype(np.float32)

                # apply LM head to logits and measure timing
                logits = hidden @ weight.T + bias

                # choose token with highest probability and add to output
                next_token = int(np.argmax(logits))

                # break on EOS...
                if next_token == eos_token_id:
                    break

                # ...or prepare for next iteration
                if step + 1 < dec_len:
                    decoder_input_ids[0, step + 1] = next_token

            print(decoder_input_ids)

            decoder_input_ids_npy = f"decoder_input_ids_{line_index}.npy"
            decoder_input_ids_path = output_dir / decoder_input_ids_npy
            np.save(decoder_input_ids_path, decoder_input_ids)

            encoder_attention_mask_npy = f"encoder_attention_mask_{line_index}.npy"
            encoder_attention_mask_path = output_dir / encoder_attention_mask_npy
            np.save(encoder_attention_mask_path, attention_mask)

            f.write(
                f"decoder/{decoder_input_ids_npy} "
                f"decoder/{encoder_attention_mask_npy} "
                f"decoder/{encoder_hidden_state_npy}\n"
            )

def quantize_encoder(onnx_enc_path, calibration_cache, output_path, platform, dynamic_input, encoder_input_size_list):
    """
    Quantize the encoder RKNN model using collected calibration data.
    """
    return quantize_model(
        model_path=onnx_enc_path,
        platform=platform,
        dynamic_input=dynamic_input,
        output_path=f"{output_path}/encoder_quant.rknn",
        inputs=ENCODER_INPUTS,
        input_size_list=encoder_input_size_list,
        quantization_dataset=f"{calibration_cache}/encoder.txt"
    )


def quantize_decoder(onnx_dec_path, calibration_cache, output_path, platform, dynamic_input, decoder_input_size_list):
    """
    Quantize the decoder RKNN model using collected calibration data.
    """
    return quantize_model(
        model_path=onnx_dec_path,
        platform=platform,
        dynamic_input=dynamic_input,
        output_path=f"{output_path}/decoder_quant.rknn",
        inputs=DECODER_INPUTS,
        input_size_list=decoder_input_size_list,
        quantization_dataset=f"{calibration_cache}/decoder.txt"
    )


def main():
    """Convert encoder/decoder ONNX models and LM head weights to RKNN assets."""
    input_path, platform, output_path, dynamic_input, enc_len, dec_len, calibration_cache, calibration_data = parse_arg()

    config = load_config(f"{input_path}/config.json")
    model_dim = config.get("d_model")
    if model_dim is None:
        raise ValueError("Missing 'd_model' in config.json")

    print('Performing initial conversion to RKNN format...')

    encoder_input_size_list = [[1, enc_len], [1, enc_len]]
    decoder_input_size_list = [[1, dec_len], [1, enc_len], [1, enc_len, model_dim]]

    rknn_enc, rknn_dec, lm_weight, lm_bias = initial_conversion(
        input_path,
        output_path,
        platform,
        dynamic_input,
        encoder_input_size_list,
        decoder_input_size_list
    )

    if rknn_enc.init_runtime(target=None) != 0:
        raise RuntimeError("Failed to initialize RKNN encoder runtime")

    if rknn_dec.init_runtime(target=None) != 0:
        raise RuntimeError("Failed to initialize RKNN decoder runtime")

    vocab_size = config.get("vocab_size")
    if vocab_size is None:
        raise ValueError("Missing 'vocab_size' in config.json")

    vocab, _vocab_inv = load_vocab(f"{input_path}/vocab.json", vocab_size)

    if lm_weight.size % vocab_size != 0:
        raise ValueError("LM weight size is not divisible by vocab size.")
    if lm_bias.size % vocab_size != 0:
        raise ValueError("LM bias size is not divisible by vocab size.")

    hidden_size = lm_weight.size // vocab_size
    if hidden_size != model_dim:
        raise ValueError("LM weight size not compatible with hidden size.")

    lm_weight = lm_weight.reshape(vocab_size, hidden_size)

    spm_src = load_sentencepiece_model(f"{input_path}/source.spm")

    pad_token_id, eos_token_id, unk_token_id = resolve_special_token_ids(config)
    print(f"Resolved special token IDs - pad: {pad_token_id}, eos: {eos_token_id}, unk: {unk_token_id}")

    calibration_data = "calibration" if calibration_data is None else calibration_data
    print(f"Using calibration data from: {calibration_data}")

    # this step preprocesses the data using the sentencepiece model and vocab, then saves the input IDs
    # and attention masks to disk for use during quantization of the encoder
    print('Generating encoder calibration data...')
    generate_encoder_calibration_data(
        spm_src, vocab, unk_token_id, pad_token_id, eos_token_id,
        calibration_cache, calibration_data, enc_len)

    # this runs the model end-to-end to capture encoder hidden states, and saves them to disk for use
    # during quantization of the decoder
    print('Generating decoder calibration data...')
    decoder_start_token_id = config.get("decoder_start_token_id", pad_token_id)

    generate_decoder_calibration_data(
        spm_src, vocab, unk_token_id, pad_token_id, eos_token_id, decoder_start_token_id,
        calibration_cache, calibration_data, enc_len, dec_len,
        rknn_enc, rknn_dec, lm_weight, lm_bias)

    # quantize the encoder using the collected calibration data
    print('Quantizing encoder...')
    rknn_enc_quant = quantize_encoder(f"{input_path}/encoder.onnx", calibration_cache, output_path, platform, dynamic_input, encoder_input_size_list)

    # quantize the decoder using the collected calibration data
    print('Quantizing decoder...')
    rknn_dec_quant = quantize_decoder(f"{input_path}/decoder.onnx", calibration_cache, output_path, platform, dynamic_input, decoder_input_size_list)

    try:
        if rknn_enc_quant.init_runtime(target=None) != 0:
            raise RuntimeError("Failed to initialize RKNN encoder runtime")

        if rknn_dec_quant.init_runtime(target=None) != 0:
            raise RuntimeError("Failed to initialize RKNN decoder runtime")

        inference(
            rknn_enc_quant,
            rknn_dec_quant,
            lm_weight,
            lm_bias,
            input_path,
            enc_len=enc_len,
            dec_len=dec_len,
        )

    finally:
        rknn_enc_quant.release()
        rknn_dec_quant.release()


if __name__ == '__main__':
    main()
