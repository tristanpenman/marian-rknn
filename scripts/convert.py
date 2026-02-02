#!/usr/bin/env python3

import argparse
import torch
import numpy as np

from infer import inference, DEC_LEN, ENC_LEN, MODEL_DIM

from rknn.api import RKNN

DEFAULT_QUANT = False

DECODER_INPUTS = ['input_ids', 'attention_mask', 'encoder_hidden_states']
DECODER_INPUT_SIZE_LIST = [[1, DEC_LEN], [1, DEC_LEN], [1, DEC_LEN, MODEL_DIM]]

ENCODER_INPUTS = ['input_ids', 'attention_mask']
ENCODER_INPUT_SIZE_LIST = [[1, ENC_LEN], [1, ENC_LEN]]

def parse_arg():
    parser = argparse.ArgumentParser(
        description="Convert Marian ONNX models to RKNN.",
    )
    parser.add_argument("input_path", help="Path to the directory containing the ONNX files.")
    parser.add_argument("--dynamic-input", action="store_true", default=False, help="Export model using dynamic inputs (default=off).")
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

    do_quant = DEFAULT_QUANT
    output_path = args.output_path or args.input_path

    return args.input_path, args.platform, do_quant, output_path, args.dynamic_input

def convert_model(model_path, platform, do_quant, dynamic_input, output_path, inputs, input_size_list):
    rknn = RKNN(verbose=False)

    print('--> Config model')
    if dynamic_input:
        # Warning: This is disabled by default, as the C++ rknn_api fails with
        # std::out_of_range exceptions when loading models with `dynamic_input`
        rknn.config(target_platform=platform, dynamic_input=[input_size_list])
    else:
        rknn.config(target_platform=platform)

    print('done')

    # Load model
    print('--> Loading model')
    ret = rknn.load_onnx(model=model_path,
                         inputs=inputs,
                         input_size_list=input_size_list)
    if ret != 0:
        print('Load model failed!')
        exit(ret)
    print('done')

    # Build model
    print('--> Building model')
    ret = rknn.build(do_quantization=do_quant)
    if ret != 0:
        print('Build model failed!')
        exit(ret)
    print('done')

    # Export rknn model
    print('--> Export rknn model')
    ret = rknn.export_rknn(output_path)
    if ret != 0:
        print('Export rknn model failed!')
        exit(ret)
    print('done')

    return rknn


def convert_weights(input_path, output_path):
    tensor = torch.load(input_path, weights_only=True)
    weights = tensor.detach().numpy().astype(np.float32)
    weights.tofile(output_path)
    return weights


def main():
    input_path, platform, do_quant, output_path, dynamic_input = parse_arg()

    print('Converting encoder...')
    rknn_enc = convert_model(f"{input_path}/encoder.onnx", platform, do_quant, dynamic_input,
                             f"{output_path}/encoder.rknn", ENCODER_INPUTS, ENCODER_INPUT_SIZE_LIST)

    print('Converting decoder...')
    rknn_dec = convert_model(f"{input_path}/decoder.onnx", platform, do_quant, dynamic_input,
                             f"{output_path}/decoder.rknn", DECODER_INPUTS, DECODER_INPUT_SIZE_LIST)

    print('Converting LM weights...')
    lm_weight = convert_weights(f"{input_path}/lm_weight.bin",
                                f"{output_path}/lm_weight.raw")

    print('Converting LM biases...')
    lm_bias = convert_weights(f"{input_path}/lm_bias.bin",
                              f"{output_path}/lm_bias.raw")

    try:
        if rknn_enc.init_runtime(target=None) != 0:
            raise RuntimeError("Failed to initialize RKNN encoder runtime")

        if rknn_dec.init_runtime(target=None) != 0:
            raise RuntimeError("Failed to initialize RKNN decoder runtime")

        inference(rknn_enc, rknn_dec, lm_weight, lm_bias, input_path, ENC_LEN, DEC_LEN)

    finally:
        rknn_enc.release()
        rknn_dec.release()


if __name__ == '__main__':
    main()
