#!/usr/bin/env python3

import sys

from rknn.api import RKNN

DEFAULT_QUANT = False

DECODER_INPUTS = ['input_ids', 'attention_mask', 'encoder_hidden_states']
DECODER_INPUT_SIZE_LIST = [[1, 32], [1, 32], [1, 32, 512]]

ENCODER_INPUTS = ['input_ids', 'attention_mask']
ENCODER_INPUT_SIZE_LIST = [[1, 32], [1, 32]]

ENABLE_DYNAMIC_INPUTS = False

def parse_arg():
    if len(sys.argv) < 3:
        print("Usage: python3 {} <input_path> [platform] [dtype(optional)] [output_path(optional)]".format(sys.argv[0]))
        print("       platform choose from [rk3562, rk3566, rk3568, rk3576, rk3588, rv1126b, rv1109, rv1126, rk1808]")
        print("       dtype choose from [fp] for [rk3562, rk3566, rk3568, rk3576, rk3588, rv1126b]")
        print("       dtype choose from [fp] for [rv1109, rv1126, rk1808]")
        exit(1)

    input_path = sys.argv[1]
    platform = sys.argv[2]

    do_quant = DEFAULT_QUANT
    if len(sys.argv) > 3:
        model_type = sys.argv[3]
        if model_type not in ['fp']:
            print("ERROR: Invalid model type: {}".format(model_type))
            exit(1)

    if len(sys.argv) > 4:
        output_path = sys.argv[4]
    else:
        output_path = input_path

    return input_path, platform, do_quant, output_path

def convert(model_path, platform, do_quant, output_path, inputs, input_size_list):
    # Create RKNN object
    rknn = RKNN(verbose=False)

    # Pre-process config
    print('--> Config model')
    if ENABLE_DYNAMIC_INPUTS:
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

    # Release
    rknn.release()

def main():
    input_path, platform, do_quant, output_path = parse_arg()

    print('Converting encoder!')
    convert(f"{input_path}/encoder.onnx", platform, do_quant,
            f"{output_path}/encoder.rknn", ENCODER_INPUTS, ENCODER_INPUT_SIZE_LIST)

    print('Converting decoder!')
    convert(f"{input_path}/decoder.onnx", platform, do_quant,
            f"{output_path}/decoder.rknn", DECODER_INPUTS, DECODER_INPUT_SIZE_LIST)

if __name__ == '__main__':
    main()
