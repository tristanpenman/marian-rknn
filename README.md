# Marian RKNN

This repo contains an implementation of MarianMT that runs on Rockchip NPU (RKNN) devices. It also includes Python scripts and step-by-step instructions to assist with the model conversion process.

### Contents

* [Background](#background)
  * [MarianNMT](#mariannmt)
  * [MarianMT](#marianmt)
  * [Key Challenges](#key-challenges)
* [Hugging Face](#hugging-face)
  * [Preflight](#preflight)
  * [Prerequisites](#prerequisites)
  * [Docker (CPU-only preflight)](#docker-cpu-only-preflight)
  * [Docker Compose](#docker-compose)
* [Conversion](#conversion)
  * [Get Model Path](#get-model-path)
  * [Export to ONNX](#export-to-onnx)
  * [ONNX to RKNN](#onnx-to-rknn)
* [Inference](#inference)
  * [Dependencies](#dependencies)
  * [Model Output](#model-output)
  * [Show Time!](#show-time)
  * [Beam Search](#beam-search)
* [Benchmarking](#benchmarking)
* [License](#license)

## Background

Neural machine translation (NMT) systems translate text by learning sequence-to-sequence mappings between languages. Earlier models relied on recurrent neural networks, or even statistical machine translation (SMT). Modern architectures typically use transformers with attention mechanisms. Deploying these models on Edge AI devices such as Rockchip NPUs requires careful conversion to the neural network primitives supported by the device.

### MarianNMT

MarianNMT is a machine translation framework developed by the [University of Helsinki Language Technology Group](https://blogs.helsinki.fi/language-technology/). MarianNMT focuses on efficiency, with an implementation written in pure C++, with very few dependencies. The framework includes a custom auto-differentiation engine and efficient algorithms to train encoder-decoder models.

GPU support can be enabled if CUDA and cuDNN are available. However, this does not port easily to embedded NPUs, such as the Rockchip NPU.

### MarianMT

MarianMT is a PyTorch implementation and collection of pretrained models that have been trained on a large number of datasets and language pairs. Pretrained models are available on [Hugging Face](https://huggingface.co/Helsinki-NLP/models). This includes encoder-decoder checkpoints and tokenizers.

Being a PyTorch implementation is valuable, because we can convert that to ONNX, then to RKNN format.

### Key Challenges

Adapting MarianMT models for Rockchip NPUs involves several challenges. Rockchip's RKNN toolchain expects static computation graphs, so dynamic control flow and variable sequence lengths must be converted during ONNX export.

The NPU also has a limited set of supported operators, meaning unsupported layers need to be reimplemented or approximated with the primitives that are available. Finally, RKNN memory and quantization constraints require calibration and profiling to preserve accuracy once deployed on the target device.

## Hugging Face

Hugging Face hosts the official MarianMT checkpoints, tokenizers, and configuration files that seed our RKNN conversion workflow. We can use the `transformers` library to simplify downloading these artifacts, ensuring that the encoder, decoder, and vocabulary files remain synchronized across languages.

### Preflight

The `preflight.py` script can be used to check that your system can download and run a pretrained model from Hugging Face. You can choose a device (e.g. CUDA) using the `--device <type>` argument, and a specific model using `--model-name <id>`.

For example, to download an _OPUS MT English-to-French model_ and run it on a CUDA device:

```bash
python scripts/preflight.py --device cuda --model-name Helsinki-NLP/opus-mt-en-fr
```

After downloading the model, this will drop you in a prompt where you can enter English text, which will then be translated to French:

```
Using device: cuda
Using model: Helsinki-NLP/opus-mt-en-fr
Enter text to translate (empty line to quit):
> I am a fish
Je suis un poisson
>
```

### Prerequisites

Dependencies can be installed using `pip`:

```bash
pip install -r requirements.txt
```

This includes dependencies for scripts in submodules too.

### Docker (CPU-only preflight)

A lightweight Docker image has been provided for testing MarianMT on systems _without_ a GPU. The container uses a CPU-only PyTorch build to maximise compatibility.

Build the image from the repository root:

```bash
docker build -f Dockerfile -t marian-rknn .
```

You can then run the preflight check inside the container:

```bash
docker run --rm -it marian-rknn -v "$PWD:/workspace" \
  python scripts/preflight.py --device cpu --model-name Helsinki-NLP/opus-mt-en-fr
```

### Docker Compose

For an even easier time, you can use Docker Compose:

```bash
docker compose run --build marian-rknn
```

When the command is omitted this will automatically run the preflight script, and drop you into the interactive translator.

The same Docker Compose command can be used to run arbitrary scripts inside the container, or even just to load a bash shell:

```bash
docker compose run --rm --build marian-rknn /bin/bash
```

> ![NOTE]
> All of the commands listed below can be run within this shell.

## Conversion

After validating a model locally using the `preflight.py` script, you're ready to export the weights to ONNX, feed them into the RKNN conversion pipeline, and package the resulting artifacts for deployment on Rockchip devices.

### Get Model Path

We'll need to find out where the `preflight.py` script downloaded the model:

```bash
python -c "from huggingface_hub import snapshot_download; print(snapshot_download('Helsinki-NLP/opus-mt-en-fr'))"
```

The output will look something like this:

```bash
/workspace/.cache/huggingface/hub/models--Helsinki-NLP--opus-mt-en-fr/snapshots/dd7f6540a7a48a7f4db59e5c0b9c42c8eea67f18
```

This is the local path to the model.

### Export to ONNX

The [Marian-ONNX-Converter](./Marian-ONNX-Converter) submodule contains an ONNX implementation of Marian. This includes a script for converting pretrained models from Hugging Face to ONNX format.

If you haven't already, initialise the submodule:

```bash
git submodule update --init Marian-ONNX-Converter
```

You should be able to run the `convert.py` script without installing any additional dependencies:

```bash
python Marian-ONNX-Converter/convert.py
```

This will prompt you to provide an input file, an optionally a path for an output:

```bash
usage: convert.py [-h] [-o OUTPUT] [--no-quantize] input
convert.py: error: the following arguments are required: input
```

Use the model path we found earlier:

```bash
python Marian-ONNX-Converter/convert.py \
  /workspace/.cache/huggingface/hub/models--Helsinki-NLP--opus-mt-en-fr/snapshots/dd7f6540a7a48a7f4db59e5c0b9c42c8eea67f18 \
  --no-quantize
```

It's very important to specify `--no-quantize`. Failure to do so will produce a model graph that contains layers/operations that are not supported by RKNN. Quantization will be handled later, when converting to RKNN.

The output of `convert.py` will look like this:

```
Exporting encoder to ONNX...
Exporting decoder to ONNX...
Verifying export...
Model outputs from torch and ONNX Runtime are similar.
Success.
Creating archive file...
Done.
```

The ONNX-format encoder and decoder will be written to `outs/<model-name>`:

```
$ ls -l
total 227860
-rw-r--r-- 1 root root      1416 Oct 16 12:10 config.json
-rw-r--r-- 1 root root  56780822 Oct 16 12:10 decoder.onnx        <--
-rw-r--r-- 1 root root  50146129 Oct 16 12:10 encoder.onnx        <--
-rw-r--r-- 1 root root    239196 Oct 16 12:10 lm_bias.bin
-rw-r--r-- 1 root root 121885926 Oct 16 12:10 lm_weight.bin
-rw-r--r-- 1 root root    778395 Oct 16 12:10 source.spm
-rw-r--r-- 1 root root    802397 Oct 16 12:10 target.spm
-rw-r--r-- 1 root root        42 Oct 16 12:10 tokenizer_config.json
-rw-r--r-- 1 root root   1339166 Oct 16 12:10 vocab.json
```

We will later convert `decoder.onnx`, `encoder.onnx`, `lm_bias.bin` and `lm_weight.bin` into formats that can be used by our C++ RKNN implementation.

### Verify ONNX

Before proceeding any further, we should verify that the ONNX models work correctly, using the `test.py` script:

```bash
python Marian-ONNX-Converter/test.py \
  outs/dd7f6540a7a48a7f4db59e5c0b9c42c8eea67f18
```

You should be able to translate from English to French with ease:

```
Enter text to translate (empty line to quit):
> I am a fish
Je suis un poisson
```

### ONNX to RKNN

Now we can convert the encoder and decoder from ONNX to RKNN. We can use our own [rknn_convert.py](scripts/rknn_convert.py) script for this, passing in the same model output path from earlier:

```bash
python scripts/rknn_convert.py \
  outs/dd7f6540a7a48a7f4db59e5c0b9c42c8eea67f18 rk3588
```

This will look for `encoder.onnx` and `decoder.onnx` in the specified directory, and convert them to `encoder.rknn` and `decoder.rknn` respectively.

Don't be alarmed that the output is a little noisy. The conversion process should look something like this:

```
Converting encoder...
I rknn-toolkit2 version: 2.3.2
--> Config model
done
--> Loading model
W load_onnx: If you don't need to crop the model, don't set 'inputs'/'input_size_list'/'outputs'!
I Loading : 100%|██████████████████████████████████████████████████| 98/98 [00:00<00:00, 788.63it/s]
done
--> Building model
W build: For tensor ['/encoder/Constant_14_output_0'], the value smaller than -3e+38 or greater than 3e+38 has been corrected to -10000 or 10000. Set opt_level to 2 or lower to disable this correction.
I OpFusing 0 : 100%|██████████████████████████████████████████████| 100/100 [00:00<00:00, 386.07it/s]
I OpFusing 1 : 100%|██████████████████████████████████████████████| 100/100 [00:00<00:00, 157.24it/s]
I OpFusing 0 : 100%|██████████████████████████████████████████████| 100/100 [00:01<00:00, 77.62it/s]
I OpFusing 1 : 100%|██████████████████████████████████████████████| 100/100 [00:01<00:00, 75.28it/s]
I OpFusing 0 : 100%|██████████████████████████████████████████████| 100/100 [00:01<00:00, 73.08it/s]
I OpFusing 1 : 100%|██████████████████████████████████████████████| 100/100 [00:01<00:00, 72.35it/s]
I OpFusing 2 : 100%|██████████████████████████████████████████████| 100/100 [00:01<00:00, 70.86it/s]
I OpFusing 0 : 100%|██████████████████████████████████████████████| 100/100 [00:01<00:00, 66.73it/s]
I OpFusing 1 : 100%|██████████████████████████████████████████████| 100/100 [00:01<00:00, 65.93it/s]
I OpFusing 2 : 100%|██████████████████████████████████████████████| 100/100 [00:01<00:00, 59.69it/s]
I rknn building ...
E RKNN: [08:30:31.518] Unkown op target: 0
E RKNN: [08:30:31.518] Unkown op target: 0
I rknn building done.
done
--> Export rknn model
done
Converting decoder...
I rknn-toolkit2 version: 2.3.2
--> Config model
done
--> Loading model
W load_onnx: If you don't need to crop the model, don't set 'inputs'/'input_size_list'/'outputs'!
I Loading : 100%|███████████████████████████████████████████████| 158/158 [00:00<00:00, 1221.60it/s]
W load_onnx: The config.mean_values is None, zeros will be set for input 2!
W load_onnx: The config.std_values is None, ones will be set for input 2!
done
--> Building model
W build: For tensor ['/decoder/Constant_36_output_0'], the value smaller than -3e+38 or greater than 3e+38 has been corrected to -10000 or 10000. Set opt_level to 2 or lower to disable this correction.
W build: For tensor ['/decoder/Expand_output_0'], the value smaller than -3e+38 or greater than 3e+38 has been corrected to -10000 or 10000. Set opt_level to 2 or lower to disable this correction.
W build: For tensor ['/decoder/Expand_output_0_1'], the value smaller than -3e+38 or greater than 3e+38 has been corrected to -10000 or 10000. Set opt_level to 2 or lower to disable this correction.
W build: For tensor ['/decoder/Expand_output_0_2'], the value smaller than -3e+38 or greater than 3e+38 has been corrected to -10000 or 10000. Set opt_level to 2 or lower to disable this correction.
W build: For tensor ['/decoder/Expand_output_0_3'], the value smaller than -3e+38 or greater than 3e+38 has been corrected to -10000 or 10000. Set opt_level to 2 or lower to disable this correction.
W build: For tensor ['/decoder/Expand_output_0_4'], the value smaller than -3e+38 or greater than 3e+38 has been corrected to -10000 or 10000. Set opt_level to 2 or lower to disable this correction.
W build: For tensor ['/decoder/Expand_output_0_5'], the value smaller than -3e+38 or greater than 3e+38 has been corrected to -10000 or 10000. Set opt_level to 2 or lower to disable this correction.
I OpFusing 0 : 100%|██████████████████████████████████████████████| 100/100 [00:00<00:00, 286.03it/s]
I OpFusing 1 : 100%|██████████████████████████████████████████████| 100/100 [00:01<00:00, 92.72it/s]
I OpFusing 0 : 100%|██████████████████████████████████████████████| 100/100 [00:02<00:00, 42.73it/s]
I OpFusing 1 : 100%|██████████████████████████████████████████████| 100/100 [00:02<00:00, 41.58it/s]
I OpFusing 0 : 100%|██████████████████████████████████████████████| 100/100 [00:02<00:00, 40.47it/s]
I OpFusing 1 : 100%|██████████████████████████████████████████████| 100/100 [00:02<00:00, 40.13it/s]
I OpFusing 2 : 100%|██████████████████████████████████████████████| 100/100 [00:02<00:00, 39.36it/s]
I OpFusing 0 : 100%|██████████████████████████████████████████████| 100/100 [00:02<00:00, 37.25it/s]
I OpFusing 1 : 100%|██████████████████████████████████████████████| 100/100 [00:02<00:00, 36.86it/s]
I OpFusing 2 : 100%|██████████████████████████████████████████████| 100/100 [00:02<00:00, 33.90it/s]
I rknn building ...
E RKNN: [08:30:41.713] Unkown op target: 0
E RKNN: [08:30:41.713] Unkown op target: 0
I rknn building done.
done
--> Export rknn model
done
Converting LM weights...
Converting LM biases...
```

Once conversion is complete, a simulator will be started. Look for these lines specifically:

```
I Target is None, use simulator!
I Target is None, use simulator!
```

This will also drop you into a prompt where you can input individual strings to be translated:

```
Enter text to translate (empty line to quit):
> I am a fish
W inference: The 'data_format' is not set, and its default value is 'nhwc'!
I GraphPreparing : 100%|████████████████████████████████████████| 145/145 [00:00<00:00, 6668.06it/s]
I SessionPreparing : 100%|███████████████████████████████████████| 145/145 [00:00<00:00, 812.20it/s]
W inference: The 'data_format' is not set, and its default value is 'nhwc'!
I GraphPreparing : 100%|████████████████████████████████████████| 237/237 [00:00<00:00, 9004.81it/s]
I SessionPreparing : 100%|███████████████████████████████████████| 237/237 [00:00<00:00, 990.41it/s]
W inference: The 'data_format' is not set, and its default value is 'nhwc'!
W inference: The 'data_format' is not set, and its default value is 'nhwc'!
Je suis un poisson
>
```

This is enough to confirm that the model is working.

## Inference

To run the model on your Rockchip device, you will need to install some dependencies and copy across the converted model files.

### Dependencies

It is recommended that you install Python dependencies in Python virtual environment on the Rockchip device. Start by creating the environment:

```bash
python3 -m venv venv
```

Be sure to also _activate_ it:

```bash
source venv/bin/activate
```

Now you can install other packages using `pip`:

```bash
pip install -r requirements.lite.txt
```

The most important dependency here is [RKNN Toolkit Lite](https://github.com/rockchip-linux/rknn-toolkit/tree/master/rknn-toolkit-lite) - a trimmed down version of the RKNN Toolkit with individual device / NPU support added.

### Model Outputs

You will also need to copy the conversion output from earlier onto your device:

```bash
scp -r outs <edge2-ip>:~
```

### Show Time!

You can now run the inference script on the Rockchip device:

```bash
python scripts/rknn_infer.py
```

When no arguments are provided, this script will simply print out usage information:

```
usage: rknn_infer.py [-h] [--beam-search] [--beam-depth BEAM_DEPTH] [--beam-width BEAM_WIDTH] model_path [inputs ...]
rknn_infer.py: error: the following arguments are required: model_path, inputs
```

We'll just use the model output files that we copied above:

```bash
python scripts/rknn_infer.py ~/outs/dd7f6540a7a48a7f4db59e5c0b9c42c8eea67f18
```

While loading, the output should look something like this:

```log
W rknn-toolkit-lite2 version: 2.3.2
I RKNN: [08:58:26.109] RKNN Runtime Information, librknnrt version: 2.3.2 (429f97ae6b@2025-04-09T09:09:27)
I RKNN: [08:58:26.109] RKNN Driver Information, version: 0.9.8
I RKNN: [08:58:26.109] RKNN Model Information, version: 6, toolkit version: 2.3.2(compiler version: 2.3.2 (e045de294f@2025-04-07T19:48:25)), target: RKNPU v2, target platform: rk3588, framework name: ONNX, framework layout: NCHW, model inference type: static_shape
W RKNN: [08:58:26.240] query RKNN_QUERY_INPUT_DYNAMIC_RANGE error, rknn model is static shape type, please export rknn with dynamic_shapes
W Query dynamic range failed. Ret code: RKNN_ERR_MODEL_INVALID. (If it is a static shape RKNN model, please ignore the above warning message.)
W rknn-toolkit-lite2 version: 2.3.2
I RKNN: [08:58:26.377] RKNN Runtime Information, librknnrt version: 2.3.2 (429f97ae6b@2025-04-09T09:09:27)
I RKNN: [08:58:26.377] RKNN Driver Information, version: 0.9.8
I RKNN: [08:58:26.378] RKNN Model Information, version: 6, toolkit version: 2.3.2(compiler version: 2.3.2 (e045de294f@2025-04-07T19:48:25)), target: RKNPU v2, target platform: rk3588, framework name: ONNX, framework layout: NCHW, model inference type: static_shape
W RKNN: [08:58:26.527] query RKNN_QUERY_INPUT_DYNAMIC_RANGE error, rknn model is static shape type, please export rknn with dynamic_shapes
W Query dynamic range failed. Ret code: RKNN_ERR_MODEL_INVALID. (If it is a static shape RKNN model, please ignore the above warning message.)
Enter text to translate (empty line to quit):
>
```

Don't worry about the warnings. The most important thing is that the final prompt is visible, and that translations behave as expected:

```
Enter text to translate (empty line to quit):
> I am a fish
Je suis un poisson
>
```

### Beam Search

The last thing worth mentioning is support for [Beam Search](https://en.wikipedia.org/wiki/Beam_search). The default behaviour of the inference script is to use Greedy Decoding, which simply consumes tokens as they are generated. Beam Search is an alternative that allows multiple paths to be explored iteratively. Although it is a little slower, it can be lead to higher quality outputs.

This can enabled using the `--beam-search` option. The beam depth and beam width can also be configured using command line arguments:

```bash
python scripts/rknn_infer.py --beam-search --beam-width 3 \
  outs/dd7f6540a7a48a7f4db59e5c0b9c42c8eea67f18
```

### Benchmarking

To run the Python benchmark wrapper, provide a model path, a text file with input lines, and a maximum runtime (in seconds). The benchmark will loop over the inputs until the time budget is reached and will report aggregate throughput and per-stage timings:

```bash
python scripts/benchmark.py \
  outs/dd7f6540a7a48a7f4db59e5c0b9c42c8eea67f18 inputs.txt 30
```

Beam search options are supported for benchmarking as well:

```bash
python scripts/benchmark.py --beam-search --beam-width 3 \
  outs/dd7f6540a7a48a7f4db59e5c0b9c42c8eea67f18 inputs.txt 30
```

## License

This code is released under the Apache License 2.0 license. See the [LICENSE](./LICENSE) file for more information.
