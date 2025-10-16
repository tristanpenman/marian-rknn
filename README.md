# Marian RKNN

Python scripts and other resources to run MarianMT models on Rockchip NPU devices.

### Contents

* [Background](#background)
  * [MarianMT](#marianmt)
  * [Key Challenges](#key-challenges)
* [Preflight](#preflight)
  * [Prerequisites](#prerequisites)
  * [Docker (CPU-only preflight)](#docker-cpu-only-preflight)
  * [Docker Compose](#docker-compose)
* [Hugging Face](#hugging-face)
  * [Available Models](#available-models)
* [Conversion](#conversion)
  * [Get Model Path](#get-model-path)
  * [Export to ONNX](#export-to-onnx)
  * [ONNX to RKNN](#onnx-to-rknn)

## Background

Neural machine translation (NMT) systems translate text by learning sequence-to-sequence mappings between languages. Earlier models relied on recurrent neural networks, or even statistical machine translation (SMT). Modern architectures typically use transformers with attention mechanisms. Deploying these models on Edge AI devices such as Rockchip NPUs requires careful conversion to the neural network primitives supported by the device.

### MarianMT

MarianMT is a machine translation framework developed by the [University of Helsinki Language Technology Group](https://blogs.helsinki.fi/language-technology/), more commonly known as [Helsinki-NLP](https://huggingface.co/Helsinki-NLP). MarianMT provides Transformer-based models that have been trained on a large number of datasets and language pairs. Pre-trained models are available on [Hugging Face](https://huggingface.co/Helsinki-NLP/models), with both encoder-decoder checkpoints and tokenization assets.

MarianMT focuses on efficiency, with an implementation written in pure C++, with very few dependencies. The framework includes a custom auto-differentiation engine and efficient algorithms to train encoder-decoder models.

### Key Challenges

Adapting MarianMT models for Rockchip NPUs involves several challenges. Rockchip's RKNN toolchain expects static computation graphs, so dynamic control flow and variable sequence lengths must be converted during ONNX export.

The NPU also has a limited set of supported operators, meaning unsupported layers need to be reimplemented or approximated with the primitives that are available. Finally, RKNN memory and quantization constraints require calibration and profiling to preserve accuracy once deployed on the target device.

## Preflight

The script `preflight.py` can be used to check that your system can run a pretrained model from Hugging Face. You can choose a device (e.g. CUDA) using the `--device <type>` argument, and a specific model using `--model-name <id>`.

For example, to download an English-to-French model and run on a CUDA device:

```
$ python3 preflight.py --device cuda --model-name Helsinki-NLP/opus-mt-en-fr
Using device: cuda
Using model: Helsinki-NLP/opus-mt-en-fr
Enter text to translate (empty line to quit):
> I am a fish
Je suis un poisson
>
```

### Prerequisites

Dependencies can be installed using `pip`:

```
pip install -r requirements.txt
```

This includes dependencies for scripts in submodules too.

### Docker (CPU-only preflight)

A lightweight Docker image is provided for testing MarianMT on systems without a GPU. The container uses a CPU-only PyTorch build to maximise compatibility.

Build the image from the repository root:

```bash
docker build -f docker/Dockerfile -t marian-rknn-preflight .
```

You can then run the preflight check inside the container:

```bash
docker run --rm -it marian-rknn-preflight python preflight.py --device cpu --model-name Helsinki-NLP/opus-mt-en-fr
```

The command above drops you into the interactive translator. To persist the Hugging Face cache between runs, mount a volume:

```bash
docker run --rm -it -v "$PWD/.cache:/app/.cache/huggingface" marian-rknn-preflight
```

### Docker Compose

For an even easier time, you can use Docker Compose:

```bash
docker compose run --build marian-rknn
```

This will automatically run the preflight script, and drop you into the interactive transalator.

The same Docker Compose command can be used to run arbitrary scripts inside the container, or even just to load a bash shell:

```bash
docker compose run --build marian-rknn /bin/bash
```

## Hugging Face

Hugging Face hosts the official MarianMT checkpoints, tokenizers, and configuration files that seed the RKNN conversion workflow. We can use the `transformers` library to simplify downloading these artifacts, ensuring that the encoder, decoder, and vocabulary files remain synchronized across languages.

## Conversion

After validating a model locally using the `preflight.py` script, you can export the weights to ONNX, feed them into the RKNN conversion pipeline, and package the resulting artifacts for deployment on Rockchip devices.

### Get Model Path

```
python -c "from huggingface_hub import snapshot_download; print(snapshot_download('Helsinki-NLP/opus-mt-en-fr'))"
```

The output will look something like this:

```
/workspace/.cache/huggingface/hub/models--Helsinki-NLP--opus-mt-en-fr/snapshots/dd7f6540a7a48a7f4db59e5c0b9c42c8eea67f18
```

This is the local path to the model.

### Export to ONNX

In the [Marian-ONNX-Converter](./Marian-ONNX-Converter) submodule you will find an ONNX implementation of Marian. This includes a conversion script that allows pretrained models from Hugging Face to be converted to ONNX format.

```bash
python Marian-ONNX-Converter/convert.py
```

This will prompt you to provide an input file, an optionally a path for an output:

```bash
$ python Marian-ONNX-Converter/convert.py
usage: convert.py [-h] [-o OUTPUT] [--no-quantize] input
convert.py: error: the following arguments are required: input
```

Use the model path we found earlier:

```bash
python Marian-ONNX-Converter/convert.py /workspace/.cache/huggingface/hub/models--Helsinki-NLP--opus-mt-en-fr/snapshots/dd7f6540a7a48a7f4db59e5c0b9c42c8eea67f18
```

While converting / exporting, the output will look like this:

```
Exporting encoder to ONNX...
Quantizing...
WARNING:root:Please consider to run pre-processing before quantization. Refer to example: https://github.com/microsoft/onnxruntime-inference-examples/blob/main/quantization/image_classification/cpu/ReadMe.md
Done
Exporting decoder to ONNX...
Quantizing...
WARNING:root:Please consider to run pre-processing before quantization. Refer to example: https://github.com/microsoft/onnxruntime-inference-examples/blob/main/quantization/image_classification/cpu/ReadMe.md
Done
Verifying export...

Not equal to tolerance rtol=0.001, atol=0.001

(shapes (1, 6), (1, 5) mismatch)
 x: array([[59513,  8703,    19,   507,   291,     0]])
 y: array([[59513,  8703,   507,   291,     0]])
Creating archive file...
Done.
```

The ONNX-format encoder/decoder will be written to `outs/<model-name>`:

```
$ ls -l
total 227860
-rw-r--r-- 1 root root      1416 Oct 16 12:10 config.json
-rw-r--r-- 1 root root  56780822 Oct 16 12:10 decoder.onnx
-rw-r--r-- 1 root root  50146129 Oct 16 12:10 encoder.onnx
-rw-r--r-- 1 root root    239196 Oct 16 12:10 lm_bias.bin
-rw-r--r-- 1 root root 121885926 Oct 16 12:10 lm_weight.bin
-rw-r--r-- 1 root root    778395 Oct 16 12:10 source.spm
-rw-r--r-- 1 root root    802397 Oct 16 12:10 target.spm
-rw-r--r-- 1 root root        42 Oct 16 12:10 tokenizer_config.json
-rw-r--r-- 1 root root   1339166 Oct 16 12:10 vocab.json
```
