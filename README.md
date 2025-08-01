# Marian RKNN

Python scripts and other resources to run MarianMT models on Rockchip NPU devices.

### Contents

* [Background](#background)
  * [MarianMT](#marianmt)
  * [Key Challenges](#key-challenges)
* [Prerequisites](#prerequisites)
  * [Preflight](#preflight)
* [Hugging Face](#hugging-face)
* [ONNX Conversion](#onnx-conversion)

## Background

TODO: Brief discussion of NMT models.

### MarianMT

TODO: What is MarianMT

### Key Challenges

TODO: What are the key challenges in converting this to RKNN format.

## Prerequisites

Dependencies can be installed using `pip`:

```
pip install -r requirements.txt
```

This includes dependencies for scripts in submodules too.

### Preflight

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

## Hugging Face

TODO: Discuss using Hugging Face as a source of pretrained models.

## ONNX Converter

In the [Marian-ONNX-Converter](./Marian-ONNX-Converter) submodule you will find an ONNX implementation of Marian. This includes a conversion script that allows pretrained models from Hugging Face to be converted to ONNX format.
