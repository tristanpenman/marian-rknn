# Marian RKNN

Python scripts and other resources to run MarianMT models on Rockchip NPU devices.

### Contents

* [Background](#background)
  * [MarianMT](#marianmt)
  * [Key Challenges](#key-challenges)
* [Preflight](#preflight)
  * [Prerequisites](#prerequisites)
  * [Docker (CPU-only preflight)](#docker-cpu-only-preflight)
* [Hugging Face](#hugging-face)
* [ONNX Conversion](#onnx-conversion)

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

A lightweight Docker image is provided for testing MarianMT on systems without a GPU. The
container uses a CPU-only PyTorch build to maximise compatibility.

Build the image from the repository root:

```bash
docker build -f docker/Dockerfile -t marian-rknn-preflight .
```

Run the preflight check inside the container (defaults to CPU):

```bash
docker run --rm -it marian-rknn-preflight
```

The command above drops you into the interactive translator. To persist the Hugging Face cache between runs, mount a volume:

```bash
docker run --rm -it -v "$PWD/cache:/app/.cache/huggingface" marian-rknn-preflight
```

## Hugging Face

Hugging Face hosts the official MarianMT checkpoints, tokenizers, and configuration files that seed the RKNN conversion workflow. We can use the `transformers` library to simplify downloading these artifacts, ensuring that the encoder, decoder, and vocabulary files remain synchronized across languages.

After validating a model locally—such as through the `preflight.py` script, you can export the weights to ONNX, feed them into the RKNN conversion pipeline, and package the resulting artifacts for deployment on Rockchip devices.

## ONNX Converter

In the [Marian-ONNX-Converter](./Marian-ONNX-Converter) submodule you will find an ONNX implementation of Marian. This includes a conversion script that allows pretrained models from Hugging Face to be converted to ONNX format.
