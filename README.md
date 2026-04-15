# Marian RKNN

This repo contains Python and C++ implementations of MarianMT that run on Rockchip NPU (RKNN) devices. It also includes Python code and step-by-step instructions to assist with the model conversion process.

This README is intended both as a tutorial and a usage guide for the code in this repo.

### Contents

* [Background](#background)
  * [MarianNMT](#mariannmt)
  * [MarianMT](#marianmt)
  * [Key Challenges](#key-challenges)
  * [Hugging Face](#hugging-face)
* [Prerequisites](#prerequisites)
  * [Docker Compose](#docker-compose)
  * [Preflight](#preflight)
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
* [Native Implementation](#native-implementation)
  * [Cross-Compilation](#cross-compilation)
  * [Release Builds](#release-builds)
  * [Benchmarking (Native)](#benchmarking-native)
* [Android 14 (Khadas Edge2 / RK3588S)](#android-14-khadas-edge2--rk3588s)
  * [NDK Build](#ndk-build)
  * [Push and Run](#push-and-run)
* [Evaluation](#evaluation)
  * [WMT Datasets](#wmt-datasets)
  * [Downloader](#downloader)
  * [BLEU and chrF](#bleu-and-chrf)
* [Future Work](#future-work)
  * [Quantization](#quantization)
  * [Parallelisation](#parallelisation)
  * [KV-caching](#kv-caching)
* [Contributing](#contributing)
* [License](#license)

## Background

Neural machine translation (NMT) systems translate text by learning sequence-to-sequence mappings between languages. Earlier models relied on recurrent neural networks, or even statistical machine translation (SMT). Modern architectures typically use transformers with attention mechanisms. Deploying these models on Edge AI devices such as Rockchip NPUs requires careful conversion to the neural network primitives supported by the device.

### MarianNMT

MarianNMT is a machine translation framework developed by the [University of Helsinki Language Technology Group](https://blogs.helsinki.fi/language-technology/). MarianNMT focuses on efficiency, with an implementation written in pure C++, with very few dependencies. The framework includes a custom auto-differentiation engine and efficient algorithms to train encoder-decoder models.

GPU support can be enabled if CUDA and cuDNN are available. However, this does not port easily to embedded NPUs, such as the Rockchip NPU.

### MarianMT

MarianMT is a PyTorch implementation and collection of pretrained models that have been trained on a large number of datasets and language pairs. Pretrained models are available on [Hugging Face](https://huggingface.co/Helsinki-NLP/models). This includes encoder-decoder checkpoints and tokenizers.

Being a PyTorch implementation is valuable because we can convert that to ONNX, then to RKNN format.

### Key Challenges

Adapting MarianMT models for Rockchip NPUs involves several challenges. The Rockchip RKNN API has limited support for dynamic graph operations. Inputs that use variable sequence lengths must be 'unrolled' during ONNX export.

The NPU also has a limited set of supported operators, meaning unsupported layers need to be reimplemented or approximated with the primitives that are available. Finally, RKNN memory and quantization constraints require calibration and profiling to preserve accuracy once deployed on the target device.

### Hugging Face

Hugging Face hosts the official MarianMT checkpoints, tokenizers, and configuration files that seed our RKNN conversion workflow. We can use the `transformers` library to simplify downloading these artifacts, ensuring that the encoder, decoder, and vocabulary files remain synchronized across languages.

## Prerequisites

A lightweight Docker image has been provided for running the Python code provided by this repo.

Build the image from the repository root:

```bash
docker build -f Dockerfile.python -t marian-rknn-python .
```

You can then run a shell inside the container that has all the dependencies installed:

```bash
docker run --rm -it marian-rknn-python -v "$PWD:/workspace" /bin/bash
```

If you do not provide a command to run (e.g. `/bin/bash`) the container will run the preflight script, as described below. This will drop you into an interactive translation prompt.

### Docker Compose

For an even easier time, it is recommended that you use Docker Compose:

```bash
docker compose run --build --rm python
```

The same Docker Compose command can be used to run arbitrary commands inside the container:

```bash
docker compose run --build --rm python <cmd> <args...>
```

> ![NOTE]
> All the commands listed below can be run within this shell.
> 
> The remainder of this guide assumes that you are running commands from inside the Docker container.

### Preflight

Once you have the Docker container running, you can use the [preflight.py](python/marian_rknn/preflight.py) script to download and run a pretrained model from Hugging Face.

You can choose a specific model using `--model-name <id>`. For example, to download the [OPUS English-to-French model](https://huggingface.co/Helsinki-NLP/opus-mt-en-fr):

```bash
python -m marian_rknn.preflight --model-name Helsinki-NLP/opus-mt-en-fr
```

After downloading the model, this will drop you in a prompt where you can enter English text to be translated to French:

```
Using device: cpu
Using model: Helsinki-NLP/opus-mt-en-fr
Enter text to translate (empty line to quit):
> I am a fish
Je suis un poisson
>
```

## Conversion

After downloading the model via the preflight script, you're ready to export the weights to ONNX, feed them into the RKNN conversion pipeline, and package the resulting artifacts for deployment on Rockchip devices.

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

> [!WARNING]
> 

### Export to ONNX

The [Marian-ONNX-Converter](./Marian-ONNX-Converter) submodule contains an ONNX implementation of Marian. This includes a script for converting pretrained models from Hugging Face to ONNX format.

If you haven't already, fetch the submodule:

```bash
git submodule update --init Marian-ONNX-Converter
```

You should be able to run the `convert.py` script without installing any additional dependencies:

```bash
python Marian-ONNX-Converter/convert.py
```

This will prompt you to provide an input file, and optionally a path for an output:

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

It's crucial to specify `--no-quantize`. Failure to do so will produce a model graph that contains layers/operations that are not supported by RKNN. Quantization will be handled later when converting to RKNN.

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

We will later convert `decoder.onnx`, `encoder.onnx`, `lm_bias.bin` and `lm_weight.bin` into formats that can be used by our Python and C++ RKNN inference implementations.

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

Now we can convert the encoder and decoder from ONNX to RKNN using the [rknn_convert.py](python/marian_rknn/rknn_convert.py) script. Pass in the same model output path from earlier:

```bash
python -m marian_rknn.rknn_convert \
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

To run the model on your Rockchip device, you will need to install Python dependencies and copy across the converted model files.

### Dependencies

It is recommended that you install Python dependencies in a Python virtual environment on the Rockchip device. Start by creating the environment:

```bash
python3 -m venv venv
```

Be sure to also _activate_ it:

```bash
source venv/bin/activate
```

Now you can install other packages using `pip`:

```bash
cd python
pip install -r python/requirements.lite.txt
```

The most important dependency here is [RKNN Toolkit Lite](https://github.com/rockchip-linux/rknn-toolkit/tree/master/rknn-toolkit-lite) - a trimmed-down version of the RKNN Toolkit with individual device / NPU support added.

### Model Output

You will also need to copy the conversion output from earlier onto your device:

```bash
scp -r outs <edge2-ip>:~
```

### Show Time!

You can now run the inference script on the Rockchip device:

```bash
python -m marian_rknn.rknn_infer
```

When no arguments are provided, this script will simply print out usage information. You can use `-h` to get more detailed information:

```
usage: rknn_infer.py [-h] [--beam-search] [--beam-depth BEAM_DEPTH] [--beam-width BEAM_WIDTH] [--enc-len ENC_LEN] [--dec-len DEC_LEN] model_path [inputs ...]

Run RKNN Marian translation.

positional arguments:
  model_path            Path to the directory containing the model files.
  inputs                Optional text strings to translate (quote to preserve spaces).

options:
  -h, --help            show this help message and exit
  --beam-search         Use beam search decoding instead of greedy decoding.
  --beam-depth BEAM_DEPTH
                        Maximum decoding depth for beam search.
  --beam-width BEAM_WIDTH
                        Beam width for beam search decoding.
  --enc-len ENC_LEN     Encoder sequence length (default: 32).
  --dec-len DEC_LEN     Decoder sequence length (default: 32).
```

We'll just use the model output files that we copied above:

```bash
python -m marian_rknn.rknn_infer ~/outs/dd7f6540a7a48a7f4db59e5c0b9c42c8eea67f18
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

The last thing worth mentioning is support for [Beam Search](https://en.wikipedia.org/wiki/Beam_search). The default behaviour of the inference script is to use Greedy Decoding, which simply consumes tokens as they are generated. Beam Search is an alternative that allows multiple paths to be explored iteratively. Although it is a little slower, it can lead to higher quality outputs.

This can be enabled using the `--beam-search` option. The beam depth and beam width can also be configured using command line arguments:

```bash
python -m marian_rknn.rknn_infer --beam-search --beam-width 3 \
  ../outs/dd7f6540a7a48a7f4db59e5c0b9c42c8eea67f18
```

### Benchmarking

To run the Python benchmark wrapper, provide a model path, a text file containing a range of input sentences, and a maximum runtime (in seconds). The benchmark will loop over the inputs until the time budget is reached and will report aggregate throughput and per-stage timings:

```bash
python -m marian_rknn.benchmark \
  outs/dd7f6540a7a48a7f4db59e5c0b9c42c8eea67f18 \
  datasets/en-phrases.txt \
  120
```

Sample output from a single run:

```
Benchmark complete
Elapsed: 121.442 s
Sentences: 207
Sentences/sec: 1.705
Total time: 121295.378 ms
Encoder time: 2492.285 ms
Decoder time: 91355.883 ms
LM head time: 26394.237 ms
Avg total time per sentence: 585.968 ms
Avg encoder time per sentence: 12.040 ms
Avg decoder time per sentence: 441.333 ms
Avg LM head time per sentence: 127.508 ms
Input tokens: 2379
Output tokens: 2766
Decoder iterations: 2766
Input tokens/sec: 19.590
Output tokens/sec: 22.776
```

Beam search options are supported for benchmarking as well:

```bash
python -m marian_rknn.benchmark \
  --beam-search \
  --beam-width 3 \
  outs/dd7f6540a7a48a7f4db59e5c0b9c42c8eea67f18 \
  datasets/en-phrases.txt \
  120
```

The results with beam search enabled show how much of an impact it can have...

```
Benchmark complete
Elapsed: 121.982 s
Sentences: 52
Sentences/sec: 0.426
Total time: 121851.846 ms
Encoder time: 631.315 ms
Decoder time: 72169.512 ms
LM head time: 20800.915 ms
Avg total time per sentence: 2343.305 ms
Avg encoder time per sentence: 12.141 ms
Avg decoder time per sentence: 1387.875 ms
Avg LM head time per sentence: 400.018 ms
Input tokens: 663
Output tokens: 729
Decoder iterations: 2177
Input tokens/sec: 5.435
Output tokens/sec: 5.976
```

Average decoder time per sentence goes up considerably, and the number of output tokens produced drops from 2766 to 729. Overall, we processed 52 sentences @ 0.426 sentences per second.

## Native Implementation

A native RKNN implementation of Marian MT can be found in the [cpp](./cpp) directory. This can be cross-compiled for the Rockchip platform using a Docker container. This build relies on a CMake [Out-of-Source Build](https://cmake.org/cmake/help/book/mastering-cmake/chapter/Getting%20Started.html#directory-structure).

### Cross-Compilation

Build and run the container using Docker Compose:

```bash
docker compose run --build native
```

This will drop you into a bash shell. Now you can compile the project using CMake:

```bash
mkdir -p build
cd build
cmake ..
make
```

The `marian-rknn` executable can then be copied over to the target device, using `scp` or another file transfer utility. For example:

```bash
scp marian-rknn <edge2-ip>:~
```

### Release Builds

By default, CMake will produce `Debug` builds, which include debugging symbols and have optimisations disabled.

To produce a `Release` build with `-O3` optimisations enabled:

```bash
cmake -DCMAKE_BUILD_TYPE=Release ..
```

Alternatively, you can produce a `RelWithDebInfo` build. This will have `-O2` optimisations enabled while also including debug symbols:

```bash
cmake -DCMAKE_BUILD_TYPE=RelWithDebInfo ..
```

### Benchmarking (Native)

The benchmark binary is built alongside the main executable and can be copied in the same way:

```bash
scp marian-rknn-benchmark <edge2-ip>:~
```

You can run the benchmark by passing a model directory, an input text file, and a max runtime in seconds:

```bash
./marian-rknn-benchmark /path/to/model inputs.txt 30
```

## Android 14 (Khadas Edge2 / RK3588S)

The C++ runtime can also be built for Android, following the same pattern used in Rockchip's official Android demos in the [RKNN Model Zoo](https://github.com/airockchip/rknn_model_zoo) (notably `build-android.sh` and C API examples).

### NDK Build

Build and run the Android build container using Docker Compose:

```bash
docker compose run --build android
```

This includes the Android NDK and CMake. From inside the container you can build the project using a CMake wrapper script:

```bash
./scripts/android-build.sh Release
```

Or for a debug build:

```bash
./scripts/android-build.sh Debug
```

This build script configures CMake with:

- `-DANDROID_ABI=arm64-v8a`
- `-DANDROID_PLATFORM=android-34`
- `-DCMAKE_TOOLCHAIN_FILE=<ndk>/build/cmake/android.toolchain.cmake`
- `-DRKNN_RUNTIME_LIB=<...>/librknnrt.so`

### Push and Run

Artifacts are written to `build-android`. These can be pushed to a device using `adb`:

```bash
adb push build-android/marian-rknn /data/local/tmp
adb push thirdparty/rknpu2/lib-android/arm64-v8a/librknnrt.so /data/local/tmp
```

You'll also need to push model files:

```bash
adb push outs/dd7f6540a7a48a7f4db59e5c0b9c42c8eea67f18 /data/local/tmp
```

On the device, you will need to override `LD_LIBRARY_PATH`:

```bash
LD_LIBRARY_PATH=$LD_LIBRARY_PATH:`pwd` \
  ./marian-rknn dd7f6540a7a48a7f4db59e5c0b9c42c8eea67f18/
```

## Evaluation

### WMT Datasets

The evaluation script included in this repo focuses on WMT datasets. WMT refers to a collection of _shared task_ datasets that cover a range of different machine translation tasks and domains. These are released yearly as part of the [Workshop on Machine Translation](https://statmt.org/).

For instance, [WMT17](https://statmt.org/wmt17/) states:

> This year's conference will feature the following shared tasks:
>
> - a news translation task,
> - a biomedical translation task,
> - an automatic post-editing task,
> - a metrics task (assess MT quality given reference translation).
> - a quality estimation task (assess MT quality without access to any reference),
> - a multimodal translation task
> - a task dedicated to the training of neural MT systems
> - a task on bandit learning for MT

The data is drawn from various sources, such as [Common Crawl](https://commoncrawl.org/), and the [European Parliament Proceedings](https://www.statmt.org/europarl/) and [News Commentary Parallel Corpus](https://autonlp.ai/datasets/news-commentary-parallel-corpus) datasets.

The WMT datasets date back to 2006. However, it is common to use WMT16 or later for evaluation purposes.

### Downloader

[SacreBLEU](https://github.com/mjpost/sacrebleu) includes utility functions that can be used to download WMT test sets, and to preprocess and tokenize the data so that it is ready for testing.

As a convenience, the [downloader.py](python/marian_rknn/downloader.py) script has been provided as a wrapper around SacreBLEU. It can be used to download and prepare WMT datasets. This script works with arbitrary language pairs, not just English-to-French.

To inspect available test sets for `en-fr`:

```bash
python -m marian_rknn.downloader \
  --langpair en-fr \
  --list
```

The list is quite long. We can see at the top of the list, several `wmt` datasets:

```
Available test sets for en-fr:
- wmt15
- wmt14
- wmt14/full
- wmt13
- wmt12
...
```

Coverage for English-to-French was reduced after 2015, so we'll download the `wmt14` and `wmt15` datasets, and save them to `datasets/eval`:

```bash
python -m marian_rknn.downloader \
  --langpair en-fr \
  --test-sets wmt14,wmt15 \
  --output-dir datasets/eval
```

This creates one directory per test set containing:

- `source.en.txt`
- `reference1.fr.txt`
- `reference2.fr.txt` (and so on, if the set has multiple references)
- `...`

A manifest file is also generated at `datasets/eval/manifest.en-fr.tsv` to make scripting easy.

### BLEU and chrF

The [eval.py](python/marian_rknn/eval.py) script can be used to generate BLEU and chrF metrics for any of the downloaded datasets.

[BLEU](https://en.wikipedia.org/wiki/BLEU) (or _Bilingual Evaluation Understudy_) measures the similarity between text generated by a model, and a human-generated reference text by comparing word-level n-grams.

[chrF](https://aclanthology.org/W15-3049.pdf) (or _CHaRacter-level F-score_), measures the similarity between text generated by a model and a human-generated reference text using character-level n-grams rather than whole words. This makes it more sensitive to spelling, inflection, and morphology differences.

## Future Work

Any further work on this repo will likely revolve around optimisation. T

### Quantization

Quantization aims to optimise the model for the Rockchip NPU by using INT8 weights and activations, instead of the default floating point (FP16) output from the RKNN build process. This depends on a calibration dataset (built using a full version of the model) which is used to choose quantization coefficients.

An early attempt at quantization can be found in [python/marian_rknn/rknn_quantize.py](./python/marian_rknn/rknn_quantize.py). This works similar to the `rknn_convert.py` script:

```bash
python -m marian_rknn.rknn_quantize \
  --calibration-data datasets/en-phrases.txt \
  outs/dd7f6540a7a48a7f4db59e5c0b9c42c8eea67f18 rk3588
```

Quantization depends on a calibration dataset, such as the [en-phrases.txt](./datasets/en-phrases.txt) dataset included in this repo.

> [!WARNING]
> Quantization is currently incomplete. The calibration process runs successfully, but the translation output is quite broken.

### Parallelisation

Knowing that the RK3588 has a three-core NPU, I would like to explore methods for better utilising the resources on the NPU. This may include batch processing, or concurrent execution strategies when using Beam Search.

### KV-caching

My rudimentary understanding of KV-caching (or key-value caching) is that it is a Transformer-specific optimisation to avoid repeated calculations. This can work well for auto-regressive decoders, since they revisit the same inputs multiple times.

There may be an opportunity to use that here.

## Contributing

Contributions are welcome. I will make an effort to review any bona fide contributions.

You are also welcome to raise GitHub issues against this repo, however please note this is merely a hobby project. I cannot offer any guarantee that issues will be responded to in a timely fashion.

## License

This code is released under the Apache License 2.0 license. See the [LICENSE](./LICENSE) file for more information.
