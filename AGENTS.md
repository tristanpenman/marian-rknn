# Agents

## Project Context

This repo contains a starter CMake project for running MarianMT models on Rockchip devices via RKNN/RKLLM.

## Structure

The `cpp` directory contains C++ source code for a MarianMT demo.

The `thirdparty` directory contains RKNN libraries.

## Guidelines

Do not try to run CMake directly. Use the build steps documented below.

After making any C++ or build-system change, always check the Docker Compose builds before finalising the task. While iterating on changes, building for `native` is fine. To finalise changes, ensure both the native and Android builds have been attempted:

```bash
./scripts/build-native.sh docker
./scripts/build-android.sh docker
```

If a Docker Compose build cannot be run or fails for an environment reason, report that explicitly along with the command that was attempted.

## Build Android

To build the CLI for Android:

```bash
./scripts/build-android.sh docker
```

## Build Native

To build for native Linux:

```bash
./scripts/build-native.sh docker
```

## Caveats

The build is intended to be run via Docker. The target is an RK3588 device, so the output of these builds cannot be run locally.

## Markdown

Markdown documents should use soft-wrapping. Tables should be aligned to be human readable.
