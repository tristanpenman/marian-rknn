#!/bin/bash

docker compose run --rm --remove-orphans --build python \
  ./scripts/rknn_quantize.py --calibration-data datasets/en-phrases-short.txt outs/dd7f6540a7a48a7f4db59e5c0b9c42c8eea67f18 rk3588
