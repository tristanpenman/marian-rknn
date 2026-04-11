#!/bin/bash

docker compose run --rm --remove-orphans --build build \
  /bin/bash -c 'cd cpp/build && cmake .. && make'
