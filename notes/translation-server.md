# Translation Server

Goal is to build a "production-ready" server variant of Marian RKNN that supports:

- single-request low-latency translation
- dynamic batching for throughput
- special handling for long sentences (chunking + scheduling)
- configurable optimisation strategies for different device constraints

This could then be wired up to a frontend (also hosted on the device) to create a fully self-contained translation hub.

## Prerequisites

- A server architecture doc with component boundaries
- An initial API contract (`/translate`, `/health`, `/metrics`)
- A latency/throughput target table for RK356x/RK3588 class devices

## Architecture

- New server binary target under `cpp/`
  - Assumption is that the C++ implementation will be used for performance
  - Prefer a process-local server (HTTP/gRPC) over shell
  - This needs to be validated

### Model preparation

Need to figure out how to handle model conversion. Should this be externalised?

Other concerns:
- Define sequence-length buckets for scheduler
  - e.g. token lengths: `1-32`, `33-64`, `65-128`, `129-256`, `257+`.
- Establish hard limits
  - Max input chars
  - Max source tokens
  - Timeouts
  - Concurrent requests

### Endpoints

Simple endpoints to begin:

- `POST /v1/translate`
- `GET /health` (or `healthz` to use the latest slang)
- `GET /metrics`

Additional notes:

- Metrics should be in Prometheus format if possible
- Structured logging and request IDs are a must

## Plan

1. Add a server module that owns:
   - tokenizer/model resources
   - worker threads
   - queue/scheduler
   - stats collector
2. Add startup checks
   - validate encoder/decoder/lm files
   - warm-up inference pass
3. Add baseline metrics
   - request count/status
   - queue depth
   - per-stage timings
     - tokenization
     - encoding
     - decoding
     - detokenization
   - batch size histogram
   - sentence-length histogram

## Future

- Can the service be instructed to download alternative models?
