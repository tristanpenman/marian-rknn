# GGML Port

Porting the C++ RKNN implementation to GGML:

1. Keep the same high-level flow (SentencePiece → vocab IDs → encoder → autoregressive decoder → reverse vocab → SentencePiece decode).
2. Replace RKNN runtime and memory APIs with ggml tensor/graph execution.
3. Fold the LM head matmul+bias into ggml graph execution rather than post-processing with Eigen.

## RKNN to GGML

High level mapping of RKNN concepts to GGML...

### Context struct

The current context is RKNN-centric (`MODEL_INFO enc/dec`, token IDs, seq lengths, vocab maps, SentencePiece processors, LM head buffers).

A GGML context would keep tokenizer/vocab/config fields, but replace runtime objects with:
- GGML model tensors
- ggml backend/context handles
- scratch/work buffers
- optional KV cache for decoder token-by-token generation

###  Init/load path

Current initialization:
- loads `config.json`, `vocab.json`, `source.spm`, `target.spm`
- initializes RKNN encoder and decoder models
- allocates RKNN I/O buffers
- loads LM head raw files into memory

GGML initialization would instead:
- load GGUF (or equivalent) tensor weights
- map named tensors to encoder/decoder layers
- retain the same SentencePiece and vocab loading behavior

### Inference path

Current inference does:
- source SentencePiece encode
- vocab lookup
- fixed-length normalize/pad/mask
- run encoder once
- run decoder in a loop
- convert FP16 output and apply LM head
- argmax, stop at EOS, reverse vocab, target SentencePiece decode

A GGML implementation should do the same algorithmically, but each step becomes ggml graph eval. Prefer KV cache to avoid recomputing full decoder prefix every token.

### Build system changes

Current CMake links RKNN runtime headers/libs.

For GGML:
- remove RKNN runtime/header requirements
- add ggml + gguf/backend linkage
- keep Eigen only if needed for residual CPU-side ops (otherwise remove)

## Rough C++ shape

```cpp
struct ggml_marian_context_t {
    sentencepiece::SentencePieceProcessor spm_src, spm_tgt;
    std::unordered_map<std::string, int32_t> vocab;
    std::unordered_map<int32_t, std::string> vocab_inv;

    int32_t bos_token_id, eos_token_id, decoder_start_token_id, pad_token_id, unk_token_id;
    size_t enc_len, dec_len, d_model, vocab_size;

    // ggml bits
    ggml_model_t model;      // wrapper around ggml/gguf tensors
    ggml_backend_t backend;  // CPU/GPU backend
    kv_cache_t kv;           // optional but recommended
};

int init_marian_ggml_model(const std::string& model_dir, ggml_marian_context_t* ctx);
int inference_marian_ggml_model(ggml_marian_context_t* ctx,
                                const std::string& input_sentence,
                                std::string& output_sentence);
int release_marian_ggml_model(ggml_marian_context_t* ctx);
```

## Practical migration order

1. Keep tokenizer + vocab code unchanged first.
2. Replace encoder execution.
3. Replace decoder loop.
4. Move LM head into graph.
5. Add KV cache and sampling modes.
