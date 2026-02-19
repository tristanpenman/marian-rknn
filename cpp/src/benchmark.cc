// Copyright (c) 2026 Tristan Penman
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <chrono>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include "logger.h"
#include "marian_rknn.h"
#include "time_utils.h"

namespace {

void log_metric(const std::string& label, double value_ms)
{
    LOG(INFO) << label << ": " << std::fixed << std::setprecision(3) << value_ms << " ms";
}

}  // namespace

int main(const int argc, char **argv)
{
    bool verbose = false;
    std::vector<const char*> positional_args;
    positional_args.reserve(static_cast<size_t>(argc - 1));
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-v") == 0 || strcmp(argv[i], "--verbose") == 0) {
            verbose = true;
            continue;
        }
        positional_args.push_back(argv[i]);
    }

    Logger::configure(std::cout, verbose ? Logger::Level::Verbose : Logger::Level::Info);
    LOG(INFO) << "Marian RKNN Benchmark";

    if (positional_args.size() != 3) {
        LOG(ERROR) << "Usage: " << argv[0] << " [-v|--verbose] <model_dir> <input_file> <max_seconds>";
        return -1;
    }

    const char *model_dir = positional_args[0];
    const std::string input_path = positional_args[1];
    const std::string max_seconds_arg = positional_args[2];

    double max_seconds = 0.0;
    try {
        max_seconds = std::stod(max_seconds_arg);
    } catch (const std::exception& ex) {
        LOG(ERROR) << "Failed to parse max_seconds: " << ex.what();
        return -1;
    }

    if (max_seconds <= 0.0) {
        LOG(ERROR) << "max_seconds must be greater than 0";
        return -1;
    }

    std::ifstream input_stream(input_path);
    if (!input_stream) {
        LOG(ERROR) << "Failed to open input file: " << input_path;
        return -1;
    }

    std::vector<std::string> input_lines;
    std::string line;
    while (std::getline(input_stream, line)) {
        if (line.empty()) {
            continue;
        }
        input_lines.push_back(line);
    }

    if (input_lines.empty()) {
        LOG(ERROR) << "No non-empty input lines found in: " << input_path;
        return -1;
    }

    rknn_marian_rknn_context_t rknn_app_ctx;
    int ret = init_marian_rknn_model(model_dir, &rknn_app_ctx);
    if (ret != 0) {
        LOG(ERROR) << "init_marian_rknn_model failed";
        return 1;
    }

    LOG(INFO) << "Model init complete";

    rknn_marian_inference_stats_t total_stats;
    total_stats.reset();
    size_t total_sentences = 0;
    const auto start_time = std::chrono::steady_clock::now();
    size_t index = 0;
    std::string output_text;

    while (true) {
        const auto now = std::chrono::steady_clock::now();
        if (elapsed_seconds(start_time, now) >= max_seconds) {
            break;
        }

        rknn_marian_inference_stats_t stats;
        stats.reset();
        ret = inference_marian_rknn_model(&rknn_app_ctx, input_lines[index], output_text, &stats);
        if (ret != 0) {
            LOG(ERROR) << "marian_rknn_model inference failed. ret=" << ret;
            break;
        }

        total_stats.accumulate(stats);
        total_sentences++;

        index = (index + 1) % input_lines.size();
    }
    const auto end_time = std::chrono::steady_clock::now();

    ret = release_marian_rknn_model(&rknn_app_ctx);
    if (ret != 0) {
        LOG(ERROR) << "release_marian_rknn_model failed. ret=" << ret;
    }

    const double elapsed_s = elapsed_seconds(start_time, end_time);
    LOG(INFO) << "Benchmark complete";
    LOG(INFO) << "Elapsed: " << std::fixed << std::setprecision(3) << elapsed_s << " s";
    LOG(INFO) << "Sentences: " << total_sentences;
    if (elapsed_s > 0.0) {
        LOG(INFO) << "Sentences/sec: " << std::fixed << std::setprecision(3)
                  << (static_cast<double>(total_sentences) / elapsed_s);
    }

    if (total_sentences > 0) {
        log_metric("Total time", total_stats.total_ms);
        log_metric("Encoder time", total_stats.encoder_ms);
        log_metric("Decoder time", total_stats.decoder_ms);
        log_metric("LM head time", total_stats.lm_head_ms);

        log_metric("Avg total time per sentence", total_stats.total_ms / total_sentences);
        log_metric("Avg encoder time per sentence", total_stats.encoder_ms / total_sentences);
        log_metric("Avg decoder time per sentence", total_stats.decoder_ms / total_sentences);
        log_metric("Avg LM head time per sentence", total_stats.lm_head_ms / total_sentences);

        LOG(INFO) << "Input tokens: " << total_stats.input_tokens;
        LOG(INFO) << "Output tokens: " << total_stats.output_tokens;
        LOG(INFO) << "Decoder iterations: " << total_stats.decoder_iterations;
        if (elapsed_s > 0.0) {
            LOG(INFO) << "Input tokens/sec: " << std::fixed << std::setprecision(3)
                      << (static_cast<double>(total_stats.input_tokens) / elapsed_s);
            LOG(INFO) << "Output tokens/sec: " << std::fixed << std::setprecision(3)
                      << (static_cast<double>(total_stats.output_tokens) / elapsed_s);
        }
    }

    return 0;
}
