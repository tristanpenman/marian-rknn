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
#include <exception>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include "logger.h"
#include "marian_rknn.h"
#include "time_utils.h"

namespace {

void logMetric(const std::string& label, double valueMs)
{
    LOG(INFO) << label << ": " << std::fixed << std::setprecision(3) << valueMs << " ms";
}

}  // namespace

int main(const int argc, char** argv)
{
    bool eigen = false;
    bool verbose = false;
    std::vector<const char*> positionalArgs;
    positionalArgs.reserve(static_cast<size_t>(argc - 1));
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-v") == 0 || strcmp(argv[i], "--verbose") == 0) {
            verbose = true;
            continue;
        }
        if (strcmp(argv[i], "--eigen") == 0) {
            eigen = true;
            continue;
        }
        positionalArgs.push_back(argv[i]);
    }

    Logger::configure(std::cout, verbose ? Logger::Level::kVerbose : Logger::Level::kInfo);
    LOG(INFO) << "Marian RKNN Benchmark";

    if (positionalArgs.size() != 3) {
        LOG(ERROR) << "Usage: " << argv[0] << " [-v|--verbose] [--eigen] <modelDir> <input_file> <maxSeconds>";
        return -1;
    }

    const char* modelDir = positionalArgs[0];
    const std::string inputPath = positionalArgs[1];
    const std::string maxSecondsArg = positionalArgs[2];

    double maxSeconds = 0.0;
    try {
        maxSeconds = std::stod(maxSecondsArg);
    } catch (const std::exception& ex) {
        LOG(ERROR) << "Failed to parse maxSeconds: " << ex.what();
        return -1;
    }

    if (maxSeconds <= 0.0) {
        LOG(ERROR) << "maxSeconds must be greater than 0";
        return -1;
    }

    std::ifstream inputStream(inputPath);
    if (!inputStream) {
        LOG(ERROR) << "Failed to open input file: " << inputPath;
        return -1;
    }

    std::vector<std::string> inputLines;
    std::string line;
    while (std::getline(inputStream, line)) {
        if (line.empty()) {
            continue;
        }
        inputLines.push_back(line);
    }

    if (inputLines.empty()) {
        LOG(ERROR) << "No non-empty input lines found in: " << inputPath;
        return -1;
    }

    RknnMarianContext appCtx;
    int ret = initMarianRknnModel(modelDir, &appCtx, eigen);
    if (ret != 0) {
        LOG(ERROR) << "initMarianRknnModel failed";
        return 1;
    }

    LOG(INFO) << "Model init complete";

    RknnMarianInferenceStats totalStats;
    totalStats.reset();
    size_t totalSentences = 0;
    const auto startTime = std::chrono::steady_clock::now();
    size_t index = 0;
    std::string outputText;

    while (true) {
        const auto now = std::chrono::steady_clock::now();
        if (elapsedSeconds(startTime, now) >= maxSeconds) {
            break;
        }

        RknnMarianInferenceStats stats;
        stats.reset();
        ret = inferenceMarianRknnModel(&appCtx, inputLines[index], outputText, &stats);
        if (ret != 0) {
            LOG(ERROR) << "marian_rknn_model inference failed. ret=" << ret;
            break;
        }

        totalStats.accumulate(stats);
        totalSentences++;

        index = (index + 1) % inputLines.size();
    }
    const auto endTime = std::chrono::steady_clock::now();

    ret = releaseMarianRknnModel(&appCtx);
    if (ret != 0) {
        LOG(ERROR) << "releaseMarianRknnModel failed. ret=" << ret;
    }

    const double elapsedSecondsTotal = elapsedSeconds(startTime, endTime);
    LOG(INFO) << "Benchmark complete";
    LOG(INFO) << "Elapsed: " << std::fixed << std::setprecision(3) << elapsedSecondsTotal << " s";
    LOG(INFO) << "Sentences: " << totalSentences;
    if (elapsedSecondsTotal > 0.0) {
        LOG(INFO) << "Sentences/sec: " << std::fixed << std::setprecision(3)
                  << (static_cast<double>(totalSentences) / elapsedSecondsTotal);
    }

    if (totalSentences > 0) {
        logMetric("Total time", totalStats.totalMs);
        logMetric("Encoder time", totalStats.encoderMs);
        logMetric("Decoder time", totalStats.decoderMs);
        logMetric("LM head time", totalStats.lmHeadMs);

        logMetric("Avg total time per sentence", totalStats.totalMs / totalSentences);
        logMetric("Avg encoder time per sentence", totalStats.encoderMs / totalSentences);
        logMetric("Avg decoder time per sentence", totalStats.decoderMs / totalSentences);
        logMetric("Avg LM head time per sentence", totalStats.lmHeadMs / totalSentences);

        LOG(INFO) << "Input tokens: " << totalStats.inputTokens;
        LOG(INFO) << "Output tokens: " << totalStats.outputTokens;
        LOG(INFO) << "Decoder iterations: " << totalStats.decoderIterations;
        if (elapsedSecondsTotal > 0.0) {
            LOG(INFO) << "Input tokens/sec: " << std::fixed << std::setprecision(3)
                      << (static_cast<double>(totalStats.inputTokens) / elapsedSecondsTotal);
            LOG(INFO) << "Output tokens/sec: " << std::fixed << std::setprecision(3)
                      << (static_cast<double>(totalStats.outputTokens) / elapsedSecondsTotal);
        }
    }

    return 0;
}
