// Copyright (c) 2023 Rockchip Electronics Co., Ltd. All Rights Reserved.
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

#include <cstdlib>
#include <cstring>
#include <iostream>
#include <optional>
#include <string>
#include <vector>

#include "easy_timer.h"
#include "logger.h"
#include "marian_rknn.h"

namespace {

constexpr size_t kMaxUserInputLen = 1024;

void printUsage(const char* program)
{
    std::cout
        << "Usage: " << program
        << " [-v|--verbose] [--eigen] [--cores <numCores>] <modelDir>"
        << " <sentence ...>"
        << std::endl;
}

int readUserInput(std::string& line)
{
    std::cout << "Enter text to translate:\n";
    line.clear();
    if (!std::getline(std::cin, line)) {
        return -1;
    }

    return 0;
}

}  // namespace

int main(const int argc, char** argv)
{
    bool eigen = false;
    bool verbose = false;
    std::optional<int> numCores;
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
        if (strcmp(argv[i], "--cores") == 0) {
            if (i + 1 >= argc) {
                LOG(WARNING) << "--cores option requires an argument specifying the number of cores to use";
                return -1;
            }
            numCores = std::atoi(argv[++i]);
            if (*numCores <= 0 || *numCores > 3) {
                LOG(WARNING) << "Invalid number of cores specified: " << argv[i];
                return -1;
            }
            continue;
        }
        positionalArgs.push_back(argv[i]);
    }

    Logger::configure(std::cout, verbose ? Logger::Level::kVerbose : Logger::Level::kInfo);
    if (positionalArgs.empty()) {
        printUsage(argv[0]);
        return -1;
    }

    LOG(INFO) << "Marian RKNN Translator Demo";

    EasyTimer timer;
    bool isReceipt = false;
    const char* modelDir = positionalArgs[0];

    RknnMarianContext appCtx;

    std::string inputText;
    std::string outputText;

    int ret = initMarianRknnModel(modelDir, &appCtx, eigen, numCores);
    if (ret != 0) {
        LOG(ERROR) << "initMarianRknnModel failed";
        return 1;
    }

    LOG(INFO) << "Model init complete";
    if (positionalArgs.size() > 1) {
        isReceipt = true;
        for (size_t i = 1; i < positionalArgs.size(); i++) {
            inputText += positionalArgs[i];
            inputText += " ";
        }

        LOG(INFO) << "Read input from cmd line: " << inputText;
    }

    while (true) {
        if (!isReceipt) {
            if (ret = readUserInput(inputText); ret == -1) {
                break;
            }
        }

        if (inputText.size() >= kMaxUserInputLen) {
            inputText.resize(kMaxUserInputLen - 1);
        }

        LOG(INFO) << "About to run inference...";

        timer.tik();
        ret = inferenceMarianRknnModel(&appCtx, inputText, outputText);
        if (ret != 0) {
            LOG(ERROR) << "marian_rknn_model inference failed. ret=" << ret;
            break;
        }
        timer.tok();
        timer.printTime("Inference time");

        LOG(INFO) << "Output: " << outputText;

        if (isReceipt) {
            break;
        }
    }

    ret = releaseMarianRknnModel(&appCtx);
    if (ret != 0) {
        LOG(ERROR) << "releaseMarianRknnModel failed. ret=" << ret;
    }

    return 0;
}
