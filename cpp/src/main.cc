// Copyright (c) 2023 Rockchip Electronics Co., Ltd. All Rights Reserved.
// Copyright (c) 2025 Tristan Penman
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

#include <array>
#include <iostream>
#include <string>
#include <vector>

#include "easy_timer.h"
#include "logger.h"
#include "marian_rknn.h"

int read_user_input(std::string &line)
{
    std::cout << "Enter text to translate:\n";
    line.clear();
    if (!std::getline(std::cin, line)) {
        return -1;
    }

    return 0;
}

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
    LOG(INFO) << "Marian RKNN Translator Demo";

    if (positional_args.empty()) {
        LOG(ERROR) << "Usage: " << argv[0] << " [-v|--verbose] <model_dir> <sentence ...>";
        return -1;
    }

    TIMER timer;
    bool is_receipt = false;
    const char *model_dir = positional_args[0];

    rknn_marian_rknn_context_t rknn_app_ctx;

    std::string input_text;
    std::string output_text;

    int ret = init_marian_rknn_model(model_dir, &rknn_app_ctx);
    if (ret != 0) {
        LOG(ERROR) << "init_marian_rknn_model failed";
        return 1;
    }

    LOG(INFO) << "Model init complete";
    if (positional_args.size() > 2) {
        is_receipt = true;
        for (size_t i = 2; i < positional_args.size(); i++) {
            input_text += positional_args[i];
            input_text += " ";
        }

        LOG(INFO) << "Read input from cmd line: " << input_text;
    }

    while (true) {
        if (is_receipt == false) {
            if (ret = read_user_input(input_text); ret == -1) {
                break;
            }
        }

        if (input_text.size() >= MAX_USER_INPUT_LEN) {
            input_text.resize(MAX_USER_INPUT_LEN - 1);
        }

        LOG(INFO) << "About to run inference...";

        timer.tik();
        ret = inference_marian_rknn_model(&rknn_app_ctx, input_text, output_text);
        if (ret != 0) {
            LOG(ERROR) << "marian_rknn_model inference failed. ret=" << ret;
            break;
        }
        timer.tok();
        timer.print_time("Inference time");

        LOG(INFO) << "Output: " << output_text;

        if (is_receipt == true) {
            break;
        }
    }

    ret = release_marian_rknn_model(&rknn_app_ctx);
    if (ret != 0) {
        LOG(ERROR) << "release_marian_rknn_model failed. ret=" << ret;
    }

    return 0;
}
