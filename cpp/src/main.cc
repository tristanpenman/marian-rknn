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

#include <iostream>
#include <string>
#include <vector>

#include "easy_timer.h"
#include "logger.h"
#include "marian_rknn.h"

int read_user_input(char* buffer)
{
    std::cout << "Enter text to translate:\n";
    std::string line;
    if (!std::getline(std::cin, line)) {
        return -1;
    }

    if (line.size() >= MAX_USER_INPUT_LEN) {
        line.resize(MAX_USER_INPUT_LEN - 1);
    }

    std::strncpy(buffer, line.c_str(), MAX_USER_INPUT_LEN);
    buffer[MAX_USER_INPUT_LEN - 1] = '\0';

    if (strcmp(buffer, "q") == 0) {
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

    const auto input_strings = static_cast<char *>(malloc(MAX_USER_INPUT_LEN));
    const auto output_strings = static_cast<char *>(malloc(MAX_USER_INPUT_LEN));
    memset(input_strings, 0, MAX_USER_INPUT_LEN);
    memset(output_strings, 0, MAX_USER_INPUT_LEN);

    int ret = init_marian_rknn_model(model_dir, &rknn_app_ctx);
    if (ret != 0) {
        LOG(ERROR) << "init_marian_rknn_model failed";
        goto out;
    }

    LOG(INFO) << "Model init complete";
    if (positional_args.size() > 2) {
        is_receipt = true;
        for (size_t i = 2; i < positional_args.size(); i++) {
            strcat(input_strings, positional_args[i]);
            strcat(input_strings, " ");
        }

        LOG(INFO) << "Read input from cmd line: " << input_strings;
    }

    while (true) {
        if (is_receipt == false) {
            memset(input_strings, 0, MAX_USER_INPUT_LEN);
            if (const int num_word = read_user_input(input_strings); num_word == -1) {
                break;
            }
        }

        LOG(INFO) << "About to run inference...";

        timer.tik();
        ret = inference_marian_rknn_model(&rknn_app_ctx, input_strings, output_strings);
        if (ret != 0) {
            LOG(ERROR) << "marian_rknn_model inference failed. ret=" << ret;
            break;
        }
        timer.tok();
        timer.print_time("Inference time");

        LOG(INFO) << "Output: " << output_strings;

        if (is_receipt == true) {
            break;
        }
    }

out:
    ret = release_marian_rknn_model(&rknn_app_ctx);
    if (ret != 0) {
        LOG(ERROR) << "release_marian_rknn_model failed. ret=" << ret;
    }
    free(input_strings);
    free(output_strings);

    return 0;
}
