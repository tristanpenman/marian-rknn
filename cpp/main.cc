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

#include <cstdio>

#include "marian_rknn.h"
#include "easy_timer.h"

void safe_flush()
{
    char c;
    while ((c = getchar()) != '\n' && c != EOF);
}

int read_user_input(char* buffer)
{
    rewind(stdin);
    printf("Enter text to translate:\n");
    // TODO: use C++ standard library here
    scanf("%[^\n]", buffer);
    safe_flush();

    if (strcmp(buffer, "q") == 0) {
        return -1;
    }

    return 0;
}

int main(int argc, char **argv)
{
    if (argc < 5) {
        printf("%s <encoder_path> <decoder_path> <source_spm> <target_spm> <sentence ...>\n", argv[0]);
        return -1;
    }

    TIMER timer;

    const char *encoder_path = argv[1];
    const char *decoder_path = argv[2];
    const char* source_spm_path = argv[3];
    const char* target_spm_path = argv[4];

    rknn_marian_rknn_context_t rknn_app_ctx;

    char *input_strings = (char *)malloc(MAX_USER_INPUT_LEN);
    char *output_strings = (char *)malloc(MAX_USER_INPUT_LEN);
    memset(input_strings, 0, MAX_USER_INPUT_LEN);
    memset(output_strings, 0, MAX_USER_INPUT_LEN);

    bool is_receipt = false;

    int ret = init_marian_rknn_model(
        encoder_path,
        decoder_path,
        source_spm_path,
        target_spm_path,
        &rknn_app_ctx);

    if (ret != 0) {
        printf("init_marian_rknn_model fail!\n");
        goto out;
    }

    printf("--> model init complete\n");

    // receipt string to translate
    if (argc > 5) {
        is_receipt = true;
        for (int i = 5; i < argc; i++) {
            strcat(input_strings, argv[i]);
            strcat(input_strings, " ");
        }

        printf("--> read input from cmd line: %s\n", input_strings);
    }

    while (1) {
        if (is_receipt == false) {
            memset(input_strings, 0, MAX_USER_INPUT_LEN);
            int num_word = read_user_input(input_strings);
            if (num_word == -1) {
                break;
            }
        }

        printf("--> about to run inference...\n");

        timer.tik();
        ret = inference_marian_rknn_model(&rknn_app_ctx, input_strings, output_strings);
        if (ret != 0) {
            printf("marian_rknn_model inference fail! ret=%d\n", ret);
            break;
        }
        timer.tok();
        timer.print_time("inference time");

        printf("output_strings: %s\n", output_strings);

        if (is_receipt == true) {
            break;
        }
    }

out:
    ret = release_marian_rknn_model(&rknn_app_ctx);
    if (ret != 0) {
        printf("release_marian_rknn_model fail! ret=%d\n", ret);
    }
    free(input_strings);
    free(output_strings);

    return 0;
}
