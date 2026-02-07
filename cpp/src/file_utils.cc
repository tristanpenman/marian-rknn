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

#include <cstdio>
#include <fstream>
#include <string>

#include <nlohmann/json.hpp>

#include "file_utils.h"
#include "logger.h"

using json = nlohmann::json;

std::string join_path(const std::string& dir, const char* name)
{
    if (dir.empty()) {
        return {name};
    }

    if (dir.back() == '/') {
        return dir + name;
    }

    return dir + "/" + name;
}

long read_data_from_file(const char *path, char **out_data)
{
    FILE *fp = fopen(path, "rb");
    if (fp == nullptr) {
        LOG(ERROR) << "Failed to open file: " << path;
        return -1;
    }
    fseek(fp, 0, SEEK_END);
    long file_size = ftell(fp);
    char *data = static_cast<char *>(malloc(file_size + 1));
    data[file_size] = 0;
    fseek(fp, 0, SEEK_SET);
    if (file_size != fread(data, 1, file_size, fp)) {
        LOG(ERROR) << "Failed to read file: " << path;
        free(data);
        fclose(fp);
        return -1;
    }
    fclose(fp);
    *out_data = data;
    return file_size;
}

int read_fp32_from_file(const char *path, int len, float *data)
{
    FILE *fp = fopen(path, "rb");
    if (fp == nullptr) {
        LOG(ERROR) << "Failed to open file: " << path;
        return -1;
    }
    size_t read_len = fread(data, sizeof(float), len, fp);
    fclose(fp);
    if (read_len != static_cast<size_t>(len)) {
        LOG(ERROR) << "Failed to read file: " << path;
        return -1;
    }
    return 0;
}

void read_map_from_file(const std::string& path, std::unordered_map<std::string, int>& map)
{
    std::ifstream f(path);
    if (!f) {
        throw std::runtime_error("Failed to open: " + path);
    }

    json j;
    f >> j;

    if (!j.is_object()) {
        throw std::runtime_error("document is not an object: " + path);
    }

    map.clear();
    map.reserve(j.size());

    for (auto it = j.begin(); it != j.end(); ++it) {
        map.emplace(it.key(), it.value().get<int>());
    }
}
