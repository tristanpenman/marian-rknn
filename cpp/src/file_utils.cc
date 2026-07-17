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

#include "file_utils.h"

#include <cstdlib>
#include <cstdio>
#include <fstream>
#include <stdexcept>
#include <string>

#include <nlohmann/json.hpp>

#include "logger.h"

using json = nlohmann::json;

std::string joinPath(const std::string& dir, const char* name)
{
    if (dir.empty()) {
        return {name};
    }

    if (dir.back() == '/') {
        return dir + name;
    }

    return dir + "/" + name;
}

long readDataFromFile(const char* path, char** outData)
{
    FILE* fp = fopen(path, "rb");
    if (fp == nullptr) {
        LOG(ERROR) << "Failed to open file: " << path;
        return -1;
    }
    fseek(fp, 0, SEEK_END);
    const long fileSize = ftell(fp);
    char* data = static_cast<char*>(malloc(fileSize + 1));
    data[fileSize] = 0;
    fseek(fp, 0, SEEK_SET);
    if (fileSize != fread(data, 1, fileSize, fp)) {
        LOG(ERROR) << "Failed to read file: " << path;
        free(data);
        fclose(fp);
        return -1;
    }
    fclose(fp);
    *outData = data;
    return fileSize;
}

int readFp32FromFile(const char* path, int len, float* data)
{
    FILE* fp = fopen(path, "rb");
    if (fp == nullptr) {
        LOG(ERROR) << "Failed to open file: " << path;
        return -1;
    }
    const size_t readLen = fread(data, sizeof(float), len, fp);
    fclose(fp);
    if (readLen != static_cast<size_t>(len)) {
        LOG(ERROR) << "Failed to read file: " << path;
        return -1;
    }
    return 0;
}

void readMapFromFile(const std::string& path, std::unordered_map<std::string, int>& outData)
{
    std::ifstream file(path);
    if (!file) {
        throw std::runtime_error("Failed to open: " + path);
    }

    json document;
    file >> document;

    if (!document.is_object()) {
        throw std::runtime_error("document is not an object: " + path);
    }

    outData.clear();
    outData.reserve(document.size());

    for (auto it = document.begin(); it != document.end(); ++it) {
        outData.emplace(it.key(), it.value().get<int>());
    }
}
