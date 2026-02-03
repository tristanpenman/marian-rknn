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
