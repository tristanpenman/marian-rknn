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

#pragma once

#include <string>
#include <unordered_map>

/**
 * @brief Join directory and file name to form a path
 *
 * @param dir [in] Directory path
 * @param name [in] File name
 * @return std::string Joined path
 */
std::string join_path(const std::string& dir, const char* name);

/**
 * @brief Read data from file
 *
 * @param path [in] File path
 * @param out_data [out] Read data
 * @return int -1: error; > 0: Read data size
 */
long read_data_from_file(const char *path, char **out_data);

/**
 * @brief Read floats (fp32) from file
 *
 * @param path [in] File path
 * @param len [in] Number of floats to read
 * @param out_data [out] Read data
 * @return int -1: error; > 0: Read data size
 */
int read_fp32_from_file(const char *path, int len, float *out_data);

/**
 * @brief Read a map of strings and integers from file
 *
 * @param path [in] File path
 * @param out_data [out] Read data
 */
void read_map_from_file(const std::string& path, std::unordered_map<std::string, int>& out_data);
