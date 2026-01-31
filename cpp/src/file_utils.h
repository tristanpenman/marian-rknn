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
int read_data_from_file(const char *path, char **out_data);

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
