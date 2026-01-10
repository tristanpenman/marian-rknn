#pragma once

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
