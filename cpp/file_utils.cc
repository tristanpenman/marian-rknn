#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "file_utils.h"

int read_data_from_file(const char *path, char **out_data)
{
    FILE *fp = fopen(path, "rb");
    if (fp == NULL) {
        printf("fopen %s fail!\n", path);
        return -1;
    }
    fseek(fp, 0, SEEK_END);
    int file_size = ftell(fp);
    char *data = (char *)malloc(file_size+1);
    data[file_size] = 0;
    fseek(fp, 0, SEEK_SET);
    if (file_size != fread(data, 1, file_size, fp)) {
        printf("fread %s fail!\n", path);
        free(data);
        fclose(fp);
        return -1;
    }
    if (fp) {
        fclose(fp);
    }
    *out_data = data;
    return file_size;
}

int read_fp32_from_file(const char* path, int len, float* data)
{
    FILE *fp = fopen(path, "rb");
    if (fp == NULL) {
        printf("fopen %s fail!\n", path);
        return -1;
    }
    fread(data, sizeof(float), len, fp);
    fclose(fp);
    return 0;
}
