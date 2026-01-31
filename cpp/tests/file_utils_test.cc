#include <cstdio>
#include <fstream>
#include <string>
#include <unordered_map>

#include <gtest/gtest.h>

#include "file_utils.h"

namespace {

TEST(FileUtilsTest, JoinPathHandlesEmptyDir)
{
    EXPECT_EQ(join_path("", "file.txt"), "file.txt");
}

TEST(FileUtilsTest, JoinPathPreservesTrailingSlash)
{
    EXPECT_EQ(join_path("/tmp/", "data.bin"), "/tmp/data.bin");
}

TEST(FileUtilsTest, JoinPathAddsSeparator)
{
    EXPECT_EQ(join_path("/tmp", "data.bin"), "/tmp/data.bin");
}

TEST(FileUtilsTest, ReadMapFromFileParsesJsonObject)
{
    const std::string path = "read_map_fixture.json";
    std::ofstream output(path);
    output << R"({"hello": 1, "world": 2})";
    output.close();

    std::unordered_map<std::string, int> values;
    read_map_from_file(path, values);

    EXPECT_EQ(values.size(), 2u);
    EXPECT_EQ(values.at("hello"), 1);
    EXPECT_EQ(values.at("world"), 2);

    std::remove(path.c_str());
}

}  // namespace
