#include <cstdio>
#include <fstream>
#include <string>
#include <unordered_map>

#include <gtest/gtest.h>
#include <nlohmann/json.hpp>

#include "file_utils.h"

namespace {

TEST(FileUtilsTest, JoinPathHandlesEmptyDir)
{
    EXPECT_EQ(joinPath("", "file.txt"), "file.txt");
}

TEST(FileUtilsTest, JoinPathPreservesTrailingSlash)
{
    EXPECT_EQ(joinPath("/tmp/", "data.bin"), "/tmp/data.bin");
}

TEST(FileUtilsTest, JoinPathAddsSeparator)
{
    EXPECT_EQ(joinPath("/tmp", "data.bin"), "/tmp/data.bin");
}

TEST(FileUtilsTest, ReadMapFromFileParsesJsonObject)
{
    const std::string path = "read_map_fixture.json";
    std::ofstream output(path);
    output << R"({"hello": 1, "world": 2})";
    output.close();

    std::unordered_map<std::string, int> values;
    readMapFromFile(path, values);

    EXPECT_EQ(values.size(), 2u);
    EXPECT_EQ(values.at("hello"), 1);
    EXPECT_EQ(values.at("world"), 2);

    std::remove(path.c_str());
}

TEST(FileUtilsTest, JoinPathWithRootDirectory)
{
    EXPECT_EQ(joinPath("/", "file.txt"), "/file.txt");
}

TEST(FileUtilsTest, ReadMapFromFileThrowsOnMissingFile)
{
    std::unordered_map<std::string, int> values = {{"existing", 1}};
    EXPECT_THROW(readMapFromFile("does_not_exist.json", values), std::runtime_error);
    EXPECT_EQ(values.size(), 1u);
}

TEST(FileUtilsTest, ReadMapFromFileRejectsNonObjectJson)
{
    const std::string path = "read_map_non_object_fixture.json";
    std::ofstream output(path);
    output << R"([1, 2, 3])";
    output.close();

    std::unordered_map<std::string, int> values;
    EXPECT_THROW(readMapFromFile(path, values), std::runtime_error);

    std::remove(path.c_str());
}

TEST(FileUtilsTest, ReadMapFromFileRejectsNonIntegerValue)
{
    const std::string path = "read_map_non_integer_fixture.json";
    std::ofstream output(path);
    output << R"({"hello": 1, "world": "two"})";
    output.close();

    std::unordered_map<std::string, int> values;
    EXPECT_THROW(readMapFromFile(path, values), nlohmann::json::exception);

    std::remove(path.c_str());
}

}  // namespace
