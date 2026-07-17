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

#include <atomic>
#include <mutex>
#include <ostream>
#include <sstream>
#include <string>

/**
 * Simple streaming logger
 *
 * This logger class is designed for ease-of-use and convenience, while providing performance and thread safety.
 *
 * It is intended to be used like this:
 *
 *   #include "Logger.h"
 *
 *   #undef LOG
 *   #define LOG Logger("MyCategory")
 *
 *   void myFunction()
 *   {
 *     // enable logging, uses std::cout by default
 *     Logger::configure();
 *
 *     // log info by default
 *     LOG() << "Log some stuff, maybe even some value in hex: 0x" << hex << 23030;
 *     LOG(WARNING) << "This is a warning message";
 *   }
 *
 * This would write the following to std::cout:
 *
 *   [I][MyCategory] Log some stuff, maybe even some value in hex: 0x59F6
 *   [W][MyCategory] This is a warning message
 *
 * The logger takes care of writing a new line to the end of the output, when its destructor is called. Content is
 * only written to the std::stringstream if the logger is configured and the log level is enabled.
 *
 * Although including <sstream> in this header is not great for compile times, this ensures that the necessary
 * stream operators are available for common types. And the compile time overhead could be mitigated using
 * precompiled headers.
 */
class Logger
{
public:
    enum class Level
    {
        kVerbose = 0,
        kInfo = 1,
        kWarning = 2,
        kError = 3
    };

    class Writer
    {
    public:
        Writer(Logger& logger, Level level);
        ~Writer();

        template<typename T>
        Writer& operator<<(const T& value)
        {
            if (enabled_) {
                stream_ << value;
            }

            return *this;
        }

    private:
        Logger& logger_;
        Level level_;
        std::stringstream stream_;
        bool enabled_ = false;
    };

    explicit Logger(std::string name = {});

    Writer operator()(const Level level = Level::kInfo)
    {
        return Writer{*this, level};
    }

    static void configure();
    static void configure(std::ostream& os);
    static void configure(Level minLevel);
    static void configure(std::ostream& os, Level minLevel);

    static bool verbose()
    {
        return g_minLevel.load() <= Level::kVerbose;
    }

private:
    static std::atomic<std::ostream*> g_output;
    static std::atomic<Level> g_minLevel;
    static std::mutex g_mutex;

    std::string name_;
};

#define LOG     Logger()
#define INFO    Logger::Level::kInfo
#define WARNING Logger::Level::kWarning
#define ERROR   Logger::Level::kError
#define VERBOSE Logger::Level::kVerbose
