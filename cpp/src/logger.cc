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

#include "logger.h"

#include <iostream>
#include <mutex>
#include <utility>

std::atomic<std::ostream*> Logger::g_output = nullptr;
std::atomic<Logger::Level> Logger::g_minLevel = Logger::Level::kInfo;
std::mutex Logger::g_mutex;

namespace {

const char* levelLabel(Logger::Level level)
{
    switch (level) {
    case Logger::Level::kInfo:
        return "I";
    case Logger::Level::kWarning:
        return "W";
    case Logger::Level::kError:
        return "E";
    case Logger::Level::kVerbose:
        return "V";
    default:
        return "U";
    }
}

}  // namespace

Logger::Logger(std::string name)
    : name_(std::move(name))
{
}

void Logger::configure()
{
    g_output = &std::cout;
}

void Logger::configure(std::ostream& os)
{
    g_output = &os;
}

void Logger::configure(Level minLevel)
{
    g_minLevel = minLevel;
}

void Logger::configure(std::ostream& os, Level minLevel)
{
    g_output = &os;
    g_minLevel = minLevel;
}

Logger::Writer::Writer(Logger& logger, Level level)
    : logger_(logger)
    , level_(level)
{
    enabled_ = g_output.load() && level >= g_minLevel.load();
    if (!enabled_) {
        return;
    }

    stream_ << "[" << levelLabel(level) << "]";
    if (!logger_.name_.empty()) {
        stream_ << "[" << logger_.name_ << "]";
    }
    stream_ << " ";
}

Logger::Writer::~Writer()
{
    if (!enabled_) {
        return;
    }

    std::ostream* os = g_output.load();
    if (!os) {
        return;
    }

    std::lock_guard lock(g_mutex);
    *os << stream_.str() << '\n';
}
