// Copyright (c) 2023 Rockchip Electronics Co., Ltd. All Rights Reserved.
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

#include <sys/time.h>

#include "logger.h"

// Define this macro to disable timing logs
// #define TIMING_DISABLED // if you don't need to print the time used, uncomment this line of code

class EasyTimer
{
    timeval startTime_{};
    timeval stopTime_{};

    static double getMicroseconds(const timeval t)
    {
        return t.tv_sec * 1000000 + t.tv_usec;
    }

public:
    EasyTimer() = default;
    ~EasyTimer() = default;

    void tik()
    {
        gettimeofday(&startTime_, nullptr);
    }

    void tok()
    {
        gettimeofday(&stopTime_, nullptr);
    }

#ifdef TIMING_DISABLED
    void printTime(const char* label)
    {
    }
#else
    void printTime(const char* label) const
    {
        static Logger timerLogger("timer");
        timerLogger(VERBOSE) << label << " use: " << elapsedMs() << " ms";
    }
#endif

    [[nodiscard]] float elapsedMs() const
    {
        return (getMicroseconds(stopTime_) - getMicroseconds(startTime_)) / 1000;
    }
};
