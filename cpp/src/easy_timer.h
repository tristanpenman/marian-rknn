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
    timeval start_time{};
    timeval stop_time{};

    static double _get_us(const timeval t)
    {
        return t.tv_sec * 1000000 + t.tv_usec;
    }

public:
    EasyTimer() = default;
    ~EasyTimer() = default;

    void tik()
    {
        gettimeofday(&start_time, nullptr);
    }

    void tok()
    {
        gettimeofday(&stop_time, nullptr);
    }

#ifdef TIMING_DISABLED
    void print_time(const char *str)
    {
        // No action if TIMING_DISABLED is defined
    }
#else
    void print_time(const char *str) const
    {
        static Logger timer_logger("timer");
        timer_logger(VERBOSE) << str << " use: " << get_time() << " ms";
    }
#endif

    [[nodiscard]] float get_time() const
    {
        return (_get_us(stop_time) - _get_us(start_time)) / 1000;
    }
};
