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
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "logger.h"

// Define this macro to disable timing logs
// #define TIMING_DISABLED // if you don't need to print the time used, uncomment this line of code

class TIMER
{
private:
    struct timeval start_time{}, stop_time{};
    double __get_us(struct timeval t)
    {
        return (t.tv_sec * 1000000 + t.tv_usec);
    }

public:
    TIMER() = default;
    ~TIMER() = default;

    void tik()
    {
        gettimeofday(&start_time, NULL);
    }

    void tok()
    {
        gettimeofday(&stop_time, NULL);
    }

#ifdef TIMING_DISABLED
    void print_time(char *str)
    {
        // No action if TIMING_DISABLED is defined
    }
    void print_time(const char *str)
    {
        // No action if TIMING_DISABLED is defined
    }
#else
    void print_time(char *str)
    {
        static Logger timer_logger("timer");
        timer_logger(VERBOSE) << str << " use: " << get_time() << " ms";
    }
    void print_time(const char *str)
    {
        static Logger timer_logger("timer");
        timer_logger(VERBOSE) << str << " use: " << get_time() << " ms";
    }
#endif

    float get_time()
    {
        return (__get_us(stop_time) - __get_us(start_time)) / 1000;
    }
};
