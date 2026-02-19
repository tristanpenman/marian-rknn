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

#include <cstdint>

typedef uint16_t half;

inline uint32_t as_uint(const float x)
{
    return *reinterpret_cast<const uint32_t*>(&x);
}

inline float as_float(const uint32_t x)
{
    return *reinterpret_cast<const float*>(&x);
}

// IEEE-754 16-bit floating-point format (without infinity):
// 1-5-10, exp-15, +-131008.0, +-6.1035156E-5, +-5.9604645E-8, 3.311 digits
inline float half_to_float(const half x)
{
    const uint32_t e = (x&0x7C00) >> 10; // exponent
    const uint32_t m = (x&0x03FF) << 13; // mantissa

    // evil log2 bit hack to count leading zeros in denormalized format
    const uint32_t v = as_uint(static_cast<float>(m)) >> 23;

    // sign : normalized : denormalized
    return as_float((x&0x8000)<<16 | (e!=0)*((e+112)<<23|m) | ((e==0)&(m!=0))*((v-37)<<23|((m<<(150-v))&0x007FE000)));
}

union suf32
{
    int32_t i;
    uint32_t u;
    float f;
};

inline half float_to_half(float x)
{
    suf32 in{};
    in.f = x;
    const uint32_t sign = in.u & 0x80000000;
    in.u ^= sign;
    uint16_t w;

    if (in.u >= 0x47800000) {
        w = static_cast<uint16_t>(in.u > 0x7f800000 ? 0x7e00 : 0x7c00);
    } else if (in.u < 0x38800000) {
        in.f += 0.5f;
        w = static_cast<uint16_t>(in.u - 0x3f000000);
    } else {
        const uint32_t t = in.u + 0xc8000fff;
        w = static_cast<uint16_t>((t + (in.u >> 13 & 1)) >> 13);
    }

    w = static_cast<uint16_t>(w | sign >> 16);

    return w;
}

inline void float_to_half_array(const float *src, half *dst, const int size)
{
    for (int i = 0; i < size; i++) {
        dst[i] = float_to_half(src[i]);
    }
}

inline void half_to_float_array(const half *src, float *dst, const int size)
{
    for (int i = 0; i < size; i++) {
        dst[i] = half_to_float(src[i]);
    }
}
