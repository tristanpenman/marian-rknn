#pragma once

#include <cstdint>
#include <cstring>

typedef unsigned short half;
typedef unsigned short ushort;

inline uint32_t as_uint(const float x)
{
    return *(uint*)&x;
}

inline float as_float(const uint32_t x)
{
    return *(float*)&x;
}

// IEEE-754 16-bit floating-point format (without infinity): 1-5-10, exp-15, +-131008.0, +-6.1035156E-5, +-5.9604645E-8, 3.311 digits
inline float half_to_float(half x)
{
    const uint32_t e = (x&0x7C00) >> 10; // exponent
    const uint32_t m = (x&0x03FF) << 13; // mantissa
    const uint32_t v = as_uint((float)m) >> 23; // evil log2 bit hack to count leading zeros in denormalized format

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
    suf32 in;
    in.f = x;
    uint32_t sign = in.u & 0x80000000;
    in.u ^= sign;
    ushort w;

    if (in.u >= 0x47800000) {
        w = (ushort)(in.u > 0x7f800000 ? 0x7e00 : 0x7c00);
    } else if (in.u < 0x38800000) {
        in.f += 0.5f;
        w = (ushort)(in.u - 0x3f000000);
    } else {
        uint32_t t = in.u + 0xc8000fff;
        w = (ushort)((t + ((in.u >> 13) & 1)) >> 13);
    }

    w = (ushort)(w | (sign >> 16));

    return w;
}

inline void float_to_half_array(float *src, half *dst, int size)
{
    for (int i = 0; i < size; i++) {
        dst[i] = float_to_half(src[i]);
    }
}

inline void half_to_float_array(half *src, float *dst, int size)
{
    for (int i = 0; i < size; i++) {
        dst[i] = half_to_float(src[i]);
    }
}
