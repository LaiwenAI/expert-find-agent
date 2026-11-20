#include <cstdint>
#include <limits>

#if defined(__x86_64__) && defined(__GNUC__)
#include <immintrin.h>
#define TITLE_HAS_X86 1
#else
#define TITLE_HAS_X86 0
#endif

static std::int32_t dot_scalar(const std::int8_t* row, const std::int8_t* query, int dim) {
    std::int32_t result = 0;
    for (int i = 0; i < dim; ++i) {
        result += static_cast<int>(row[i]) * static_cast<int>(query[i]);
    }
    return result;
}

#if TITLE_HAS_X86
__attribute__((target("avx512vnni,avx512bw,avx512f")))
static std::int32_t dot_vnni(const std::int8_t* row, const std::uint8_t* shifted_query,
                             int dim, std::int32_t row_sum) {
    __m512i value = _mm512_setzero_si512();
    for (int i = 0; i < dim; i += 64) {
        const __m512i source = _mm512_loadu_si512(row + i);
        const __m512i query = _mm512_loadu_si512(shifted_query + i);
        value = _mm512_dpbusd_epi32(value, query, source);
    }
    return _mm512_reduce_add_epi32(value) - 128 * row_sum;
}
#endif

extern "C" bool title_vnni_available() {
#if TITLE_HAS_X86
    return __builtin_cpu_supports("avx512vnni") && __builtin_cpu_supports("avx512bw");
#else
    return false;
#endif
}

extern "C" int title_top_int8(
        const std::int8_t* codes, const float* scales, const std::int32_t* row_sums,
        const std::int8_t* query, const std::uint8_t* shifted_query, float query_scale,
        const std::uint8_t* eligible, int rows, int dim, int limit, bool use_vnni,
        int* selected) {
    if (limit < 1 || limit > 4 || dim < 1 || dim > 16384) {
        return -1;
    }
    const bool vectorized = use_vnni && dim % 64 == 0 && title_vnni_available();
    double best[4];
    for (int i = 0; i < limit; ++i) {
        best[i] = -std::numeric_limits<double>::infinity();
        selected[i] = -1;
    }
    int retained = 0;
    for (int source = 0; source < rows; ++source) {
        if (!eligible[source]) {
            continue;
        }
        const std::int8_t* row = codes + static_cast<std::int64_t>(source) * dim;
        std::int32_t score;
#if TITLE_HAS_X86
        if (vectorized) {
            score = dot_vnni(row, shifted_query, dim, row_sums[source]);
        } else
#endif
        {
            score = dot_scalar(row, query, dim);
        }
        const double scaled = static_cast<double>(score) * scales[source] * query_scale;
        // Scan in source order. Strict comparison preserves the stable source ID tie-break.
        for (int position = 0; position < limit; ++position) {
            if (scaled > best[position]) {
                for (int next = limit - 1; next > position; --next) {
                    best[next] = best[next - 1];
                    selected[next] = selected[next - 1];
                }
                best[position] = scaled;
                selected[position] = source;
                if (retained < limit) {
                    ++retained;
                }
                break;
            }
        }
    }
    return retained;
}

// Return all rows whose upper bound can reach the second-largest lower bound.
extern "C" int title_top_two_guard(
        const std::int8_t* codes, const float* scales, const std::int32_t* row_sums,
        const float* residuals, const std::int8_t* query, const std::uint8_t* shifted_query,
        float query_scale, double query_norm, double common_error, int rows, int dim,
        double* upper, int* selected) {
    if (rows < 2 || dim < 1 || dim > 16384) {
        return -1;
    }
    const bool vectorized = dim % 64 == 0 && title_vnni_available();
    double first_lower = -std::numeric_limits<double>::infinity();
    double second_lower = first_lower;
    for (int source = 0; source < rows; ++source) {
        const std::int8_t* row = codes + static_cast<std::int64_t>(source) * dim;
        std::int32_t integer_score;
#if TITLE_HAS_X86
        if (vectorized) {
            integer_score = dot_vnni(row, shifted_query, dim, row_sums[source]);
        } else
#endif
        {
            integer_score = dot_scalar(row, query, dim);
        }
        const double score = static_cast<double>(integer_score) * scales[source] * query_scale;
        const double error = residuals[source] * query_norm + common_error;
        upper[source] = score + error;
        const double lower = score - error;
        if (lower > first_lower) {
            second_lower = first_lower;
            first_lower = lower;
        } else if (lower > second_lower) {
            second_lower = lower;
        }
    }
    int retained = 0;
    for (int source = 0; source < rows; ++source) {
        if (upper[source] >= second_lower) {
            selected[retained++] = source;
        }
    }
    return retained;
}
