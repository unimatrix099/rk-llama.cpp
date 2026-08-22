#include "rknpu2-calibration.h"

#include <vector>
#include <cmath>
#include <algorithm>
#include <cstdlib>
#include <limits>
#include <cstring>
#include <omp.h>

#ifdef __ARM_NEON
#include <arm_neon.h>
#endif

namespace rknpu2_calibration {

// --- Calibration Implementations ---

float calculate_percentile_amax(const float * data, size_t n_elements, float percentile) {
    if (n_elements == 0) {
        return 0.0f;
    }

    // Creating a vector of absolute values. We need a mutable copy to use nth_element.
    std::vector<float> abs_values(n_elements);
    #pragma omp parallel for
    for (size_t i = 0; i < n_elements; ++i) {
        abs_values[i] = std::abs(data[i]);
    }

    // Finding the index corresponding to the percentile
    size_t index = static_cast<size_t>((float)n_elements * percentile / 100.0f);

    // Clamping the index to be within the valid range
    index = std::min(index, n_elements - 1);

    // Useing std::nth_element to find the k-th smallest element without a full sort.
    std::nth_element(abs_values.begin(), abs_values.begin() + index, abs_values.end());

    return abs_values[index];
}

float calculate_min_mse_amax(const float * data, size_t n_elements, int num_steps) {
    if (n_elements == 0) {
        return 0.0f;
    }

    // Determining search range for amax
    float abs_max_val = 0.0f;
    for (size_t i = 0; i < n_elements; ++i) {
        abs_max_val = std::max(abs_max_val, std::abs(data[i]));
    }
    if (abs_max_val == 0.0f) {
        return 0.0f;
    }
    // Using a high percentile as the lower bound to narrow the search range
    float percentile_max_val = calculate_percentile_amax(data, n_elements, 99.9f);

    float search_min = percentile_max_val;
    float search_max = abs_max_val;

    if (search_min >= search_max) {
        return search_max;
    }

    // Iteratively searching for the best amax
    float best_amax = search_max;
    double min_mse = std::numeric_limits<double>::max();
    const float step_size = (search_max - search_min) / num_steps;

    for (int i = 0; i < num_steps; ++i) {
        const float current_amax = search_min + (float)i * step_size;
        const float current_scale = current_amax / 7.0f;
        if (current_scale < 1e-9f) continue;
        const float iscale = 1.0f / current_scale;

        // Calculating MSE for the current amax candidate
        double current_mse = 0.0;
        #pragma omp parallel for reduction(+:current_mse)
        for (size_t j = 0; j < n_elements; ++j) {
            const float original_val = data[j];
            
            // Quantizing
            const float quantized_f = original_val * iscale;
            const int8_t quantized_i = std::max((int8_t)-7, std::min((int8_t)7, (int8_t)roundf(quantized_f)));
            
            // De-quantizing
            const float dequantized_val = (float)quantized_i * current_scale;
            
            // Accumulating error
            const double diff = (double)original_val - (double)dequantized_val;
            current_mse += diff * diff;
        }

        if (current_mse < min_mse) {
            min_mse = current_mse;
            best_amax = current_amax;
        }
    }
    return best_amax;
}

float calculate_entropy_amax(const float* data, size_t n_elements, int num_bins, int num_steps) {
    if (n_elements == 0) {
        return 0.0f;
    }

    // Defining search range and creating reference distribution P
    float abs_max_val = 0.0f;
    for (size_t i = 0; i < n_elements; ++i) {
        abs_max_val = std::max(abs_max_val, std::abs(data[i]));
    }
    if (abs_max_val == 0.0f) {
        return 0.0f;
    }

    std::vector<double> p_dist(num_bins, 0.0);
    const double bin_width = (double)abs_max_val * 2.0 / num_bins;
    for (size_t i = 0; i < n_elements; ++i) {
        int bin_index = static_cast<int>(((data[i] + abs_max_val) / bin_width));
        bin_index = std::max(0, std::min(num_bins - 1, bin_index));
        p_dist[bin_index]++;
    }

    // Normalizing histogram to a probability distribution, adding epsilon for stability
    const double epsilon = 1e-9;
    for (int i = 0; i < num_bins; ++i) {
        p_dist[i] = (p_dist[i] / n_elements) + epsilon;
    }

    // Iteratively searching for the best amax by minimizing KL-divergence
    float best_amax = abs_max_val;
    double min_kl_div = std::numeric_limits<double>::max();
    
    // Narrowing the search range to avoid wasting time on obviously bad values
    const float search_min = calculate_percentile_amax(data, n_elements, 99.5f);
    const float search_max = abs_max_val;
    const float step_size = (search_max - search_min) / num_steps;

    for (int i = 0; i < num_steps; ++i) {
        const float current_amax = search_min + (float)i * step_size;
        const float current_scale = current_amax / 7.0f;
        if (current_scale < 1e-9f) continue;
        const float iscale = 1.0f / current_scale;

        // Creating quantized distribution Q based on P
        std::vector<double> q_dist(num_bins, 0.0);
        for (int bin_idx = 0; bin_idx < num_bins; ++bin_idx) {
            const float original_val = -abs_max_val + (bin_idx + 0.5f) * bin_width;

            // Quantizing-dequantizing the value from the center of the bin
            const float quantized_f = original_val * iscale;
            const int8_t quantized_i = std::max((int8_t)-7, std::min((int8_t)7, (int8_t)roundf(quantized_f)));
            const float dequantized_val = (float)quantized_i * current_scale;

            // Finding which bin the de-quantized value falls into
            int new_bin_idx = static_cast<int>(((dequantized_val + abs_max_val) / bin_width));
            new_bin_idx = std::max(0, std::min(num_bins - 1, new_bin_idx));

            // Transfering the "probability mass" from the old bin to the new one
            q_dist[new_bin_idx] += p_dist[bin_idx];
        }

        // Normalizing Q
        for (int bin_idx = 0; bin_idx < num_bins; ++bin_idx) {
            q_dist[bin_idx] += epsilon;
        }

        // Calculating KL-divergence
        double current_kl_div = 0.0;
        for (int bin_idx = 0; bin_idx < num_bins; ++bin_idx) {
            if (p_dist[bin_idx] > epsilon * 1.1) {
                current_kl_div += p_dist[bin_idx] * log(p_dist[bin_idx] / q_dist[bin_idx]);
            }
        }

        if (current_kl_div < min_kl_div) {
            min_kl_div = current_kl_div;
            best_amax = current_amax;
        }
    }
    return best_amax;
}


// --- Hadamard Transform Implementations ---

// Helper to check if a number is a power of two
static bool is_power_of_two(int n) {
    return (n > 0) && ((n & (n - 1)) == 0);
}

// Iterative Fast Walsh-Hadamard Transform (in-place)
static void fwht_iterative(float* data, int size) {
#ifdef __ARM_NEON
    if (size >= 4) {
        // stage h=1: pairwise butterflies inside each 4-lane vector.
        // [a,b,c,d] -> [a+b, a-b, c+d, c-d]: even lanes from v+rev,
        // odd lanes from v-rev (vtrn1 interleaves them back).
        for (int i = 0; i < size; i += 4) {
            float32x4_t v   = vld1q_f32(data + i);
            float32x4_t sw  = vrev64q_f32(v);
            float32x4_t sum = vaddq_f32(v, sw);
            float32x4_t dif = vsubq_f32(v, sw);
            vst1q_f32(data + i, vtrn1q_f32(sum, dif));
        }
        // stage h=2: butterflies between the two lane pairs of one vector
        for (int i = 0; i < size; i += 4) {
            float32x4_t v = vld1q_f32(data + i);
            float32x2_t lo = vget_low_f32(v), hi = vget_high_f32(v);
            vst1_f32(data + i,     vadd_f32(lo, hi));
            vst1_f32(data + i + 2, vsub_f32(lo, hi));
        }
        // stages h>=4: contiguous 4-wide butterflies
        for (int h = 4; h < size; h <<= 1) {
            for (int i = 0; i < size; i += h * 2) {
                for (int j = i; j < i + h; j += 4) {
                    float32x4_t x = vld1q_f32(data + j);
                    float32x4_t y = vld1q_f32(data + j + h);
                    vst1q_f32(data + j,     vaddq_f32(x, y));
                    vst1q_f32(data + j + h, vsubq_f32(x, y));
                }
            }
        }
        return;
    }
#endif
    for (int h = 1; h < size; h <<= 1) {
        for (int i = 0; i < size; i += h * 2) {
            for (int j = i; j < i + h; ++j) {
                float x = data[j];
                float y = data[j + h];
                data[j] = x + y;
                data[j + h] = x - y;
            }
        }
    }
}

int next_power_of_two(int n) {
    if (n == 0) return 1;
    n--;
    n |= n >> 1;
    n |= n >> 2;
    n |= n >> 4;
    n |= n >> 8;
    n |= n >> 16;
    n++;
    return n;
}

// RKNPU_HADAMARD_BLOCK semantics (read once: packed buffers, matmul
// contexts and the dequant divisor depend on it — no runtime changes):
//   1 (default) pure block-diagonal: block = largest pow2 divisor of K,
//               zero padding — fastest and smallest (E4B: tg +28%, NPU
//               memory -29%) at a measured W4A4 quality cost (wikitext
//               PPL 228.9 vs 198.4 legacy on an already heavily degraded
//               capacity mode; W8A8 reference 37.8)
//   0           legacy: pad K to the next power of two, one full FWHT —
//               the pre-2026-08-16 behavior, bit-identical
//   <pow2 n>    minimum block n, K padded to the next block multiple.
//               Measured WORSE than both (n=1024: PPL 280.4, tg 5.13 —
//               the half-empty pad block hurts more than small blocks
//               do); kept for experiments only.
// See RKNPU2-decode-research.md #3b for all measurements.
static int hadamard_min_block() {
    static const int min_block = []() {
        const char* env = std::getenv("RKNPU_HADAMARD_BLOCK");
        int v = env ? std::atoi(env) : 1;
        if (v < 0) v = 0;
        if (v > 1 && (v & (v - 1)) != 0) v = next_power_of_two(v);
        return v;
    }();
    return min_block;
}

int hadamard_block_len(int K) {
    const int mb = hadamard_min_block();
    if (mb == 0) return next_power_of_two(K);   // legacy
    // 1 = natural: the largest power of two dividing K, so the transform
    // covers as many lanes as possible with no padding.
    const int divisor = K & (-K);
    if (mb == 1) return divisor;
    // explicit block size. Smaller than the natural divisor still divides
    // K exactly (any smaller power of two does), so it costs fewer FWHT
    // passes with no padding — the speed/quality dial. Larger pads K up to
    // a block multiple. Never exceed next_pow2(K): a K=256 row must not
    // inflate to a 1024 block.
    return std::min(mb, next_power_of_two(K));
}

int hadamard_k_op(int K) {
    const int mb = hadamard_min_block();
    if (mb == 0) return next_power_of_two(K);   // legacy
    const int block = hadamard_block_len(K);
    return ((K + block - 1) / block) * block;   // next multiple of block
}

bool per_channel_b_scales() {
    static const bool per_channel = []() {
        const char* env = std::getenv("RKNPU_PER_CHANNEL");
        return !(env != nullptr && std::atoi(env) == 0);
    }();
    return per_channel;
}

static float clip_from_env(const char* name, float dflt) {
    const char* env = std::getenv(name);
    if (env == nullptr) return dflt;
    float v = std::strtof(env, nullptr);
    if (!(v > 0.0f) || v > 1.0f) return dflt;   // junk or out of range
    return v;
}

// Defaults from a 32-chunk wikitext sweep on two models (E4B B-curve is a
// smooth U with its minimum at 0.93; Qwen agrees 0.93 > 0.95). Set both to
// 1.0 for unclipped amax scales. See RKNPU2-decode-research.md #3d.
float a_clip_factor() {
    static const float clip = clip_from_env("RKNPU_A_CLIP", 0.9f);
    return clip;
}

float b_clip_factor() {
    static const float clip = clip_from_env("RKNPU_B_CLIP", 0.93f);
    return clip;
}


void hadamard_transform(float* dst, const float* src, int K, int padded_size) {
    // padded_size is hadamard_k_op(K): a multiple of the block length,
    // which is a power of two. Legacy mode degenerates to one full-length
    // transform (block == padded_size == next_pow2(K)); power-of-two K is
    // bit-identical in every mode. dst must hold padded_size elements and
    // never aliases src at the call sites.
    const int block = hadamard_block_len(K);

    memcpy(dst, src, K * sizeof(float));
    if (padded_size > K) {
        memset(dst + K, 0, (padded_size - K) * sizeof(float));
    }
    for (int off = 0; off < padded_size; off += block) {
        fwht_iterative(dst + off, block);
    }
}

} // namespace rknpu2_calibration