#pragma once

#include <vector>
#include <cstddef>

/**
 * @brief Provides algorithms for tensor calibration and transformation.
 *
 * This namespace contains functions for finding optimal quantization parameters (amax)
 * using various statistical methods, as well as mathematical transformations like
 * the Hadamard transform used in advanced quantization schemes.
 */
namespace rknpu2_calibration {

// --- Calibration Implementations ---

/**
 * @brief Finds the absolute maximum value based on a given percentile.
 * @param data Pointer to the source float data.
 * @param n_elements The number of elements in the data array.
 * @param percentile The percentile to calculate (e.g., 99.9f).
 * @return The absolute value at the specified percentile.
 */
float calculate_percentile_amax(const float * data, size_t n_elements, float percentile);

/**
 * @brief Finds the optimal amax by iteratively minimizing Mean Squared Error (MSE).
 * @param data Pointer to the source float data.
 * @param n_elements The number of elements in the data array.
 * @param num_steps The number of steps to iterate through in the search space.
 * @return The amax value that results in the lowest quantization error.
 */
float calculate_min_mse_amax(const float * data, size_t n_elements, int num_steps = 128);

/**
 * @brief Finds the optimal amax by minimizing KL-Divergence between FP32 and INT4 distributions.
 * @param data Pointer to the source float data.
 * @param n_elements The number of elements in the data array.
 * @param num_bins The number of bins for the histogram distributions.
 * @param num_steps The number of steps to iterate through in the search space.
 * @return The amax value that minimizes the information loss.
 */
float calculate_entropy_amax(const float* data, size_t n_elements, int num_bins = 2048, int num_steps = 128);


// --- Hadamard Transform Implementations ---

/**
 * @brief Applies a Fast Walsh-Hadamard Transform.
 *
 * This version is optimized for scenarios where padding is always expected. It takes separate
 * source and destination buffers and avoids extra allocations by using a thread-local buffer.
 *
 * @param dst Pointer to the destination buffer. Must be of size `padded_size`.
 * @param src Pointer to the source data buffer of size `K`.
 * @param K The original number of elements in the source data.
 * @param padded_size The target size for the transform (must be a power of two >= K). The full result is written to dst.
 */
void hadamard_transform(float* dst, const float* src, int K, int padded_size);

/**
 * @brief Calculates the next power of two for a given integer.
 * @param n The input integer.
 * @return The smallest power of two that is greater than or equal to n.
 */
int next_power_of_two(int n);

/**
 * @brief The K the Hadamard pipelines operate at for a weight row of K elements.
 *
 * Block-diagonal mode (default): K itself — the FWHT is applied per
 * power-of-two block (see hadamard_block_len), so no zero-padding is stored,
 * uploaded, or read back. Legacy mode (RKNPU_HADAMARD_BLOCK=0): the next
 * power of two, zero-padding the row for one full-length transform — on
 * non-power-of-two models (Gemma-4 E4B: K=2560/10240/10752) that inflates
 * every weight read by ~1.5-1.6x. The mode is read once from the
 * environment; it must not change at runtime (packed buffers and matmul
 * contexts depend on it).
 */
int hadamard_k_op(int K);

/**
 * @brief The FWHT block length used for rows of K elements.
 *
 * Block-diagonal mode: the largest power of two dividing K (K & -K) — for
 * power-of-two K this is K, making the transform identical to the legacy
 * full-length one. Also the dequantization divisor: H*H^T = B*I per block.
 * Legacy mode: next_power_of_two(K).
 */
int hadamard_block_len(int K);

/**
 * @brief Whether INT4 weights get one scale per output channel (default)
 * instead of one per matmul segment.
 *
 * The channel scale factors out of the hardware's K summation and is
 * applied in the existing C dequant pass, so the finer granularity costs
 * no NPU work. Per-channel scales use a plain amax (no entropy search —
 * also removes the minutes-long W4A4 calibration at load). Legacy
 * per-segment entropy scales behind RKNPU_PER_CHANNEL=0. Read once; the
 * packed weights and scale-grid layout depend on it.
 */
bool per_channel_b_scales();

} // namespace rknpu2_calibration