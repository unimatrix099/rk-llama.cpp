#pragma once

// Native A/C layout tiling primitives for the RKNPU2 backend.
//
// The RK3588 matmul path is ~80x faster for INT4 when the A and C matrices
// use RKNN_MM_LAYOUT_NATIVE (see docs/backend/RKNPU2-int4-research.md).
// Native layout, as reported by rknn_matmul_io_attr dims, is a 3D tiling
//   A: [K / subK, M, subK]   (subK = 32 for int4, 16 for int8)
//   C: [N / subN, M, subN]   (subN =  8 for int4/int16 out, 4 for int8/int32)
// i.e. contiguous "cells" of `sub` elements, interleaved by row index m with
// stride M (the M the matmul context was created with).
//
// These helpers are pure byte movers: a cell is `cell_bytes` contiguous
// bytes in both layouts, so element packing inside a cell (e.g. two int4
// nibbles per byte, low nibble = even element, matching
// rknpu2_quantization::quantize_fp32_to_int4_packed) is preserved verbatim.
//
// No NPU or ggml dependencies; unit-testable on any host.

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Geometry of one native-layout matrix, parsed from io_attr dims.
typedef struct {
    int32_t outer;      // dims[0]: number of cells per row (K/subK or N/subN)
    int32_t m_stride;   // dims[1]: row interleave stride (context M)
    int32_t sub;        // dims[2]: elements per cell
} rknpu2_native_geom;

// Parse a [outer, m, sub] dims triple as reported by rknn_matmul_io_attr.
// Returns 0 on success; -1 if n_dims != 3 or any dim is not positive.
int rknpu2_native_geom_from_dims(const int32_t * dims, uint32_t n_dims,
                                 rknpu2_native_geom * out);

// Scatter one row-major row (row_cells * cell_bytes contiguous bytes) into a
// native-layout buffer: cell t of row m lands at (t * m_stride + m) * cell_bytes.
void rknpu2_native_scatter_row(uint8_t * dst, const uint8_t * src_row,
                               int32_t m, int32_t m_stride,
                               int32_t row_cells, int32_t cell_bytes);

// Gather one row back out of a native-layout buffer (exact inverse).
void rknpu2_native_gather_row(uint8_t * dst_row, const uint8_t * src,
                              int32_t m, int32_t m_stride,
                              int32_t row_cells, int32_t cell_bytes);

#ifdef __cplusplus
}
#endif
