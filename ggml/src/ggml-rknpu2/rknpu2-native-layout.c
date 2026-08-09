#include "rknpu2-native-layout.h"

#include <string.h>
#include <stdint.h>

int rknpu2_native_geom_from_dims(const uint32_t * dims, uint32_t n_dims,
                                 rknpu2_native_geom * out) {
    if (dims == NULL || out == NULL || n_dims != 3) {
        return -1;
    }
    if (dims[0] == 0 || dims[0] > (uint32_t)INT32_MAX ||
        dims[1] == 0 || dims[1] > (uint32_t)INT32_MAX ||
        dims[2] == 0 || dims[2] > (uint32_t)INT32_MAX) {
        return -1;
    }
    out->outer    = (int32_t)dims[0];
    out->m_stride = (int32_t)dims[1];
    out->sub      = (int32_t)dims[2];
    return 0;
}

void rknpu2_native_scatter_row(uint8_t * dst, const uint8_t * src_row,
                               int32_t m, int32_t m_stride,
                               int32_t row_cells, int32_t cell_bytes) {
    for (int32_t t = 0; t < row_cells; ++t) {
        memcpy(dst + ((size_t)t * m_stride + m) * cell_bytes,
               src_row + (size_t)t * cell_bytes,
               (size_t)cell_bytes);
    }
}

void rknpu2_native_gather_row(uint8_t * dst_row, const uint8_t * src,
                              int32_t m, int32_t m_stride,
                              int32_t row_cells, int32_t cell_bytes) {
    for (int32_t t = 0; t < row_cells; ++t) {
        memcpy(dst_row + (size_t)t * cell_bytes,
               src + ((size_t)t * m_stride + m) * cell_bytes,
               (size_t)cell_bytes);
    }
}
