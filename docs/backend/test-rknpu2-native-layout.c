// Unit tests for rknpu2-native-layout.{h,c} — pure CPU, no NPU required.
//
// Build & run (from docs/backend):
//   make -f Makefile.rknpu2-tools check
//
// Written test-first (TDD): these tests define the tiling contract that the
// hardware equivalence test (test-rknpu2-native-equivalence.c) then validates
// against the real runtime.

#include "rknpu2-native-layout.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int g_failures = 0;
static int g_checks = 0;

#define CHECK(cond, ...) do { \
    ++g_checks; \
    if (!(cond)) { \
        ++g_failures; \
        printf("FAIL %s:%d  ", __FILE__, __LINE__); \
        printf(__VA_ARGS__); \
        printf("\n"); \
    } \
} while (0)

// Deterministic PRNG so failures reproduce.
static uint32_t g_seed = 0x12345678u;
static uint32_t prng(void) {
    g_seed = g_seed * 1664525u + 1013904223u;
    return g_seed;
}

// ---------------------------------------------------------------- geometry

static void test_geom_parse_ok(void) {
    // Probed real dims on RK3588 (see RKNPU2-native-layout-plan.md):
    // INT4 A at M=32 K=2048: [64, 32, 32]; INT4 C at N=2048: [256, 32, 8]
    const int32_t a_dims[3] = {64, 32, 32};
    rknpu2_native_geom g;
    CHECK(rknpu2_native_geom_from_dims(a_dims, 3, &g) == 0, "A dims parse");
    CHECK(g.outer == 64 && g.m_stride == 32 && g.sub == 32,
          "A geom values %d %d %d", g.outer, g.m_stride, g.sub);

    const int32_t c_dims[3] = {256, 32, 8};
    CHECK(rknpu2_native_geom_from_dims(c_dims, 3, &g) == 0, "C dims parse");
    CHECK(g.outer == 256 && g.m_stride == 32 && g.sub == 8,
          "C geom values %d %d %d", g.outer, g.m_stride, g.sub);

    // M=1 degenerate case must parse too: [64, 1, 32]
    const int32_t m1_dims[3] = {64, 1, 32};
    CHECK(rknpu2_native_geom_from_dims(m1_dims, 3, &g) == 0, "M=1 parse");
    CHECK(g.m_stride == 1, "M=1 stride");
}

static void test_geom_parse_rejects(void) {
    rknpu2_native_geom g;
    const int32_t two[2] = {32, 2048};          // NORM layout reports 2 dims
    CHECK(rknpu2_native_geom_from_dims(two, 2, &g) == -1, "reject n_dims=2");
    const int32_t four[4] = {1, 2, 3, 4};
    CHECK(rknpu2_native_geom_from_dims(four, 4, &g) == -1, "reject n_dims=4");
    const int32_t zero[3] = {64, 0, 32};
    CHECK(rknpu2_native_geom_from_dims(zero, 3, &g) == -1, "reject zero dim[1]");
    const int32_t neg[3] = {64, -1, 32};
    CHECK(rknpu2_native_geom_from_dims(neg, 3, &g) == -1, "reject negative dim[1]");
    const int32_t zero0[3] = {0, 32, 32};
    CHECK(rknpu2_native_geom_from_dims(zero0, 3, &g) == -1, "reject zero dim[0]");
    const int32_t zero2[3] = {64, 32, 0};
    CHECK(rknpu2_native_geom_from_dims(zero2, 3, &g) == -1, "reject zero dim[2]");
    CHECK(rknpu2_native_geom_from_dims(NULL, 3, &g) == -1, "reject NULL dims");
    const int32_t ok[3] = {64, 32, 32};
    CHECK(rknpu2_native_geom_from_dims(ok, 3, NULL) == -1, "reject NULL out");
}

// ------------------------------------------------------------- known layout

// Hand-computed positions: M_stride=4, row m=1, 3 cells of 4 bytes.
// Cell t of row m must land at byte (t*4 + 1) * 4.
static void test_scatter_known_positions(void) {
    enum { CELLS = 3, CB = 4, MS = 4 };
    uint8_t src[CELLS * CB];
    for (int i = 0; i < CELLS * CB; ++i) src[i] = (uint8_t)(0xA0 + i);

    uint8_t dst[CELLS * MS * CB];
    memset(dst, 0xEE, sizeof(dst));

    rknpu2_native_scatter_row(dst, src, /*m=*/1, MS, CELLS, CB);

    for (int t = 0; t < CELLS; ++t) {
        const uint8_t * cell = dst + (t * MS + 1) * CB;
        for (int b = 0; b < CB; ++b) {
            CHECK(cell[b] == (uint8_t)(0xA0 + t * CB + b),
                  "cell %d byte %d = %02x", t, b, cell[b]);
        }
    }
    // Everything not belonging to row 1 must be untouched (0xEE).
    int untouched = 0;
    for (int t = 0; t < CELLS; ++t)
        for (int m = 0; m < MS; ++m)
            if (m != 1) {
                const uint8_t * cell = dst + (t * MS + m) * CB;
                for (int b = 0; b < CB; ++b) untouched += (cell[b] == 0xEE);
            }
    CHECK(untouched == CELLS * (MS - 1) * CB, "other rows untouched: %d", untouched);
}

// INT4 semantics: element k lives at row byte k/2, low nibble when k is even
// (contract of quantize_fp32_to_int4_packed). After scatter with
// cell_bytes = 16 (32 int4), element k must be found in cell k/32 at byte
// (k%32)/2, same nibble. Checked for k = 35 (odd, second cell).
static void test_scatter_int4_element_position(void) {
    enum { K = 64, CB = 16, CELLS = K / 32, MS = 2, M = 0 };
    uint8_t row[K / 2];
    memset(row, 0, sizeof(row));
    // element 35 = 0xB (high nibble of byte 17); element 34 = 0x5 (low nibble)
    row[17] = (uint8_t)(0xB << 4 | 0x5);

    uint8_t dst[CELLS * MS * CB];
    memset(dst, 0, sizeof(dst));
    rknpu2_native_scatter_row(dst, row, M, MS, CELLS, CB);

    // cell t=1 (k=35 -> 35/32=1), byte (35%32)/2 = 1
    const uint8_t * cell = dst + (1 * MS + M) * CB;
    CHECK((cell[1] >> 4) == 0xB, "elem 35 high nibble, got %02x", cell[1]);
    CHECK((cell[1] & 0xF) == 0x5, "elem 34 low nibble, got %02x", cell[1]);
}

// ----------------------------------------------------------- inverse + fuzz

static void test_roundtrip_fuzz(void) {
    // Sweep the geometries that matter: INT4 A (cb=16), INT8 A (cb=16),
    // INT4 C int16 (cb=16), INT8 C int32 (cb=16)... all 16B on RK3588 —
    // so also sweep odd cell sizes to keep the code honest.
    const int cell_bytes[] = {1, 2, 4, 8, 16};
    const int strides[]    = {1, 2, 5, 32};
    const int cells[]      = {1, 3, 64};

    for (unsigned ci = 0; ci < sizeof(cell_bytes)/sizeof(*cell_bytes); ++ci)
    for (unsigned si = 0; si < sizeof(strides)/sizeof(*strides); ++si)
    for (unsigned ti = 0; ti < sizeof(cells)/sizeof(*cells); ++ti) {
        const int cb = cell_bytes[ci], ms = strides[si], nc = cells[ti];
        const int row_bytes = nc * cb;
        const size_t native_bytes = (size_t)nc * ms * cb;

        uint8_t * rows_in  = malloc((size_t)ms * row_bytes);
        uint8_t * rows_out = malloc((size_t)ms * row_bytes);
        uint8_t * native   = malloc(native_bytes);
        for (size_t i = 0; i < (size_t)ms * row_bytes; ++i)
            rows_in[i] = (uint8_t)prng();
        memset(native, 0xCC, native_bytes);

        // scatter every row, then gather every row: must round-trip
        for (int m = 0; m < ms; ++m)
            rknpu2_native_scatter_row(native, rows_in + (size_t)m * row_bytes,
                                      m, ms, nc, cb);
        for (int m = 0; m < ms; ++m)
            rknpu2_native_gather_row(rows_out + (size_t)m * row_bytes, native,
                                     m, ms, nc, cb);

        CHECK(memcmp(rows_in, rows_out, (size_t)ms * row_bytes) == 0,
              "roundtrip cb=%d ms=%d cells=%d", cb, ms, nc);

        // With all rows written, the native buffer is fully determined:
        // cell (t, m) must equal cell t of input row m. Exact comparison —
        // no byte may be left over from the 0xCC prefill.
        uint8_t * expect = malloc(native_bytes);
        for (int t = 0; t < nc; ++t)
            for (int m = 0; m < ms; ++m)
                memcpy(expect + ((size_t)t * ms + m) * cb,
                       rows_in + (size_t)m * row_bytes + (size_t)t * cb,
                       (size_t)cb);
        CHECK(memcmp(native, expect, native_bytes) == 0,
              "native buffer exact cb=%d ms=%d cells=%d", cb, ms, nc);
        free(expect);

        free(rows_in); free(rows_out); free(native);
    }
}

// Padding rows (m_stride > rows actually used) must stay untouched — the
// backend creates contexts at M_op >= M and only writes real rows.
static void test_padding_rows_untouched(void) {
    enum { CB = 16, MS = 8, USED = 5, CELLS = 4 };
    uint8_t native[CELLS * MS * CB];
    memset(native, 0x77, sizeof(native));

    uint8_t row[CELLS * CB];
    memset(row, 0x11, sizeof(row));
    for (int m = 0; m < USED; ++m)
        rknpu2_native_scatter_row(native, row, m, MS, CELLS, CB);

    for (int t = 0; t < CELLS; ++t)
        for (int m = USED; m < MS; ++m) {
            const uint8_t * cell = native + ((size_t)t * MS + m) * CB;
            for (int b = 0; b < CB; ++b)
                CHECK(cell[b] == 0x77, "pad row %d cell %d", m, t);
        }
}

int main(void) {
    test_geom_parse_ok();
    test_geom_parse_rejects();
    test_scatter_known_positions();
    test_scatter_int4_element_position();
    test_roundtrip_fuzz();
    test_padding_rows_untouched();

    printf("%s: %d checks, %d failures\n",
           g_failures ? "FAIL" : "PASS", g_checks, g_failures);
    return g_failures ? 1 : 0;
}
