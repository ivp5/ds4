#include "../ds4_cdx3_reader.h"

#include <math.h>
#include <stdlib.h>
#include <stdio.h>

static float canary_input(uint32_t index) {
    return sinf((float)index * 0.0137f) + 0.25f * cosf((float)index * 0.071f);
}

static float dot_row(ds4_cdx3_file *file, ds4_cdx3_record *record, uint32_t row) {
    const uint32_t blocks_per_row = record->in_dim / 8u;
    float sum = 0.0f;
    float block[8];
    for (uint32_t block_col = 0; block_col < blocks_per_row; block_col++) {
        if (!ds4_cdx3_decode_block(file, record, row * blocks_per_row + block_col, block)) {
            fprintf(stderr, "decode failed at row=%u block_col=%u\n", row, block_col);
            exit(6);
        }
        const uint32_t base = block_col * 8u;
        for (uint32_t lane = 0; lane < 8u; lane++) sum += block[lane] * canary_input(base + lane);
    }
    return sum;
}

static float silu(float x) {
    return x / (1.0f + expf(-x));
}

int main(int argc, char **argv) {
    if (argc != 3) {
        fprintf(stderr, "usage: %s PACK.ds4cdx3[.inprogress] INDEX.cdx3i\n", argv[0]);
        return 2;
    }
    ds4_cdx3_file file;
    if (!ds4_cdx3_open(argv[1], argv[2], &file)) return 3;
    ds4_cdx3_print_summary(&file);

    ds4_cdx3_record record;
    if (!ds4_cdx3_get_record(&file, 0, 0, DS4_CDX3_KIND_GATE, &record)) {
        fprintf(stderr, "missing L0/E0/gate record\n");
        ds4_cdx3_close(&file);
        return 4;
    }
    float values[8];
    if (!ds4_cdx3_decode_block(&file, &record, 0, values)) {
        fprintf(stderr, "decode block failed\n");
        ds4_cdx3_close(&file);
        return 5;
    }
    printf("cdx3_canary ok L%u E%u kind=%u bits=%u code0=%u scale0=%.8g block0=[%.6g %.6g %.6g %.6g %.6g %.6g %.6g %.6g]\n",
           record.layer, record.expert, record.kind, record.bits,
           ds4_cdx3_get_code(&file, &record, 0),
           ds4_cdx3_get_scale(&file, &record, 0),
           values[0], values[1], values[2], values[3], values[4], values[5], values[6], values[7]);
    printf("cdx3_matvec_canary L0/E0/gate rows 0..3 = [%.9g %.9g %.9g %.9g]\n",
           dot_row(&file, &record, 0),
           dot_row(&file, &record, 1),
           dot_row(&file, &record, 2),
           dot_row(&file, &record, 3));
    ds4_cdx3_record up_record;
    ds4_cdx3_record down_record;
    if (!ds4_cdx3_get_record(&file, 0, 0, DS4_CDX3_KIND_UP, &up_record) ||
        !ds4_cdx3_get_record(&file, 0, 0, DS4_CDX3_KIND_DOWN, &down_record)) {
        fprintf(stderr, "missing L0/E0 up/down record\n");
        ds4_cdx3_close(&file);
        return 7;
    }
    if (record.out_dim != up_record.out_dim || down_record.in_dim != record.out_dim) {
        fprintf(stderr, "unexpected L0/E0 routed dims gate_out=%u up_out=%u down_in=%u\n",
                record.out_dim, up_record.out_dim, down_record.in_dim);
        ds4_cdx3_close(&file);
        return 8;
    }
    float *mid = calloc(record.out_dim, sizeof(float));
    if (!mid) {
        fprintf(stderr, "calloc mid failed\n");
        ds4_cdx3_close(&file);
        return 9;
    }
    for (uint32_t row = 0; row < record.out_dim; row++) {
        float gate_value = dot_row(&file, &record, row);
        float up_value = dot_row(&file, &up_record, row);
        if (gate_value > 10.0f) gate_value = 10.0f;
        if (up_value > 10.0f) up_value = 10.0f;
        if (up_value < -10.0f) up_value = -10.0f;
        mid[row] = silu(gate_value) * up_value;
    }
    printf("cdx3_swiglu_down_canary L0/E0/down rows 0..3 = [%.9g %.9g %.9g %.9g]\n",
           dot_row(&file, &down_record, 0),
           dot_row(&file, &down_record, 1),
           dot_row(&file, &down_record, 2),
           dot_row(&file, &down_record, 3));
    float down0 = 0.0f, down1 = 0.0f, down2 = 0.0f, down3 = 0.0f;
    const uint32_t blocks_per_down_row = down_record.in_dim / 8u;
    float down_block[8];
    for (uint32_t block_col = 0; block_col < blocks_per_down_row; block_col++) {
        const float *mid_block = mid + block_col * 8u;
        ds4_cdx3_decode_block(&file, &down_record, block_col, down_block);
        for (uint32_t lane = 0; lane < 8u; lane++) down0 += down_block[lane] * mid_block[lane];
        ds4_cdx3_decode_block(&file, &down_record, blocks_per_down_row + block_col, down_block);
        for (uint32_t lane = 0; lane < 8u; lane++) down1 += down_block[lane] * mid_block[lane];
        ds4_cdx3_decode_block(&file, &down_record, 2u * blocks_per_down_row + block_col, down_block);
        for (uint32_t lane = 0; lane < 8u; lane++) down2 += down_block[lane] * mid_block[lane];
        ds4_cdx3_decode_block(&file, &down_record, 3u * blocks_per_down_row + block_col, down_block);
        for (uint32_t lane = 0; lane < 8u; lane++) down3 += down_block[lane] * mid_block[lane];
    }
    printf("cdx3_full_chain_canary L0/E0/swiglu_down rows 0..3 = [%.9g %.9g %.9g %.9g]\n",
           down0, down1, down2, down3);
    free(mid);
    ds4_cdx3_close(&file);
    return 0;
}
