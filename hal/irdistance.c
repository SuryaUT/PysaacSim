/* IRDistance_Convert — copied from simulator/firmware/src/irdistance.c.
 * Returns distance in mm for the Sharp GP2Y0A41SK0F as wired on Lab 7.
 * The sim's IR calibration (config/calibration.yaml, curve_k=12.79) produces
 * ADC counts such that this function returns 2×true_mm; firmware's /2 in DAS
 * then recovers true mm. */

#include <stdint.h>
#include "hal.h"

static const int32_t A[4]     = { 268130, 268130, 268130, 268130 };
static const int32_t B[4]     = { -159, -159, -159, -159 };
static const int32_t C[4]     = { 0, 0, 0, 0 };
static const int32_t IRmax[4] = { 494, 494, 494, 494 };

int32_t IRDistance_Convert(int32_t adcSample, uint32_t sensor) {
    if (adcSample < IRmax[sensor]) return 800;
    return A[sensor] / (adcSample + B[sensor]) + C[sensor];
}
