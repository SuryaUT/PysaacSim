/* Filter() and Median5() — copied verbatim from
 * simulator/firmware/src/filter.c (which copied lab-7-tweinstein-1/RTOS_Labs_common/LPF.c).
 * 60-Hz IIR notch (fs=1 kHz) plus a median-of-5-out-of-7 used by the TFLuna
 * producer. */

#include <stdint.h>
#include "hal.h"

/* 60-Hz notch, fs=1 kHz:
 *   y(n) = (256x(n) - 476x(n-1) + 256x(n-2) + 471y(n-1) - 251y(n-2))/256 */
int32_t Filter(int32_t data) {
    static int32_t x[6];
    static int32_t y[6];
    static uint32_t n = 3;
    n++;
    if (n == 6) n = 3;
    x[n] = x[n-3] = data;
    y[n] = (256*(x[n] + x[n-2]) - 476*x[n-1] + 471*y[n-1] - 251*y[n-2] + 128) / 256;
    y[n-3] = y[n];
    return y[n];
}

/* Median-of-5 over positions 0..4 of a 7-slot sliding window. */
int32_t mx7[7];
int32_t f7[7];
int32_t Median5(int32_t x) {
    int i, j;
    int32_t max;
    for (i = 3; i >= 0; i--) {
        mx7[i+1] = mx7[i];
        f7[i] = 1;
    }
    f7[5] = 1;
    mx7[0] = x;
    max = mx7[0]; j = 0;
    for (i = 1; i < 7; i++) {
        if (mx7[i] > max) { max = mx7[i]; j = i; }
    }
    f7[j] = 0;
    max = INT32_MIN;
    for (i = 0; i < 7; i++) {
        if ((mx7[i] > max) && f7[i]) { max = mx7[i]; j = i; }
    }
    f7[j] = 0;
    max = INT32_MIN;
    for (i = 0; i < 7; i++) {
        if ((mx7[i] > max) && f7[i]) { max = mx7[i]; }
    }
    return max;
}
