/* PySaacSim port of the current firmware PD controller.
 * Source: lab-7-tweinstein-1/RTOS_SensorBoard/RTOS_SensorBoard.c, Robot().
 *
 * This is the VERBATIM math from the real robot (minus Model_* residual +
 * OS/file-dump plumbing). Load this in the Controller editor as the base
 * behavior for RL training — the policy then learns racing behavior
 * around this controller rather than from scratch.
 *
 * Inputs come in as firmware-style globals (see hal.h):
 *   DistanceRaw    = right IR, mm (already /2'd, so multiply by 2 to match firmware d_ir)
 *   L_DistanceRaw  = left  IR, mm (same)
 *   Distance2      = right TFLuna3, mm
 *   L_Distance2    = left  TFLuna2, mm
 *   g_DistanceCenter = front TFLuna1, mm (the firmware's FrontDist)
 * Output: CAN_SetMotors(duty_l, duty_r, servo_count).
 */

#include <stdint.h>
#include "hal.h"

/* ---- PD constants (verbatim from firmware) ------------------------------- */
#define angle_ref 5   /* degrees "0" — firmware bias on the right-side angle */
#define dist_ref  0   /* mm — target for (right_wall_dist - left_wall_dist) */
#define kp_d 1
#define kd_d 2
#define kp_a 5
#define kd_a 2

/* ---- Arctan LUT ---------------------------------------------------------- */
static const uint32_t TanTable[90] = {
       0,   17,   35,   52,   70,   87,  105,  123,  141,  158,
     176,  194,  213,  231,  249,  268,  287,  306,  325,  344,
     364,  384,  404,  424,  445,  466,  488,  510,  532,  554,
     577,  601,  625,  649,  675,  700,  727,  754,  781,  810,
     839,  869,  900,  933,  966, 1000, 1036, 1072, 1111, 1150,
    1192, 1235, 1280, 1327, 1376, 1428, 1483, 1540, 1600, 1664,
    1732, 1804, 1881, 1963, 2050, 2145, 2246, 2356, 2475, 2605,
    2747, 2904, 3078, 3271, 3487, 3732, 4011, 4331, 4705, 5145,
    5671, 6314, 7115, 8144, 9514,11430,14300,19081,28636,57290
};
static int32_t arctan(int32_t ratio_x1000) {
    uint32_t abs_ratio = (ratio_x1000 < 0) ? (uint32_t)(-ratio_x1000) : (uint32_t)ratio_x1000;
    uint32_t lo = 0, hi = 88;
    if (abs_ratio >= TanTable[89]) return (ratio_x1000 < 0) ? -89 : 89;
    while (lo + 1 < hi) {
        uint32_t mid = (lo + hi) / 2;
        if (TanTable[mid] <= abs_ratio) lo = mid;
        else hi = mid;
    }
    return (ratio_x1000 < 0) ? -(int32_t)lo : (int32_t)lo;
}

/* ---- Cosine LUT ---------------------------------------------------------- */
static const uint32_t CosTable[91] = {
    1000, 999, 999, 998, 997, 996, 994, 992, 990, 987,
     984, 981, 978, 974, 970, 965, 961, 956, 951, 945,
     939, 933, 927, 920, 913, 906, 898, 891, 882, 874,
     866, 857, 848, 838, 829, 819, 809, 798, 788, 777,
     766, 754, 743, 731, 719, 707, 694, 681, 669, 656,
     642, 629, 615, 601, 587, 573, 559, 544, 529, 515,
     500, 484, 469, 453, 438, 422, 406, 390, 374, 358,
     342, 325, 309, 292, 276, 258, 242, 224, 208, 190,
     173, 156, 139, 121, 104,  87,  69,  52,  35,  17,
       0
};
static int32_t cosine(int32_t angle_deg) {
    while (angle_deg < 0)    angle_deg += 360;
    while (angle_deg >= 360) angle_deg -= 360;
    if (angle_deg <= 90)  return  (int32_t)CosTable[angle_deg];
    if (angle_deg <= 180) return -(int32_t)CosTable[180 - angle_deg];
    if (angle_deg <= 270) return -(int32_t)CosTable[angle_deg - 180];
    return                        (int32_t)CosTable[360 - angle_deg];
}

/* ---- Lifecycle ----------------------------------------------------------- */
void robot_init(void) {
    prevError = 0;
    prevE_A   = 0;
    prevTime  = 0;
    elapsed   = 0;
    Running   = 1;
}

/* ---- One PD iteration (mirror of Robot()'s while(1) body) --------------- */
void robot_tick(void) {
    elapsed  = OS_MsTime() - prevTime;
    prevTime = OS_MsTime();

    /* Firmware averages 8 TFLuna samples per tick; in sim the globals change
     * once per tick so the average collapses to a single value. Kept as a
     * single read to stay readable. */
    uint32_t d2  = Distance2;
    uint32_t ld2 = L_Distance2;

    /* Firmware does Distance*2 / L_Distance*2 because IRDistance_Convert
     * returns 2× the true distance. PySaacSim's c_bridge already /2'd the value
     * into DistanceRaw, so *2 here recovers the firmware's "d_ir". */
    uint32_t d_ir  = DistanceRaw   * 2;
    uint32_t ld_ir = L_DistanceRaw * 2;

    /* IR out-of-range correction: if TFLuna sees open space and IR reads
     * much more than TFLuna, the IR is past its calibrated range. */
    if (d2  > 600 && d2  > d_ir  + 150) d_ir  = 305;
    if (ld2 > 600 && ld2 > ld_ir + 150) ld_ir = 305;

    int32_t angle   = arctan(((int32_t)(d_ir*1414)  - (int32_t)(d2*1000))  / (int32_t)(224 + d2))  - angle_ref;
    int32_t L_angle = arctan(((int32_t)(ld_ir*1414) - (int32_t)(ld2*1000)) / (int32_t)(224 + ld2));

    int32_t realDist   = (d_ir  * cosine(angle))   / 1000;
    int32_t L_realDist = (ld_ir * cosine(L_angle)) / 1000;
    int32_t e_d = realDist - L_realDist - dist_ref;

    int32_t intend_angle = ((kp_d * e_d) / 10) + ((kd_d * (e_d - prevError)) / 10);
    prevError = e_d;

    int32_t e_a = intend_angle - angle;
    int32_t steeringAngle = ((e_a * kp_a) / 10) + (((e_a - prevE_A) * kd_a) / 10);
    prevE_A = e_a;

    /* Front obstacle avoidance (firmware uses FrontDist; in sim that's the
     * center TFLuna -> g_DistanceCenter). */
    int32_t front = (int32_t)g_DistanceCenter;
    uint16_t throttle = 9000;
    if (front < 600) {
        throttle -= 2000;
        int32_t urgency = (600 - front) >> 4;
        steeringAngle = (ld2 > d2) ? -urgency : urgency; /* turn toward the side with more room */
    }

    if (steeringAngle < -30) steeringAngle = -30;
    else if (steeringAngle > 30) steeringAngle = 30;

    /* Differential steering: slow the inside wheel on sharp turns. */
    uint16_t throttle_l = throttle, throttle_r = throttle;
    if      (steeringAngle <= -15) throttle_r -= 2000;
    else if (steeringAngle >=  15) throttle_l -= 2000;

    /* CAN_SetMotors takes a servo PWM count (2000..4000), not raw degrees.
     * Firmware's motor board does this conversion; we do it here instead. */
    int32_t servo_count = 3000 + (steeringAngle * 1000 / 30);
    if (servo_count < 2000) servo_count = 2000;
    if (servo_count > 4000) servo_count = 4000;

    CAN_SetMotors(throttle_l, throttle_r, (int16_t)servo_count);
}
