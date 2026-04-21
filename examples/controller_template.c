/* PySaacSim controller template.
 *
 * Paste the body of your firmware Robot()'s while(1) loop into robot_tick().
 * Inputs are already set in globals before each call:
 *     DistanceRaw    -- right IR, mm (after firmware /2)
 *     L_DistanceRaw  -- left  IR, mm
 *     Distance2      -- right TFLuna3, mm
 *     L_Distance2    -- left  TFLuna2, mm
 * Output via CAN_SetMotors(dutyL, dutyR, steeringAngle).
 * OS_bWait is a no-op; remove or keep the calls as you like.
 *
 * Example below is the full wall-follower from
 * lab-7-tweinstein-1/RTOS_SensorBoard/RTOS_SensorBoard.c adapted to the
 * tick-based driver. Delete and paste your own code to try variations.
 */

#include <stdint.h>
#include "hal.h"

/* ---- PD constants (verbatim from firmware) -------------------------------*/
#define angle_ref 5     /* degrees "0"  */
#define dist_ref  130   /* mm           */
#define kp_d 1
#define kd_d 4
#define kp_a 5
#define kd_a 2

/* ---- Arctan LUT (verbatim RTOS_SensorBoard.c:91-101) --------------------*/
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

/* ---- Cosine LUT (verbatim RTOS_SensorBoard.c:123-147) -------------------*/
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

/* ---- Lifecycle ---------------------------------------------------------- */
void robot_init(void) {
    prevError = 0;
    prevE_A   = 0;
    prevTime  = 0;
    elapsed   = 0;
    DataLost  = 0;
    FilterWork = 0;
    Running   = 1;
}

/* ---- One PD iteration (was Robot()'s while(1) body) --------------------- */
void robot_tick(void) {
    elapsed  = OS_MsTime() - prevTime;
    prevTime = OS_MsTime();

    /* The firmware loop averages 8 samples of Distance2/L_Distance2. Because
     * Python writes these globals once per tick, the 8-sample average is the
     * same as the single value -- keep the loop so the pasted code is
     * unchanged, but it's a no-op in the sim. */
    uint32_t d2 = 0, ld2 = 0;
    for (uint8_t i = 0; i < 8; i++) {
        OS_bWait(&TFLuna3Ready);
        OS_bWait(&TFLuna2Ready);
        d2  += Distance2;
        ld2 += L_Distance2;
    }
    d2  >>= 3;
    ld2 >>= 3;

    uint32_t d_ir  = DistanceRaw;
    uint32_t ld_ir = L_DistanceRaw;

    int32_t angle   = arctan(((int32_t)(d_ir*1414)  - (int32_t)(d2*1000)) /(int32_t)(224+d2))  - angle_ref;
    int32_t L_angle = arctan(((int32_t)(ld_ir*1414) - (int32_t)(ld2*1000))/(int32_t)(224+ld2));

    int32_t realDist   = (d_ir  * cosine(angle))   / 1000;
    int32_t L_realDist = (ld_ir * cosine(L_angle)) / 1000;
    int32_t e_d = realDist - L_realDist - dist_ref;

    int32_t intend_angle = ((kp_d * e_d) / 10) + ((kd_d * (e_d - prevError)) / 10);
    prevError = e_d;

    int32_t e_a = intend_angle - angle;
    int32_t steeringAngle = ((e_a * kp_a) / 10) + (((e_a - prevE_A) * kd_a) / 10);
    prevE_A = e_a;

    /* Servo PWM count: firmware's Set_Servo maps (-30..+30 deg) -> 2000..4000.
     * Here we pass the raw degree value; the sim's MotorCommand translator in
     * c_bridge.py treats g_servo as a servo PWM count, so convert here. */
    int32_t servo_count = 3000 + (steeringAngle * 1000 / 30);
    if (servo_count < 2000) servo_count = 2000;
    if (servo_count > 4000) servo_count = 4000;

    CAN_SetMotors(9000, 9000, (int16_t)servo_count);
}
