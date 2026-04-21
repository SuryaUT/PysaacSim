/* PySaacSim HAL — matches simulator/firmware/include/hal_shim.h in spirit but
 * designed for the PySaacSim tick-based driver. Python writes input globals
 * (IR + lidar distances), calls robot_tick(), then reads output globals
 * written by CAN_SetMotors.
 *
 * Macro aliases (DistanceRaw, L_DistanceRaw, Distance2, L_Distance2) let the
 * user paste their firmware Robot() body into robot_tick() with no rename. */
#ifndef PYSIM_HAL_H
#define PYSIM_HAL_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ---- Types -------------------------------------------------------------- */
typedef struct Sema4 { int32_t Value; } Sema4_t;

/* ---- Sensor input globals (Python writes before each robot_tick) -------- */
extern volatile uint32_t g_DistanceRaw;    /* right IR, mm (after firmware /2)  */
extern volatile uint32_t g_L_DistanceRaw;  /* left  IR, mm                       */
extern volatile uint32_t g_Distance2;      /* right TFLuna3, mm                  */
extern volatile uint32_t g_L_Distance2;    /* left  TFLuna2, mm                  */
extern volatile uint32_t g_DistanceCenter; /* center TFLuna2, mm (optional)      */

/* Firmware-style aliases: paste original Robot() body verbatim. */
#define DistanceRaw    g_DistanceRaw
#define L_DistanceRaw  g_L_DistanceRaw
#define Distance2      g_Distance2
#define L_Distance2    g_L_Distance2
#define Distance       g_DistanceRaw      /* DAS writes the right IR; matches firmware */

/* ---- CAN output globals (CAN_SetMotors writes, Python reads) ------------ */
extern volatile uint16_t g_duty_l, g_duty_r;
extern volatile int16_t  g_servo;

/* Simulated time — Python advances this between ticks. */
extern volatile uint32_t g_t_ms;

/* Runtime flags (unused by sim but referenced by firmware) */
extern volatile uint32_t Running;
extern volatile uint32_t DataLost;
extern volatile uint32_t FilterWork;
extern volatile uint32_t NumCreated;

/* PD state carried between ticks (firmware declared these at file scope in
 * RTOS_SensorBoard.c; we declare them here so robot_tick() can use them). */
extern int32_t prevError;
extern int32_t prevE_A;
extern uint32_t prevTime;
extern uint32_t elapsed;

/* TFLuna semaphores — no-op in sim, but referenced by Robot(). */
extern Sema4_t TFLuna3Ready;
extern Sema4_t TFLuna2Ready;

/* File dump glue — stubs. */
extern char FileName[8];

/* ---- OS primitives (all no-ops in sim) ---------------------------------- */
uint32_t OS_Fifo_Get(void);
int      OS_Fifo_Put(uint32_t data);
void     OS_Fifo_Init(uint32_t size);

uint32_t OS_Time(void);
uint32_t OS_TimeDifference(uint32_t a, uint32_t b);
uint32_t OS_MsTime(void);
void     OS_ClearMsTime(void);

int  OS_AddThread(void (*task)(void), uint32_t stackSize, uint32_t priority);
void OS_Kill(void);
void OS_Sleep(uint32_t ms);
uint32_t OS_Id(void);

void OS_InitSemaphore(Sema4_t *s, int32_t v);
void OS_bWait(Sema4_t *s);
void OS_bSignal(Sema4_t *s);
void OS_MailBox_Init(void);

/* ---- UART / LCD sinks (logs go to stdout) ------------------------------- */
void UART_OutString(const char *s);
void ST7735_Message(uint32_t d, uint32_t l, const char *msg, int32_t v);

/* ---- File-system stubs -------------------------------------------------- */
int eFile_Init(void);
int eFile_Mount(void);
int eFile_Create(const char *name);
int eFile_WOpen(const char *name);
int eFile_Write(char c);
int eFile_WriteString(const char *s);
int eFile_WriteUDec(uint32_t n);
int eFile_WriteUFix2(uint32_t n);
int eFile_WClose(void);
void StartFileDump(const char *name);
void EndFileDump(void);
void Display(void);
void Jitter3_Init(void);

/* Dummy IRQ primitives for pasted firmware code. */
#define __disable_irq() ((void)0)
#define __enable_irq()  ((void)0)

/* ---- CAN ---------------------------------------------------------------- */
int CAN_SetMotors(uint16_t Duty_L, uint16_t Duty_R, int16_t SteeringAngle);

/* ---- DAS / Producer pipeline helpers ------------------------------------ */
int32_t Filter(int32_t data);
int32_t Median5(int32_t newdata);
int32_t IRDistance_Convert(int32_t adcSample, uint32_t sensor);
void    LPF_Init7(int32_t initial, int32_t size);

/* ---- Entry points Python calls ------------------------------------------ */
/* User implements these in their C file. robot_init() is called once before
 * the first tick; robot_tick() does one iteration of the PD control loop. */
void robot_init(void);
void robot_tick(void);

#ifdef __cplusplus
}
#endif
#endif /* PYSIM_HAL_H */
