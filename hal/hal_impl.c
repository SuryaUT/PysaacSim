/* PySaacSim HAL implementation: defines the input/output globals and implements
 * every OS/UART/file-system stub Robot() might touch. Mirrors
 * simulator/firmware/src/hal_shim.c but driven by Python instead of JS. */

#include "hal.h"
#include <stdio.h>

/* ---- Sensor input globals (set by Python) ------------------------------- */
volatile uint32_t g_DistanceRaw    = 0;
volatile uint32_t g_L_DistanceRaw  = 0;
volatile uint32_t g_Distance2      = 0;
volatile uint32_t g_L_Distance2    = 0;
volatile uint32_t g_DistanceCenter = 0;

/* ---- CAN output globals (set by CAN_SetMotors, read by Python) ---------- */
volatile uint16_t g_duty_l = 0;
volatile uint16_t g_duty_r = 0;
volatile int16_t  g_servo  = 3000;   /* center */

volatile uint32_t g_t_ms       = 0;
volatile uint32_t Running      = 1;
volatile uint32_t DataLost     = 0;
volatile uint32_t FilterWork   = 0;
volatile uint32_t NumCreated   = 0;

int32_t  prevError = 0;
int32_t  prevE_A   = 0;
uint32_t prevTime  = 0;
uint32_t elapsed   = 0;

Sema4_t TFLuna3Ready = { 0 };
Sema4_t TFLuna2Ready = { 0 };

char FileName[8] = { 'r','o','b','o','t','0',0,0 };

/* ---- OS FIFO (stubs) ---------------------------------------------------- */
uint32_t OS_Fifo_Get(void)            { return g_Distance2; }
int      OS_Fifo_Put(uint32_t v)      { (void)v; return 1; }
void     OS_Fifo_Init(uint32_t s)     { (void)s; }

/* ---- OS time ------------------------------------------------------------ */
uint32_t OS_Time(void)                           { return g_t_ms * 80000; }  /* 12.5 ns ticks */
uint32_t OS_TimeDifference(uint32_t a, uint32_t b){ return b - a; }
uint32_t OS_MsTime(void)                         { return g_t_ms; }
void     OS_ClearMsTime(void)                    { /* Python resets g_t_ms directly */ }

/* ---- Threads / scheduler ----------------------------------------------- */
int  OS_AddThread(void (*task)(void), uint32_t s, uint32_t p)
{ (void)task; (void)s; (void)p; NumCreated++; return 1; }
void OS_Kill(void)                 { /* robot_tick returns normally */ }
void OS_Sleep(uint32_t ms)         { (void)ms; }
uint32_t OS_Id(void)               { return 1; }
void OS_InitSemaphore(Sema4_t *s, int32_t v) { if (s) s->Value = v; }
void OS_bWait(Sema4_t *s)          { (void)s; }  /* no-op: Python gates timing */
void OS_bSignal(Sema4_t *s)        { (void)s; }
void OS_MailBox_Init(void)         { }

/* ---- UART / LCD --------------------------------------------------------- */
void UART_OutString(const char *s)
{ if (s) { fputs(s, stdout); } }

void ST7735_Message(uint32_t d, uint32_t l, const char *msg, int32_t v)
{ (void)d; (void)l; if (msg) { printf("%s%ld\n", msg, (long)v); } }

/* ---- File-system stubs -------------------------------------------------- */
int eFile_Init(void)               { return 0; }
int eFile_Mount(void)              { return 0; }
int eFile_Create(const char *n)    { (void)n; return 0; }
int eFile_WOpen(const char *n)     { (void)n; return 0; }
int eFile_Write(char c)            { (void)c; return 0; }
int eFile_WriteString(const char *s){ (void)s; return 0; }
int eFile_WriteUDec(uint32_t n)    { (void)n; return 0; }
int eFile_WriteUFix2(uint32_t n)   { (void)n; return 0; }
int eFile_WClose(void)             { return 0; }
void StartFileDump(const char *n)  { (void)n; }
void EndFileDump(void)             { }
void Display(void)                 { }
void Jitter3_Init(void)            { }

/* ---- CAN ---------------------------------------------------------------- */
int CAN_SetMotors(uint16_t Duty_L, uint16_t Duty_R, int16_t SteeringAngle)
{
    g_duty_l = Duty_L;
    g_duty_r = Duty_R;
    g_servo  = SteeringAngle;
    return 1;
}

/* ---- Unused LPF init ---------------------------------------------------- */
void LPF_Init7(int32_t initial, int32_t size) { (void)initial; (void)size; }
