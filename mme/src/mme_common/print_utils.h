#ifndef MME__PRINT_UTILS_H
#define MME__PRINT_UTILS_H

#include <mutex>

#define COLORED_PRINT_EN 1

#ifdef _WIN32
#include <windows.h>
#define COLOR_RED     12
#define COLOR_GREEN   10
#define COLOR_MAGENTA 13
#define COLOR_YELLOW  14
#define COLOR_CYAN    11
#else
#define COLOR_RED     "\x1b[31m"
#define COLOR_GREEN   "\x1b[32m"
#define COLOR_YELLOW  "\x1b[33m"
#define COLOR_BLUE    "\x1b[34m"
#define COLOR_MAGENTA "\x1b[35m"
#define COLOR_CYAN    "\x1b[36m"
#define COLOR_RESET   "\x1b[0m"
#endif

#ifdef COLORED_PRINT_EN
#ifdef _WIN32
#define coloredPrint(color, format, ...)                                                                               \
    {                                                                                                                  \
        HANDLE hConsole = GetStdHandle(STD_OUTPUT_HANDLE);                                                             \
        CONSOLE_SCREEN_BUFFER_INFO consoleInfo;                                                                        \
        WORD saved_attributes;                                                                                         \
        GetConsoleScreenBufferInfo(hConsole, &consoleInfo);                                                            \
        saved_attributes = consoleInfo.wAttributes;                                                                    \
        SetConsoleTextAttribute(hConsole, color);                                                                      \
        printf(format, __VA_ARGS__);                                                                                   \
        SetConsoleTextAttribute(hConsole, saved_attributes);                                                           \
    }
#else
#define coloredPrint(color, format, ...) printf(color format COLOR_RESET, ##__VA_ARGS__)
#endif

#else
#define coloredPrint(color, format, ...) printf(color format, ##__VA_ARGS__)
#endif

extern std::mutex printMutex;
#define atomicColoredPrint(color, format, ...)                                                                         \
    {                                                                                                                  \
        printMutex.lock();                                                                                             \
        coloredPrint(color, format, ##__VA_ARGS__);                                                                    \
        fflush(stdout);                                                                                                \
        printMutex.unlock();                                                                                           \
    }

#endif //MME__PRINT_UTILS_H
