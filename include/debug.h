#ifndef DEBUG_H
#define DEBUG_H


#include "stdio.h"
#define DEBUG 1


#if DEBUG
    #define LOG(msg)                        printf("[%s] %s\n", __func__, msg)
    #define LOG_TENSOR(t)                   print_tensor4_data(t)
    #define LOG_MASK(mask, d0, d1, d2)      print_mask(mask, d0, d1, d2)
#else
    #define LOG(msg)
    #define LOG_TENSOR(t)
    #define LOG_MASK(mask, d0, d1, d2)
#endif

#endif //