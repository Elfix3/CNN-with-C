#ifndef DEBUG_H
#define DEBUG_H


#include "stdio.h"

//Niveau des logs
#define DEBUG       1
#define INFO        1
#define WARNING     1
#define ERROR       1
#define VERBOSE     0

//Couleurs
#define RESET   "\033[0m"
#define RED     "\033[31m"
#define YELLOW  "\033[33m"
#define BLUE    "\033[34m"
#define CYAN    "\033[36m"
#define GREEN   "\033[32m"
#define GRAY    "\033[90m"


//Message générique de log
#define LOG(msg)                        printf("\n##### [%s] %s #####\n", __func__, msg)

//---> DEBUG
#if DEBUG
    #define LOG_DEBUG(fmt, ...) fprintf(stderr, GRAY "[DEBUG] %s:%d (%s): " fmt RESET "\n", \
                __FILE__, __LINE__, __func__, ##__VA_ARGS__)
#else
    #define LOG_DEBUG(fmt, ...) 
#endif

//---> INFO
#if INFO
    #define LOG_INFO(fmt, ...) fprintf(stdout, CYAN "[INFO] (in %s) : " fmt RESET "\n", \
                __func__, ##__VA_ARGS__)
#else
    #define LOG_INFO(fmt, ...)
#endif


//---> WARNING
#if DEBUG
    
#else

#endif


//---> ERROR
#if DEBUG
    
#else

#endif

//---> VERBOSE
#if DEBUG
    
#else

#endif


#endif //