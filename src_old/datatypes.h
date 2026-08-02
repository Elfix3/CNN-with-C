#ifndef DTYPES_H
#define DTYPES_H

#include "stdio.h"
#include "stdlib.h"
#include "assert.h"
#include "mem.h"
#include "float.h"
#include "math.h"
#include "omp.h"


//<-----    fan_in is :     size_kernel * size_kernel * n_channels        ------>
#define HE_LIMIT(fan_in)    (sqrt(6.0 / (double)(fan_in)))

typedef enum{
    ZEROS,
    UNIFORM,
} init_type_t;

typedef enum{
    VALID,
    SAME,
} padding_type_t;



//Inits random float
void init_floats(float *fp, size_t n_float, init_type_t type);



//###########################//
//####      Array        ####//
//###########################//


typedef struct{
    size_t rows;
    size_t cols;
    float *datas;
} Array;

//Inits with random float
Array* init_array(size_t rows, size_t cols, init_type_t type, size_t fan_in);

//Frees array from a reference to a pointer;
void free_array(Array **a);

//Print function
void print_array(const Array *a);

float get_val(Array *a ,size_t i, size_t j);

//Applies a configuration to an existing Array
void set_array(Array *a, init_type_t type, size_t fan_in); // not used atm

Array *padded(Array *a, size_t n_padding);

//###########################//
//####    Array OPS      ####//
//###########################//

//Convolution
Array *conv(Array *fmap, Array *kernel, padding_type_t padding);


//Add scalar
void add_scalar(Array *output_map, float b);

//Cumulates (Source is released during process)
void cumulate(Array *destination, Array **source); //frees Source

//ReLU
void ReLU(Array *a);

//MaxPooling
void MaxPooling(Array **a);


//###########################//
//####     END ARRAY     ####//
//###########################//





//###########################//
//####    Array of Array ####//
//###########################//

typedef struct{
    size_t n_array;
    Array **a;
}   Array_Arr;

//Usual methods
Array_Arr *init_array_arr(size_t n_array, size_t rows, size_t cols);
void free_array_arr(Array_Arr **aa);
void print_array_arr(const Array_Arr* aa);

//###########################//
//#### END ARRAY OF ARRA ####//
//###########################//


//###########################//
//####    Array Bank     ####//
//###########################//

typedef struct{
    size_t n_arr_arr;
    Array_Arr ** aa;
} Array_Bank;

/**Parameters are in order : n_filter, n_feature_map, row , col */
Array_Bank *init_array_bank(size_t n_arr_arr, size_t n_array, size_t rows, size_t cols);
void free_array_bank(Array_Bank **ab);
void print_array_bank(const Array_Bank *ab);

//###########################//
//####  END ARRAY BANK   ####//
//###########################//

Array *convold(Array *fmap, Array *kernel, padding_type_t padding);


#endif