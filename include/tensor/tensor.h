#ifndef TENSOR_H
#define TENSOR_H

#include "inttypes.h"
#include "stdio.h"
#include "stdlib.h"
#include "math.h"
#include "assert.h"
#include "string.h"
#include "omp.h"

//My includes :
#include "debug.h"


//<-----    fan_in is :     size_kernel * size_kernel * n_channels        ------>
#define         HE_LIMIT(fan_in)            (sqrt(6.0 / (double)(fan_in)))

#define         MAX(A,B)                    (((A)>=(B)) ? (A) : (B) )
#define         MAX4(A,B,C,D)               MAX((MAX((A),(B))),MAX((C),(D)))

#define         MIN(A,B)                    (((A)<(B)) ? (A) : (B) )

//Defines the way to Initiates a tensor4
typedef enum{
    ZEROS,          //All values to zero
    UNIFORM,        //Uniform initalization, use this for a Kernels Tensor
    NOFILL,         //When we need to allocate only
} distribution_t;

//Tensor 4 used for : Kernels representation (All)
typedef struct{
    size_t col;
    size_t row;
    size_t nmap;

    union {
        size_t nfilter;
        size_t nbatch;
    };
} tensor4_dim_t;

//----------------------------------//             
//---           TENSOR 4         ---//
//----------------------------------//
typedef struct{
    size_t flatten_size;
    union {
        tensor4_dim_t dim;
        struct{ size_t col; size_t row; size_t nmap;  union {size_t nbatch; size_t nfilter;};};       //<---For a Kernel or parameter tensor, nbatch actually represents the number of filter instead
        size_t shape[4];
    };
    size_t strides[4];
    float *datas;
} tensor4_t;

//----------------------------------//             
//---           TENSOR 2         ---//
//----------------------------------//
typedef struct {
    tensor4_t; // membre sans nom
} tensor2_t;


//----------------------------------//            
//---       init and free        ---//
//----------------------------------//

//** Col, row, n_fmap, n_filter */
//Version avec la structure aussi ?
tensor4_t* init_tensor4(size_t d0, size_t d1, size_t d2, size_t d3, distribution_t type);
void free_tensor4(tensor4_t **t);


//----------------------------------//            
//---           Access           ---//
//----------------------------------//

//Access to a given index 
static inline size_t get_t4_idx(const tensor4_t *t,size_t idx0, size_t idx1, size_t idx2, size_t idx3){
return idx0*t->strides[0] + idx1*t->strides[1] + idx2*t->strides[2] + idx3*t->strides[3];} 

//Access to a given val index based
static inline float get_t4_val(const tensor4_t *t, size_t idx0, size_t idx1, size_t idx2, size_t idx3){
return t->datas[get_t4_idx(t,idx0,idx1,idx2,idx3)];}

//Set a given val
static inline void set_t4_val(const tensor4_t *t, float val, size_t idx0, size_t idx1, size_t idx2, size_t idx3){
t->datas[get_t4_idx(t,idx0,idx1,idx2,idx3)] = val;}


//Tensor dim operation
tensor4_dim_t get_t4_dim(const tensor4_t *T);

//Print tensordim
void print_t4_dim(const tensor4_dim_t *tdim);


//----------------------------------//            
//---           Display          ---//
//----------------------------------//

void print_tensor4_shape(const tensor4_t *T);
void print_tensor4_data(const tensor4_t *T);
void print_tensor4_mask(const uint8_t *mask, const tensor4_t *A);

//Compare
uint8_t compareTensor4(const tensor4_t *T1, const tensor4_t *T2);

//Resets tensor
void reset_t4(tensor4_t *T);

//well ok
void checkT2(const tensor2_t *T);



#endif