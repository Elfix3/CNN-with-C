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

//Defines the padding type for convolutionnal operations
typedef enum{
    VALID,          //Shrinks the input dimension : size_input-size_kernel + 1 becomes the new output dimension : ex : 7*7 feature map convolution witha 3*3 kernel : 7 -3 + 1 = 5*5 output dimension
    SAME,           //Preserves the input dimension
    FULL            //Extends the input dimension
} padding_t;


typedef struct{
    size_t col;
    size_t row;
    size_t nmap;
    size_t nbatch;  //<----------- IS A NUMBER OF FILTER FOR A KERNEL K
} dimensions;

typedef union{
    struct{
        size_t col;
        size_t row;
        size_t nmap;
        size_t nbatch;
    };


    struct{

    };

    size_t shape[4];
} tensor_shape_t;

//Tensor 4 used for : Kernels representation (All)

//----------------------------------//             
//---           TENSOR 4         ---//
//----------------------------------//
//Used for Storing Kernels, InputFeature maps X, Output of convolutions Z, output of ReLU A, and output of pooling P
typedef struct{
    size_t flatten_size;
    size_t shape[4];            //Shape[0] = cols, Shape[1] = Rows, Shape[2] = N_feature_map, Shape[3] = N_filter
    size_t strides[4];
    float *datas;
} tensor4_t;



//----------------------------------//            
//---       init and free        ---//
//----------------------------------//

//** Col, row, n_fmap, n_filter */
tensor4_t* init_tensor4(size_t d0, size_t d1, size_t d2, size_t d3, distribution_t type);
void free_tensor4(tensor4_t **t);



//----------------------------------//            
//---           Access           ---//
//----------------------------------//


//GERER LES ACCES INCORRECTS

//Access to a given index 
static inline size_t get_t4_idx(const tensor4_t *t,size_t idx0, size_t idx1, size_t idx2, size_t idx3){
return idx0*t->strides[0] + idx1*t->strides[1] + idx2*t->strides[2] + idx3*t->strides[3];} 

//Access to a given val index based
static inline float get_t4_val(const tensor4_t *t, size_t idx0, size_t idx1, size_t idx2, size_t idx3){
return t->datas[get_t4_idx(t,idx0,idx1,idx2,idx3)];}

//Set a given val
static inline void set_t4_val(const tensor4_t *t, float val, size_t idx0, size_t idx1, size_t idx2, size_t idx3){
t->datas[get_t4_idx(t,idx0,idx1,idx2,idx3)] = val;}

//----------------------------------//            
//---           Display          ---//
//----------------------------------//
//Print infos on tensors
void print_tensor4_shape(const tensor4_t *t);
void print_tensor4_data(const tensor4_t *t);
void print_tensor4_mask(const uint8_t *mask, const tensor4_t *A);



//----------------------------------//            
//---         Operation          ---//
//----------------------------------//
//Generic convolution function
//void convbuffer(const tensor4_t *X, const tensor4_t *K, tensor4_t **Z, size_t pad_top, size_t pad_bottom, size_t pad_left, size_t pad_right);



void conv4(const tensor4_t *X, const tensor4_t *K, tensor4_t **Z, padding_t padding);


//X is the input, K the parameters, type padding type and Z the output parameter
//void conv_cumulate(const tensor4_t *X, const tensor4_t *K, const padding_t type, tensor4_t **Z);

//Adds constants to the maps, on the shape[2] Axis
void addBias(tensor4_t *t, const float *b);                     //<--REWORK ?

//Performs ReLU
//void ReLU(float *tab, size_t size);
void ReLU(tensor4_t *T);

//Performs Softmax
void SoftMax(float *tab, size_t size);

//Performs MaxPool with a stored uint8 pooling mask
void MaxPool(const tensor4_t *A, tensor4_t **P, uint8_t **Pooling_Mask);

//Performs the multiplication of W*X
void matvec(const float *X, const tensor4_t *W, float **Z);

void outputConv(const tensor4_t *X, const tensor4_t *K, tensor4_t **Z, padding_t padding);

void getPadding(size_t *t, size_t *b, size_t *l, size_t *r, const tensor4_t *K, padding_t padding);

void convBuffer(const tensor4_t *X, const float *dataX,
                const tensor4_t *K, const float *dataK,
                const float b,
                const tensor4_t *Z, float *dataZ,
                size_t pad_top, size_t pad_bottom, size_t pad_left, size_t pad_right);


//Wrappers
#endif