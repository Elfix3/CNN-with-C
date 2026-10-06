#ifndef OPS_CONV_H
#define OPS_CONV_H

#include "tensor.h"

typedef enum{
    VALID,          //Shrinks the input dimension : size_input-size_kernel + 1 becomes the new output dimension : ex : 7*7 feature map convolution witha 3*3 kernel : 7 -3 + 1 = 5*5 output dimension
    SAME,           //Preserves the input dimension
    FULL            //Extends the input dimension
} padding_t;

static const char* padding_strings[3] = {
    "VALID",
    "SAME",
    "FULL",
};

//doit gérer les paddings en mode backward aussi
void getPadding(size_t *t, size_t *b, size_t *l, size_t *r, const tensor4_t *K, padding_t padding);

//Spécialisées en convolution,

//Optimised convolution of dataX with dataK (tensor pointers are just to give the information dimension)

void convBuffer(const tensor4_t *X, const float *restrict dataX,
                const tensor4_t *K, const float *restrict dataK,
                const tensor4_t *Z, float *restrict dataZ,
                size_t pad_top, size_t pad_bottom, size_t pad_left, size_t pad_right);


//threaded version
void conv4(const tensor4_t *X, const tensor4_t *K, tensor4_t *Z, padding_t padding);


//sets the bias
void setBias(tensor4_t *Z, const float *b);

void kernelFlip(const tensor4_t *K, tensor4_t **Kflipped);

void MaxPool(const tensor4_t *A, tensor4_t *P, uint8_t *Pooling_Mask);

void setBias(tensor4_t *Z, const float *b);

//dimensions :
tensor4_dim_t get_conv_output_dim(tensor4_dim_t X_dim, tensor4_dim_t K_dim, padding_t pad);

tensor4_dim_t get_maxpool_output_dim(tensor4_dim_t A_dim);


#endif