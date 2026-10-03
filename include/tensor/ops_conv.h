#ifndef OPS_CONV_H
#define OPS_CONV_H

#include "tensor.h"

void getPadding(size_t *t, size_t *b, size_t *l, size_t *r, const tensor4_t *K, padding_t padding);



//Allocation dédiées à déplacer au niveau des couches de convolutions et pas ici
void allocZ(const tensor4_t *X, const tensor4_t *K, tensor4_t **Z, padding_t padding);

void allocP(const tensor4_t *A, tensor4_t **P, uint8_t **Pooling_Mask);

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

#endif