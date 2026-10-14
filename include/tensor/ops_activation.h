#ifndef OPS_ACTIVATION_H
#define OPS_ACTIVATION_H

#include "tensor.h"

typedef enum{
    RELU,
    SOFTMAX
} activation_t;



void ReLU(tensor4_t *T);

void SoftMax(tensor4_t *T);

#endif