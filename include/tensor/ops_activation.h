#ifndef OPS_ACTIVATION_H
#define OPS_ACTIVATION_H

//------> OK ?

#include "tensor.h"

void ReLU(tensor4_t *T);

//Retravailler la signature
void SoftMax(float *tab, size_t size);
#endif