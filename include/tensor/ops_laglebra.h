#ifndef OP_LALGEBRA_H
#define OP_LALGEBRA_H

#include "tensor.h"

//Opérations de convolution


void allocMatMult(const tensor4_t *X, const tensor4_t *W, tensor4_t **Z);

void matmult(const tensor4_t *X, const tensor4_t *W, tensor4_t *Z); //et l'output connard ?
#endif