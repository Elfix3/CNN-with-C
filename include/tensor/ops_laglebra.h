#ifndef OP_LALGEBRA_H
#define OP_LALGEBRA_H

#include "tensor.h"
//Opérations de convolution

void allocMatMult(const tensor2_t *X, const tensor2_t *W, tensor2_t **Z);

void matmult(const tensor2_t *X, const tensor2_t *W, tensor2_t *Z); //et l'output connard ?
#endif