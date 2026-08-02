#ifndef DENSE_LAYER_H
#define DENSE_LAYER_H

#include "inttypes.h"

//My includes
#include "debug.h"
#include "tensor.h"


typedef enum{
    RELU,
    SOFTMAX
} activation_type_t;


typedef struct{
    
    tensor4_t *weights;         //Dimension size of input * N_neurons
    float *biases;

    size_t n_out;       //Number of neurons nececarry ?

} DenseLayer;


void init_dense();

#endif