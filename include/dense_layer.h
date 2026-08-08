#ifndef DENSE_LAYER_H
#define DENSE_LAYER_H

#include "inttypes.h"

//My includes
#include "debug.h"
#include "tensor.h"


typedef enum{
    RELU,
    SOFTMAX
} activation_t;


typedef struct{
    //Operation ;
    //X ---->   [*W + b]              ----> Z
    //Z ---->   [ReLU or SOFTMAX]     ----> A

    //-> Input 
    const float *X;             //INPUT POINTER NOT OWNED

    //-> Parameters
    tensor4_t *W;         //Store all the sizes (n_input, m_neurons, 1, 1)
    float *b;

    //-> Activation
    activation_t activation_type;

    //->Back prop cache
    tensor4_t *dW;
    float   *dB;

    //-> Output
    float *A;           //Goes to the next layer            <--- FORWARD
    float *dX;           //Goes to the previous layer       <--- BACKWARD

} DenseLayer;

//**n is the number of inputs, m the number of neurons/
DenseLayer* init_dense(size_t n, size_t m, activation_t type);    //Add some dimensions as well

#endif