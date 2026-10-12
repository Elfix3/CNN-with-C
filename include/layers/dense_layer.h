#ifndef DENSE_LAYER_H
#define DENSE_LAYER_H

#include "inttypes.h"

//My includes
#include "debug.h"
#include "tensor/tensor.h"
#include "tensor/ops_activation.h"


typedef struct {
    size_t m_neurons;
    activation_t activation_type;
} init_dense_layer_parameter_t;


typedef struct{
    //Operation ;
    //X ---->   [*W + b]              ----> Z
    //Z ---->   [ReLU or SOFTMAX]     ----> A

    //-> Input 
    const tensor4_t         *X;             //INPUT POINTER NOT OWNED

    //-> Parameters
    tensor4_t               *W;               //Store all the sizes (n_input, m_neurons, 1, 1)
    float                   *b;

    //-> Activation
    activation_t            activation_type;

    //->Back prop cache
    tensor4_t               *dW;
    float                   *dB;

    //-> Output
    tensor4_t               *A;           //Goes to the next layer            <--- FORWARD
    tensor4_t               *dX;           //Goes to the previous layer       <--- BACKWARD

} DenseLayer;

//**n is the number of inputs, m the number of neurons/
DenseLayer* init_dense_layerOld(size_t n, size_t m, activation_t type);    //Add some dimensions as well

DenseLayer *init_dense_layer(tensor4_dim_t input_dim,init_dense_layer_parameter_t param);

// ---->   WRAPPED FUNCTIONS

//Unallocates all the layer
void free_dense(DenseLayer *l);

//Clears all the buffers, keeps parameters
void clean_dense(DenseLayer *l);

//Prints all info on the layer
void recap_dense(const DenseLayer *l);

//Fetches the output dimension of the layer
tensor4_dim_t output_dim_dense(const DenseLayer *l);

//Forward process, returns A to the next layer
const tensor4_t* forward_dense(DenseLayer *l, const tensor4_t *X);

//Backward process, returns dX to the previous layer
const tensor4_t* backward_dense(DenseLayer *l, const tensor4_t *dA);

//updates the kernels and the biases
void update_dense(DenseLayer *l);

#endif