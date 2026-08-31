#ifndef LAYER_H
#define LAYER_H

#include "inttypes.h"
#include "stdio.h"

//My includes
#include "tensor.h"
#include "debug.h"

typedef struct{
    //Operation ;
    //X ---->   [conv K] + b      ----> Z
    //Z ---->   [ReLU]            ----> A
    //A ---->   [MaxPooling]      ----> P
    //P ---->[Goes to another Layer] ----> Is the next X

    //-> Input 
    const tensor4_t *X;               //INPUT POINTER NOT OWNED

    //-> Parameters
    tensor4_t *K;               //shape(size_k * size_k * n_maps * n_filters)
    float *b;                   //size is n filter
    
    //-> Padding
    padding_t padding_type;
   
    //->Back prop cache
    tensor4_t *dK;
    float   *dB;
    tensor4_t *A;                       // <=> RELU(Z) (owned)
    uint8_t *Pooling_Mask;              //Masque de pooling

    //-> Output
    tensor4_t *P;           //Goes to the next layer         <--- FORWARD
    tensor4_t *dX;          //Goes to the previous layer     <--- BACKWARD
} ConvLayer;

//More tensor4_t for adam optimisation, for K and B both momentum and RMSprop



/**Initializes the parameters of the layer */
ConvLayer *init_conv_layer(size_t kernel_size ,size_t n_fmap, size_t n_filter, padding_t type);

//Clears data except parameters
void clean_conv_layer(ConvLayer *l);

//X has : (x*y*N_features*N_Batch)
void forward(ConvLayer *l, const tensor4_t *X);
void backward(ConvLayer *l, const tensor4_t *dP);


#endif
