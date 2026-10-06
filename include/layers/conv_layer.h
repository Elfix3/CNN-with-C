#ifndef CONV_LAYER_H
#define CONV_LAYER_H

#include "inttypes.h"
#include "stdio.h"

//My includes
#include "tensor/tensor.h"
#include "tensor/ops_conv.h"
#include "tensor/ops_activation.h"
#include "debug.h"





typedef struct{
    tensor4_t *Input;
    size_t kernel_size;
    size_t n_filter;
    padding_t padding_type;

} init_conv_layer_parameter_t;


typedef struct{
    //Operation ;
    //X ---->   b + [conv K]      ----> Z
    //Z ---->   [ReLU]            ----> A
    //A ---->   [MaxPooling]      ----> P
    //P ---->[Goes to another Layer] ----> Is the next X


    //Add an ID ?

    //-> Input 
    const tensor4_t         *X;                     //INPUT POINTER NOT OWNED
    tensor4_dim_t           input_shape;

    //-> Parameters
    tensor4_t               *K;                       //shape(size_k * size_k * n_maps * n_filters)
    float                   *b;                       //size is n_filter * n_batchs
    
    //-> Padding
    padding_t               padding;
   
    //Back prop cache outputs
    tensor4_t               *A;                       // <=> RELU(Z) (owned)
    uint8_t                 *PoolingMask;             //Masque de pooling

    //K flipped for backpropconv4
    tensor4_t               *Kflip;

    //->Back prop cache gradients
    tensor4_t               *dK;                      //OWNED
    float                   *dB;                      //OWNED

    
    //-> Output
    tensor4_t               *P;                       //Goes to the next layer OWNED        <--- FORWARD
    tensor4_t               *dX;                      //Goes to the previous layer OWNED     <--- BACKWARD
} ConvLayer;

//More tensor4_t for adam optimisation, for K and B both momentum and RMSprop




/**Initializes the parameters of the layer */           //implement constructor for previous layer as well ? or from prevous output Shape ?
ConvLayer *init_conv_layerOld(size_t kernel_size ,size_t n_fmap, size_t n_filter, padding_t type);

//il faut pouvoir initaliser une couche depuis la précédente
ConvLayer *init_conv_layer(init_conv_layer_parameter_t param);

void recap_conv(ConvLayer *l);

//Clears data except parameters
void clean_conv(ConvLayer *l);

//X has : (x*y*N_features*N_Batch)

//Doit retourner ptr vers la sortie
tensor4_t* forward_conv(ConvLayer *l, const tensor4_t *X);

//Doit retourner ptr vers la sortie
void backward_conv(ConvLayer *l, const tensor4_t *dP);


#endif
