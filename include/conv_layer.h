#ifndef LAYER_H
#define LAYER_H

#include "inttypes.h"
#include "stdio.h"

//My includes
#include "tensor.h"
#include "debug.h"

/* typedef enum{

} forward_t;
 */

typedef struct{
    //Operation ;
    //X ---->[conv with K]   ----> Z
    //Z ---->[+ biases]      ----> Z
    //Z ---->[ReLU]          ----> A
    //A ---->[MaxPooling]    ----> P

    //P ---->[Goes to another Layer] ----> Is the next X

    //-> Input 
    tensor4_t *X;  //shape(n_cols * n_cols * n_maps * 1)

    //-> Parameters
    tensor4_t *K;           //shape(qize_k * size_k * n_maps * n_filters)
    float *B;               //shape(n_cols * 1 * 1 * 1) //make it a tensor ?
    padding_t padding_type;
   //More tensor4_t for adam optimisation, for K and B both momentum and RMSprop

    //->Back prop cache
    //tensor4_t *P_Prev;                  // <=> à X passé en input
    tensor4_t *A;                       // <=> RELU(Z)
    uint8_t *Pooling_Mask;              //Masque de pooling


    //-> Output
    tensor4_t *P;    //Goes to the nextlayer

    //<--- Convolution output size will be rows-kernel_size+1 , cols -kernel_size+1; --->

    /*-> Output are Feature maps after
    1) Convolution of each filter with the feature map
    2) Filter has n_input kernels, we sum the convoluted results for each filter
    3) We add the bias
    4) We perform ReLU
    5) We perform MaxPooling (Stide 1, 2*2)
    */
} ConvLayer;




/**Initializes the parameters of the layer */
ConvLayer *init_conv_layer(size_t kernel_size ,size_t n_fmap, size_t n_filter, padding_t type);
void clean_conv_layer(ConvLayer *l);
void forward(ConvLayer *l, tensor4_t *X);


#endif
