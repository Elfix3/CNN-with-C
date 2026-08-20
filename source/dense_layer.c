#include "dense_layer.h"


DenseLayer *init_dense(size_t n, size_t m, activation_t type){
    DenseLayer *l = malloc(sizeof(DenseLayer));

    l->W = init_tensor4(n,m,1,1, UNIFORM);
    l->b = calloc(m,sizeof(float));
    l->activation_type = type;
    return l;

}

void forward_dense(DenseLayer *l, const float *X){
    //Important assomption : X must be allocated with the same dimension as W->shape[0]
    
    /* l->X = X;
    
    matvec(X,l->W,&l->A);
    
    //Usage de cette fonction bof bof
    //addBias_buffer(l->A,l->b,l->W->shape[1]);

    if(l->activation_type == RELU){
        ReLU(l->A,l->W->shape[1]);
    } else if(l->activation_type == SOFTMAX){
        SoftMax(l->A,l->W->shape[1]);
    } */
}
