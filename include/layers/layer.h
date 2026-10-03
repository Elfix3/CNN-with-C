#ifndef LAYER_H
#define LAYER_H


//My 2 types of layers
#include "conv_layer.h"
#include "dense_layer.h"


typedef struct {
    void *impl;
    const void* (*forward)(void *impl, const void *input); 
    const void* (*backward)(void *impl, const void *input);
    
    const void* (*clean)(void *impl);
} Layer_Wrapper;



Layer_Wrapper *make_conv_layer(size_t kernel_size ,size_t n_fmap, size_t n_filter, padding_t type);
//Layer_Wrapper *make_dense_layer();

static const void clean_layer_W(Layer_Wrapper *L){
    L->clean(L->impl);
}

static const void forward(Layer_Wrapper *L, const void *Input){
    L->forward(L->impl, Input);
}
void backward(Layer_Wrapper* L, const void *Input){
    L->backward(L->impl, Input);
}

//Ne va ptetre pas être conservée
void update(Layer_Wrapper *L){

}
//
#endif