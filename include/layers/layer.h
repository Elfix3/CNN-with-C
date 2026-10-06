#ifndef LAYER_H
#define LAYER_H


//My 2 types of layers
#include "conv_layer.h"
#include "dense_layer.h"

typedef enum{
    CONV_LAYER,
    DENSE_LAYER,
} layer_type_t;

typedef struct {
    layer_type_t type;
    union {
        init_conv_layer_parameter_t  conv;
        //init_dense_layer_parameter_t dense;
    };
} layer_params_t;

typedef struct {
    void *impl;
    void (*free)(void *impl);
    void (*clean)(void *impl);
    void (*recap)(void *impl);
    const void* (*forward)(void *impl, const void *input); 
    const void* (*backward)(void *impl, const void *input);
} Layer_Wrapper;


/* 
Layer_Wrapper *make_conv_layer(init_conv_layer_parameter_t p);
Layer_Wrapper *make_dense_layer(void *p); */


Layer_Wrapper *createLayer(layer_type_t *ltype, void *layerparameters);
//Alloc Layer

static void recap(Layer_Wrapper *L){
    L->recap(L->impl);
}

static const void* clean(Layer_Wrapper *L){
    L->clean(L->impl);
}

static const void* forward(Layer_Wrapper *L, const void *Input){
    L->forward(L->impl, Input);
}

static const void* backward(Layer_Wrapper* L, const void *Input){
    L->backward(L->impl, Input);
}

//Ne va ptetre pas être conservée
/* void update(Layer_Wrapper *L){

} */
//
#endif