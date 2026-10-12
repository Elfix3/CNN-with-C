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
        init_dense_layer_parameter_t dense;
    };
} layer_params_t;

typedef struct {
    void *impl;
    void                (*free)(void *impl);
    void                (*clean)(void *impl);
    void                (*recap)(const void *impl);
    tensor4_dim_t       (*output_dim)(const void *impl);
    const tensor4_t*    (*forward) (void *impl, const tensor4_t *Prev_Layer_Output); 
    const tensor4_t*    (*backward)(void *impl, const tensor4_t *Next_Layer_Output);
    void (*update)      (void *impl);
} Layer_Wrapper;


/* 
Layer_Wrapper *make_conv_layer(init_conv_layer_parameter_t p);
Layer_Wrapper *make_dense_layer(void *p); */


Layer_Wrapper *createLayer(tensor4_dim_t input_dim ,const layer_params_t p);
//Alloc Layer


static inline void freeL(Layer_Wrapper *L){
    L->free(L->impl);
}

//on oublie les statics
static inline void recap(Layer_Wrapper *L){
    L->recap(L->impl);
}

static inline tensor4_dim_t output_dim(Layer_Wrapper *L){
    return L->output_dim(L->impl);
}

static inline void clean(Layer_Wrapper *L){
    L->clean(L->impl);
}

static inline const tensor4_t* forward(Layer_Wrapper *L, const void *Input){
    return L->forward(L->impl, Input);
}

static inline const tensor4_t* backward(Layer_Wrapper* L, const void *Input){
    return L->backward(L->impl, Input);
}

//Ne va ptetre pas être conservée
/* void update(Layer_Wrapper *L){

} */
//
#endif