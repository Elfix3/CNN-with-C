#include "layers/layer.h"

/* Layer_Wrapper *make_conv_layer(size_t kernel_size, size_t n_fmap, size_t n_filter, padding_t type){
    Layer_Wrapper *LW = malloc(sizeof(Layer_Wrapper));
    if(!LW){LOG_ERROR("BAD ALLOC");}

    LW->impl = (void*)init_conv_layer(kernel_size, n_fmap, n_filter, type);
    LW->forward = forward_conv;
    LW->backward = backward_conv;
    LW->clean = NULL;

    return NULL;
} */

//<--- Conv Wrappers --->

static void recap_conv_w(void *impl){
    recap_conv((ConvLayer*)impl);
}


static const void* forward_conv_w(void *impl, const void*input){
    return (const void*)forward_conv((ConvLayer*)impl, (const tensor4_t*)input);
}


//<--- Dense Wrappers --->

void recap_dense_w(void *impl){
    recap_conv((ConvLayer*)impl);
}

void forward_dense_w(void *impl, const void *input){

}

Layer_Wrapper *make_conv_layer(init_conv_layer_parameter_t p){
    Layer_Wrapper *LW = malloc(sizeof(Layer_Wrapper));

    LW->impl = (void*)init_conv_layer(p);
    LW->recap = recap_conv_w;
    LW->forward = forward_conv_w;
    
    //LW->clean = clean_conv;
    //LW->backward = backward_conv;

    return LW;
}

Layer_Wrapper *make_dense_layer(void *p){
    return NULL;
}

Layer_Wrapper *createLayer(layer_type_t ltype, void *layerparameters){
    Layer_Wrapper *LW = malloc(sizeof(Layer_Wrapper));
    
    switch(ltype){
        case DENSE_LAYER :
        break;
        case CONV_LAYER :
            LW->impl = (void*)init_conv_layer(*(init_conv_layer_parameter_t*)layerparameters);
            //tout le reste avec les bonnes fonctions
        break;
        default :
            LOG_ERROR("Unrecognized layer type");
        break;
    }

    return LW;
}
