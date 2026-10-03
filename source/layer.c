#include "layer.h"

Layer_Wrapper *make_conv_layer(size_t kernel_size, size_t n_fmap, size_t n_filter, padding_t type){
    Layer_Wrapper *LW = malloc(sizeof(Layer_Wrapper));
    if(!LW){LOG_ERROR("BAD ALLOC");}

    LW->impl = (void*)init_conv_layer(kernel_size, n_fmap, n_filter, type);
    LW->forward = forward_conv;
    LW->backward = backward_conv;
    LW->clean = clean_conv;

    return LW
}
