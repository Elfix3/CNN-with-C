#include "conv_layer.h"



ConvLayer *init_conv_layer(size_t kernel_size, size_t n_fmap, size_t n_filter, padding_type_t type){
    ConvLayer *l = (ConvLayer*)malloc(sizeof(ConvLayer));
    
    l->InputFMaps = NULL;

    l->Kernels = init_tensor4(kernel_size,kernel_size,n_fmap, n_filter,UNIFORM);
    l->b = calloc(n_filter,sizeof(float)*n_filter);
    l->padding_type = type;
    
    for(int i = 0; i<  n_filter;i ++){
        l->b[i] = 0.1f;
    }

    l->OutputFMaps = NULL;
    l->ReLUMask = NULL;
    l->Pooling_Mask = NULL;


    #if DEBUG
        printf("Layer sucessfully created, tensor info :\n");
        print_tensor4_shape(l->Kernels);
        print_tensor4_data(l->Kernels);
    #endif

    return l;
}


static inline void set_input(ConvLayer *l,tensor4_t * input){
    assert(input != NULL && "[set_input] : null input tensor");
    assert(input->shape[3] == 1 && "[set_input] : number of filter must be 1");
    assert(input->strides[2]>=2 && "[set_input] : image must have at lease 2*2 size");

    l->InputFMaps = input;

}


void forward(ConvLayer *l, tensor4_t *inputFmaps){
    set_input(l,inputFmaps);
    assert(l != NULL && "[forward] : null input layer");
    assert(l->InputFMaps != NULL && "[forward] : null input maps");

    tensor4_t* conv_output = conv_cumulate(l->InputFMaps,l->Kernels,VALID);
    LOG("Conv output :");
    LOG(l->InputFMaps);

    size_t convd0 =conv_output->shape[0];
    size_t convd1 =conv_output->shape[1];
    size_t convd2 =conv_output->shape[2];

    printf("Add bias :\n");
    addBias(conv_output,l->b);
    print_tensor4_data(conv_output);


    printf("Relu :\n");
    ReLU(conv_output);
    print_tensor4_data(conv_output);
    print_mask(l->ReLUMask,convd0,convd1,convd2);

    printf("Pooling :\n");
    //MaxPool(conv_output,(mask_4t));
    print_tensor4_data(conv_output);
    print_mask(l->Pooling_Mask,convd0,convd1,convd2);

    l->OutputFMaps = conv_output;
}