#include "conv_layer.h"



ConvLayer *init_conv_layer(size_t kernel_size, size_t n_fmap, size_t n_filter, padding_t type){
    ConvLayer *l = (ConvLayer*)malloc(sizeof(ConvLayer));
    

    //-> Input 
    l->X = NULL;
    
    //-> Parameters
    l->K = init_tensor4(kernel_size,kernel_size,n_fmap, n_filter,UNIFORM);
    l->b = calloc(n_filter,sizeof(float));  
    
    //-> Padding
    l->padding = type;
    
    //Back prop cache outputs
    l->A = NULL;
    l->Pooling_Mask = NULL;

    //->Back prop cache gradients
    l->dK = NULL;
    l->dB = NULL;
    
    //-> Output
    l->P = NULL;
    l->dX = NULL;


    
    for(size_t i = 0; i<  n_filter;i ++){
        l->b[i] = 0.2f;
    }

    #if VERBOSE
        printf("Layer sucessfully created, tensor info :\n");
        print_tensor4_data(l->K);
    #endif

    return l;
}

void clean_conv_layer(ConvLayer *l){
    assert(l != NULL && "Error null pointer in clean_conv_layer");
    
    //Back prop cache outputs
    free_tensor4(&l->A);
    free(l->Pooling_Mask);

    //->Back prop cache gradients
    free_tensor4(&l->dK);
    free(l->dB);
    
    //-> Output
    free_tensor4(&l->P);
    free_tensor4(&l->dX);

}

void forward_conv(ConvLayer *l, const tensor4_t *X){
    
    REQUIRE(l != NULL,"Layer cannot be NULL");
    REQUIRE(X != NULL,"Input X cannot be NULL");
    REQUIRE(l->K->nmap == X->nmap, "Feature map number must match between Input X and kernel K"); 

    //A terme possibilité de réallouer les Kernels ?? (Franchement mauvaise idée) (on enlève définitivement)
    
    //Sets the input pointer, not owned
    l->X = X;

    //RESETS THE TENSOR : LIKE CLEAN IT, hold up, setbias does the job !! false alert

    //Allocates the output tensor               (Va être déplacé dans le cstr)
    allocZ(l->X,l->K,&l->A,l->padding);
    
    setBias(l->A,l->b);
    //print_tensor4_data(l->A);
    
    //pas besoin de reset grâce au biais
    conv4threaded(l->X,l->K, l->A, l->padding);
    //conv4(l->X,l->K, l->A, l->padding);
    //print_tensor4_data(l->A);

    ReLU(l->A);
    //print_tensor4_data(l->A);

    //Allocates the output pooled tensor               (Va être déplacé dans le cstr)
    allocP(l->A, &l->P, &l->Pooling_Mask);
    MaxPoolThreaded(l->A, l->P, l->Pooling_Mask);
    
}

void backward(ConvLayer *l, const tensor4_t *dP){
}

//Kinda useless
/* static inline void set_input(ConvLayer *l,tensor4_t * input){
    assert(input != NULL && "[set_input] : null input tensor");
    assert(input->shape[3] == 1 && "[set_input] : number of filter must be 1");
    assert(input->strides[2]>=2 && "[set_input] : image must have at lease 2*2 size");

    l->InputFMaps = input;

}
 */