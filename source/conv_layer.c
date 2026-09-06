#include "conv_layer.h"



ConvLayer *init_conv_layer(size_t kernel_size, size_t n_fmap, size_t n_filter, padding_t type){
    ConvLayer *l = (ConvLayer*)malloc(sizeof(ConvLayer));
    

    //-> Input 
    l->X = NULL;
    l->K = init_tensor4(kernel_size,kernel_size,n_fmap, n_filter,UNIFORM);
    l->b = calloc(n_filter,sizeof(float)*n_filter);                             //0 initaliztion
    
    l->padding = type;
    
    
    l->A = NULL;
    // Test purposes
    for(size_t i = 0; i<  n_filter;i ++){
        l->b[i] = -0.1f;
    }

    //Unknown until first forward
    l->P = NULL;
    l->Pooling_Mask = NULL;

    //->Back prop cache
    tensor4_t *dK = NULL;
    float   *dB = NULL;
    tensor4_t *A;
    uint8_t *Pooling_Mask;


    #if VERBOSE
        printf("Layer sucessfully created, tensor info :\n");
        print_tensor4_data(l->K);
    #endif

    return l;
}

void clean_conv_layer(ConvLayer *l){
    assert(l != NULL && "Error null pointer in clean_conv_layer");
    
    free(l->Pooling_Mask);
    free_tensor4(&l->dK);
    free(l->dB);
    free_tensor4(&l->A);
    free_tensor4(&l->P);
    free_tensor4(&l->dX);

}

void forward(ConvLayer *l, const tensor4_t *X){
    
    REQUIRE(l != NULL,"Layer cannot be NULL");
    REQUIRE(X != NULL,"Input X cannot be NULL");

    //A terme possibilité de réallouer les Kernels ??
    REQUIRE(l->K->nmap == X->nmap, "Featur map number must match between Input X and kernel K"); 
    
    l->X = X;
    allocZ(l->X,l->K,&l->A,l->padding);
    //add bias before;
    
    //conv4(l->X, l->K, &l->A, l->padding);
    
    //--->  TIME METRICS
    
    
    //Allouer l->Z
    
    
    //double start = omp_get_wtime();
    
    
    //conv_cumulate(X,l->K,l->padding_type,&l->A);

    

    
    //addBias(l->A,l->b);
    //printf("\n\n Valeur post biais : \n\n");
    //print_tensor4_data(l->A);


    //ReLU(l->A->datas,l->A->flatten_size);
    //printf("\n\n Valeur post ReLU : \n\n");
    //print_tensor4_data(l->A);
    
    //MaxPool(l->A,&l->P,&l->Pooling_Mask);
    //printf("\n\n  Valeur post Pooling : \n\n");
    //print_tensor4_data(l->P);
    
    //printf("\n\n  Valeur masque pooling : \n\n");
    //print_tensor4_mask(l->Pooling_Mask, l->A);
    //printf("\n\n<--- CONV RESULT --->\n\n");
    //print_tensor4_data(l->A);
    
    

    
    //printf("\n\n<---  BIAS ADDED --->\n\n");
    //print_tensor4_data(l->A);
    
     
    

    //--->  TIME METRICS
    //double end = omp_get_wtime();
    //printf("Temps : %f secondes\n", end - start);
    
    //printf("\n\n<---  RELU --->\n\n");
    //print_tensor4_data(l->A);


    //LOG("Conv output :");
    

    /* size_t convd0 =conv_output->shape[0];
    size_t convd1 =conv_output->shape[1];
    size_t convd2 =conv_output->shape[2]; */

    //print_mask(l->ReLUMask,convd0,convd1,convd2);

    //printf("Pooling :\n");
    //MaxPool(conv_output,(mask_4t));
    //print_tensor4_data(conv_output);
    //print_mask(l->Pooling_Mask,convd0,convd1,convd2);

    //l->P = conv_output;
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