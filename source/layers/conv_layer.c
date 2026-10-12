#include "layers/conv_layer.h"

static const char* padding_to_string(padding_t pad){
    return padding_strings[pad];
}


static tensor4_t* allocA(tensor4_dim_t in_dim, ConvLayer *l){
    //Faire dépendre des dimensions de K plutôt
    REQUIRE(l != NULL, "Error NULL layer");
    REQUIRE(l->K != NULL, "Error K tensor in layer is NULL");
    
    tensor4_dim_t A_dim =  get_conv_output_dim(in_dim,l->K->dim,l->padding);

    tensor4_t *A = (tensor4_t*)init_tensor4(A_dim.col, A_dim.row, A_dim.nmap, A_dim.nbatch, NOFILL);
    
    if(!A){
        LOG_ERROR("Error bad alloc");
    }
    return A;
}


static tensor4_t* allocP(tensor4_dim_t A_dim){
    tensor4_dim_t P_dim = get_maxpool_output_dim(A_dim);

    tensor4_t *P = (tensor4_t*)init_tensor4(P_dim.col, P_dim.row, P_dim.nmap, P_dim.nbatch, NOFILL);

    if(!P){
        LOG_ERROR("Error bad alloc");
    }
    return P;

}

ConvLayer *init_conv_layerOld(size_t kernel_size, size_t n_fmap, size_t n_filter, padding_t type){
    ConvLayer *l = (ConvLayer*)malloc(sizeof(ConvLayer));
    

    /* //-> Input  Ca ok
    l->X = NULL;
    
    //-> Parameters
    l->K = init_tensor4(kernel_size,kernel_size,n_fmap, n_filter,UNIFORM);
    l->b = calloc(n_filter,sizeof(float));  
    
    //-> Padding
    l->padding = type;
    
    //Back prop cache outputs

    //allouer direct
    l->A = NULL;

    //allouer direct
    l->PoolingMask = NULL;

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
    #endif */

    return l;
}

ConvLayer *init_conv_layer(tensor4_dim_t input_dim ,init_conv_layer_parameter_t param){
    ConvLayer *l = (ConvLayer*)malloc(sizeof(ConvLayer));

    l->input_shape = input_dim; // Laisser ?
    l->X = NULL;
    
    l->padding = param.padding_type;

    //Init K
    l->K = init_tensor4(param.kernel_size, param.kernel_size, input_dim.nmap, param.n_filter,UNIFORM);
    
    //Init b
    l->b = calloc(param.n_filter,sizeof(float));  
    
    if(l->b == NULL){
        LOG_ERROR("Error during bias initalization");
    }

    //faut check les ptr
    l->A = allocA(input_dim, l);
    l->P = allocP(l->A->dim);

    l->PoolingMask = calloc(l->A->flatten_size, sizeof(uint8_t));

    //tout plein de truc à allouer encore
    
    return l;
}

void free_conv(ConvLayer *l){
    //completely clears the layer
}

void recap_conv(const ConvLayer *l){
    //Très insuffisant, il faut indiquer les activations, le pooling, le padding de convolution
    
    LOG_INFO_HEADER("Convolutional layer recap");
    

    LOG_INFO("%-12s : %zu x %zu x %zu\t(row x col x nchannels)", "Input",
         l->input_shape.col, l->input_shape.row, l->input_shape.nmap);
    
    LOG_INFO("%-12s : %zu x %zu x %zu\t(row x col x nfilters)", "Kernel",
         l->K->col, l->K->row, l->K->nfilter);
     
    LOG_INFO("%-12s : %s", "Padding", padding_to_string(l->padding));
   
    LOG_INFO("%-12s : ReLU", "Activation");

    LOG_INFO("%-12s : MaxPool 2 x 2\tstride 2\n", "Pooling");

    LOG_INFO("%-12s : %zu x %zu x %zu\t(row x col x nfilters)", "Output",
         l->P->col, l->P->row, l->P->nfilter);
    
    LOG_INFO("%-12s : %zu", "Num parameters", l->K->flatten_size+l->K->nfilter);


    //LOG_COLOR("")
    //LOG_INFO("Kernel dimension : %zu x %zu \nNumber of input feature maps : %zu \nNumber of filters : %zu", l->K->col, l->K->row, l->K->nmap, l->K->nfilter);
}

tensor4_dim_t output_dim_conv(const ConvLayer *l){
    return get_t4_dim(l->P);
}

void clean_conv(ConvLayer *l){
    assert(l != NULL && "Error null pointer in clean_conv_layer");
    
    //Ca devrait pas être une logique de free, plus une

    //Back prop cache outputs
    
    free_tensor4(&l->A);
    free(l->PoolingMask);

    //->Back prop cache gradients
    free_tensor4(&l->dK);
    free(l->dB);
    
    //->Kflip
    free_tensor4(&l->Kflip);

    //-> Output
    free_tensor4(&l->P);
    free_tensor4(&l->dX);

}

const tensor4_t* forward_conv(ConvLayer *l, const tensor4_t *X){
    
    REQUIRE(l != NULL,"Layer cannot be NULL");
    REQUIRE(X != NULL,"Input X cannot be NULL");
    REQUIRE(l->K->nmap == X->nmap, "Feature map number must match between Input X and kernel K"); 

    //assert(0);
    
    //Sets the input pointer, not owned
    l->X = X;

    //RESETS THE TENSOR : LIKE CLEAN IT, hold up, setbias does the job !! false alert
    setBias(l->A,l->b);
    //print_tensor4_data(l->A);
    
    //pas besoin de reset grâce au biais
    conv4(l->X,l->K, l->A, l->padding);
    //conv4(l->X,l->K, l->A, l->padding);
    //print_tensor4_data(l->A);

    ReLU(l->A);
    
    //calloc le masque
    l->PoolingMask = calloc(l->A->flatten_size, sizeof(uint8_t));
    MaxPool(l->A, l->P, l->PoolingMask);
    
    //print_tensor4_mask(l->PoolingMask, l->A);
    return l->P;
}

const tensor4_t* backward_conv(ConvLayer *l, const tensor4_t *dP){
    return NULL;
}

void update_conv(ConvLayer *l){
}

//Kinda useless
/* static inline void set_input(ConvLayer *l,tensor4_t * input){
    assert(input != NULL && "[set_input] : null input tensor");
    assert(input->shape[3] == 1 && "[set_input] : number of filter must be 1");
    assert(input->strides[2]>=2 && "[set_input] : image must have at lease 2*2 size");

    l->InputFMaps = input;

}
 */