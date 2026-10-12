#include "layers/dense_layer.h"


//méthode 
static tensor4_t t4_to_t2(const tensor4_t *T){
    tensor4_t temp = *T;
    reshape(&temp, (tensor4_dim_t){.col = T->col*T->row*T->nmap, .row = 1, .nmap = 1, .nbatch = T->nbatch});
    return temp;
}


DenseLayer *init_dense_layerOld(size_t n, size_t m, activation_t type){
    DenseLayer *l = malloc(sizeof(DenseLayer));

    l->W = init_tensor4(n,m,1,1, UNIFORM);
    l->b = calloc(m,sizeof(float));
    l->activation_type = type;

    //faut faire tout le bordel d'allouer les buffers aussi

    return l;

}


DenseLayer *init_dense_layer(tensor4_dim_t input_dim ,init_dense_layer_parameter_t param){
    DenseLayer *l = malloc(sizeof(DenseLayer));
    if(!l){LOG_ERROR("Bad alloc");}
    //ca ou l'inverse :shrug
    size_t ninput = input_dim.col*input_dim.row*input_dim.nmap;

    l->W = init_tensor4(param.m_neurons,ninput,1,1,UNIFORM); //revoir les dimensions
    
    l->b = calloc(param.m_neurons,sizeof(float));
    
    l->activation_type = param.activation_type;


    l->A = init_tensor4(param.m_neurons,1,1,input_dim.nbatch,NOFILL);
    //l->W = init_tensor4(param.m_neurons,input_dim.col,1,1,UNIFORM);
    
    return l;
}

void free_dense(DenseLayer *l){
    free_tensor4(&l->W);
    free(l->b);
    free_tensor4(&l->dW);
    free(l->dB);
    free_tensor4(&l->A);
    free_tensor4(&l->dX);

    //Ajouter free pour ADAM plus tard
}

void clean_dense(DenseLayer *l){
    //Important, préserve tous les paramètres des couches

}

void recap_dense(const DenseLayer *l){
    LOG_INFO_HEADER("Dense layer recap");

    //Input size ?? 

    //ou W->row
    LOG_INFO("%-12s : %zu", "Number of neurons",l->W->col);
    
    LOG_INFO("%-12s : ReLU", "Activation");
}

tensor4_dim_t output_dim_dense(const DenseLayer *l){
    return get_t4_dim(l->A);
}

const tensor4_t *forward_dense(DenseLayer *l, const tensor4_t *X){
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
   return NULL;
}

const tensor4_t *backward_dense(DenseLayer *l, const tensor4_t *dA){
    return NULL;
}

void update_dense(DenseLayer *l){
    //
}
