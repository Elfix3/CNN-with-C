#include "tensor/tensor.h"

//Some static methods
static inline float max4(float a, float b, float c, float d){
    return MAX(MAX(a,b),MAX(c,d));
}

static inline size_t maxindex4(float a, float b, float c, float d, size_t idxa,size_t idxb, size_t idxc, size_t idxd){
    float max_ab = (a >= b) ?  a : b;
    uint8_t idx_ab = (a >= b) ? idxa : idxb;
    float max_cd = (c >= d) ? c : d;
    uint8_t idx_cd = (c >= d) ? idxc : idxd;

    return (max_ab >= max_cd) ? idx_ab : idx_cd;
}


//----------------------------------//            
//---       init and free        ---//
//----------------------------------//

//** Col, row, n_fmap, n_filter */
tensor4_t *init_tensor4(size_t d0, size_t d1, size_t d2, size_t d3, distribution_t type){
 
    REQUIRE(d0 != 0 && d1 != 0 && d2 != 0 && d3 != 0,"Error, tensor 4 shape must be > 0");
    
    tensor4_t *T = (tensor4_t*)malloc(sizeof(tensor4_t));
    if(!T){LOG_ERROR("Bad Alloc");}

    T->flatten_size = d0*d1*d2*d3;
    T->datas= (float*)malloc(sizeof(float)*T->flatten_size);

    //Dimensions
    T->col = d0;
    T->row = d1;
    T->nmap = d2;
    T->nbatch = d3;         //<---For a Kernel or parameter tensor, nbatch actually represents the number of filter instead

    //Strides
    T->strides[0] = 1;
    T->strides[1] = d0;
    T->strides[2] = d0*d1;
    T->strides[3] = d0*d1*d2;

    size_t fan_in = d0*d1*d2;

    switch (type)
    {
    case NOFILL :
        //Garbage values
        break;

    case ZEROS:
        memset(T->datas, 0, T->flatten_size * sizeof(float));
        break;
    
    case UNIFORM :
        for(size_t i = 0; i<T->flatten_size; i++){
            T->datas[i] =  ((float)rand()/(float)RAND_MAX)*(float)2*HE_LIMIT(fan_in) - (float)HE_LIMIT(fan_in);
        }
        break;
    
    default :
        break;

    }
    
    LOG_VERBOSE("Tensor successfully allocated with shapes (row, cols, nfmap, nbatch) : (%zu %zu %zu %zu)", T->col, T->row, T->nmap, T->nbatch);


    return T;
}

void free_tensor4(tensor4_t **T){
    if (T == NULL || *T == NULL) {
        LOG_WARNING("Useless free tensor, target was already Null");
        return;
    }
    free((*T)->datas);
    free(*T);
    *T = NULL;
}




void print_tensor4_shape(const tensor4_t *T){
    REQUIRE(T != NULL, "Tensor cannot be NULL");
    printf("(%zu, %zu, %zu, %zu) : (Cols, Rows, Nfmap, Nbatch/NFilter)\n",T->shape[0],T->shape[1],T->shape[2],T->shape[3]);
}

void print_tensor4_data(const tensor4_t *T){
    REQUIRE(T != NULL, "Tensor cannot be NULL");
    
    for(size_t i = 0; i<T->flatten_size; i++){
        if(!(i%T->strides[1])){
            printf("\n");
        }
        if((T->nbatch != 1) && !(i%T->strides[3]) ){
            printf("\n<--- Batch :  %zu --->\n",i/T->strides[3]);
        }
        if((T->nmap != 1) && !(i%T->strides[2]) ){
            printf("\nFeature map : %zu\n", ((i%T->strides[3])/T->strides[2]));
        }
        printf("%.4f\t",T->datas[i]);
    }
    printf("\n\n");
}

void print_tensor4_mask(const uint8_t *mask, const tensor4_t *A){
    //assert(t->shape[0]);


    for(size_t i = 0; i<A->flatten_size; i++){
        if(!(i%A->strides[1])){
            printf("\n");
        }
        if((A->shape[3] != 1) && !(i%A->strides[3]) ){
            printf("\n<--- Filter number %zu --->\n",i/A->strides[3]);
        }
        if((A->shape[2] != 1) && !(i%A->strides[2]) ){
            printf("\nFeature map : %zu\n", ((i%A->strides[3])/A->strides[2]));
        }
        printf("%u\t",mask[i]);
    }
    printf("\n\n");
}

uint8_t compareTensor4(const tensor4_t *T1, const tensor4_t *T2){
    
    REQUIRE(T1 != NULL,"Error tensor T1 NULL");
    REQUIRE(T2 != NULL,"Error tensor T2 NULL");
    
    if(T1->flatten_size != T2->flatten_size){
        return 1;
    }
    for(size_t i = 0; i<4; i++){
        if(T1->shape[i] != T2->shape[i]){
            return 1;
        }
    }
    for(size_t i = 0;  i<T1->flatten_size; i++){
        if(T1->datas[i] != T2->datas[i]){
            return 1;
        }
    }
    return 0;
}

//A UTILISER DANS LES FONCTIONS DE CONV (BASEES SUR CUMUL POUR RAPPEL) oui mais si le biais est ajouté avant ca passe nan ?
void reset_t4(tensor4_t *T){    
    REQUIRE(T != NULL && T->datas != NULL, "Error NULL input");

    size_t numChunks = (size_t)omp_get_max_threads();
    size_t numValsPchunk = T->flatten_size / numChunks;
    size_t remainder = T->flatten_size % numChunks;

    //Parallelized reset
    if (T->flatten_size >= 64){
        #pragma omp parallel for
        for(int i = 0; i < (int)numChunks; i++){ 
            memset(T->datas + i*numValsPchunk, 0, numValsPchunk*sizeof(float));
        }
        if (remainder > 0){
            memset(T->datas + numChunks*numValsPchunk, 0, remainder*sizeof(float));
        }
    } else {
        memset(T->datas, 0, sizeof(float) * T->flatten_size);
    }
}

void checkT2(const tensor2_t *T){
    REQUIRE(((tensor4_t*)T)->nmap == 1 && ((tensor4_t*)T)->nbatch == 1,"Error T2 should have only col and row dimension");
}










