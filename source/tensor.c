#include "tensor.h"

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

void convBuffer(const tensor4_t *X, const float *dataX, const tensor4_t *K, const float *dataK, const tensor4_t *Z, float *dataZ, size_t pad_top, size_t pad_bottom, size_t pad_left, size_t pad_right){

    //TENSOR Z ALREADY ALLOCATED
    REQUIRE(Z != NULL,"Z output of convBuffer not allocated");
    REQUIRE(X->nmap == K->nmap, "Number of feature map in input must match number of feature map in kernel");

    for(size_t rZ = 0; rZ < Z->shape[1]; rZ++){
        int rKlow =  pad_top - rZ; 
        int rKhigh = pad_top - rZ + X->shape[1];
        size_t rKStart = MAX(0,rKlow);
        size_t rKEnd   = MIN(K->shape[1],(size_t)MAX(0, rKhigh));       

        for(size_t cZ = 0; cZ < Z->shape[0]; cZ++){
        
            int cKlow =  pad_left - cZ; 
            int cKhigh = pad_left - cZ + X->shape[0];
            size_t cKStart = MAX(0,cKlow);
            size_t cKEnd   = MIN(K->shape[0], (size_t)MAX(0, cKhigh));  
            
            
            float acc = 0.0f;

            for(size_t rK = rKStart; rK < rKEnd; rK++){
                size_t rX = rK + rZ - pad_top;
                for(size_t cK = cKStart; cK < cKEnd ; cK++){
                    size_t cX = cK + cZ - pad_left;
                    
                    //Optimiser les accès
                    //acc += get_t4_val(K,cK,rK,0,0)* get_t4_val(X,cX,rX,0,0);
                    acc +=dataK[cK+ rK*K->strides[1]]*dataX[cX + rX*X->strides[1]];
                }
            }
            dataZ[cZ + rZ*Z->strides[1]] += acc;
            
            
        }
    }

}


void addBias(tensor4_t *t, const float *b){
    // ??
}

void ReLU(tensor4_t *T){
    //Throw some pragma tard
    REQUIRE(T != NULL, "Input tensor ptr is NULL");
    for(size_t i = 0; i<T->flatten_size; i++){
        if (T->datas[i] < 0.0f) {
            T->datas[i] = 0.0f;
        }
    }
}

void SoftMax(float *tab, size_t size){

    float max = tab[0];    
    for(size_t i = 1; i<size; i++){
        if(tab[i]>max){
            max = tab[i];
        }
    }
    
    float sumExp = 0;
    for(size_t i = 0; i<size; i++){
        tab[i] = expf(tab[i] - max);
        sumExp += tab[i];
    }

    for(size_t i = 0; i<size; i++){
        tab[i] /= sumExp;
    }
}


void MaxPool(const tensor4_t *A, tensor4_t *P, uint8_t *Pooling_Mask){
    assert(A != NULL);

    REQUIRE(A != NULL, "Tensor A ptr cannot be NULL");
    REQUIRE(P != NULL, "Tensor P ptr cannot be NULL");
    REQUIRE(Pooling_Mask != NULL, "Pooling mask ptr cannot be NULL");


    size_t store_index = 0;    
    //Non thread safe
    for(size_t idx_batch = 0; idx_batch < A->nbatch; idx_batch++){
        size_t batch_offset = idx_batch * A->strides[3]; 

        for(size_t idx_im = 0; idx_im < A->strides[3]; idx_im += A->strides[2]){
            for(size_t idx_row = 0; idx_row < A->strides[2]; idx_row += 2*A->strides[1]){
                for(size_t idx_col = 0; idx_col < A->strides[1]; idx_col += 2){

                    float max_val;
                    size_t max_index;

                    size_t a = batch_offset + idx_im + idx_row + idx_col;
                    size_t b = a + 1;
                    size_t c = a + A->strides[1];
                    size_t d = c + 1;

                    if((idx_row + A->strides[1] >= A->strides[2]) && (idx_col+1 >= A->strides[1])){
                        max_val = A->datas[a];
                        max_index = a;
                    } else if(idx_row + A->strides[1] >= A->strides[2]){
                        max_val = MAX(A->datas[a],A->datas[b]);
                        max_index = (A->datas[a] >= A->datas[b]) ? a : b;
                    } else if(idx_col+1 >= A->strides[1]){
                        max_val = MAX(A->datas[a],A->datas[c]);
                        max_index = (A->datas[a] >= A->datas[c]) ? a : c;
                    } else {
                        max_val = max4(A->datas[a], A->datas[b], A->datas[c], A->datas[d]);
                        max_index = maxindex4(A->datas[a], A->datas[b], A->datas[c], A->datas[d], a, b, c, d);
                    }

                    Pooling_Mask[max_index] = (uint8_t)1;
                    P->datas[store_index++] = max_val;
                }
            }
        }
    }
}



void allocZ(const tensor4_t *X, const tensor4_t *K, tensor4_t **Z, padding_t padding){

    REQUIRE(X != NULL, "Tensor X ptr is NULL");
    REQUIRE(K != NULL, "Tensor K ptr is NULL");
    REQUIRE(Z != NULL, "Tensor Z ptr is NULL");


   
    size_t Z_cols, Z_rows, Z_nmaps, Z_batch;

    switch (padding){
    case SAME :
        Z_cols = X->shape[0];
        Z_rows = X->shape[1];
 
    break;
    case VALID :
        assert(X->shape[0] >= K->shape[0] &&"Error, kernel too wide for a VALID conv");
        assert(X->shape[1] >= K->shape[1] && "Error, kernel too long for a VALID conv");
        Z_cols = X->shape[0] - K->shape[0] + 1;
        Z_rows = X->shape[1] - K->shape[1] + 1;
    break;
    case FULL :
        Z_cols = X->shape[0] + K->shape[0] - 1;
        Z_rows = X->shape[1] + K->shape[1] - 1;
    break;

    default:
        assert(0 && "Error: invalid padding mode");
        break;
    }
    Z_nmaps = K->nbatch;
    Z_batch = X->nbatch;
    //printf("%zu %zu %zu %zu\n",Z_cols,Z_rows,K->nbatch,X->nbatch);
    
    
    uint8_t needs_alloc  = 0;

    if(*(Z) == NULL){
        needs_alloc = 1;
    } else if(Z_cols != (*Z)->col || Z_rows != (*Z)->row || Z_nmaps != (*Z)->nmap || Z_batch != (*Z)->nbatch){
        free_tensor4(Z);
        needs_alloc = 1;
    }

    if(needs_alloc){
        LOG_VERBOSE("New allocation");
        (*Z) = (tensor4_t*)init_tensor4(Z_cols,Z_rows,K->shape[3],X->shape[3],ZEROS);
    } else {
        LOG_VERBOSE("Previous allocation keeped");
    }
}

void allocP(const tensor4_t *A, tensor4_t **P, uint8_t **Pooling_Mask){
    REQUIRE(A != NULL, "Tensor A ptr is NULL");
    REQUIRE(P != NULL, "Tensor P ptr is NULL");
    REQUIRE(Pooling_Mask != NULL, "Mask is NULL");

    size_t P_cols = (A->col+1) /2;
    size_t P_rows = (A->row+1) /2;

    uint8_t needs_alloc  = 0;

    if(*(P) == NULL){
        needs_alloc = 1;
    } else {
        uint8_t dimensionChanged = ((*P)->shape[0] != P_cols)|| (*P)->shape[1] != P_rows ||
        (*P)->nmap != A->nmap || (*P)->nbatch != A->nbatch;
        if(dimensionChanged){
            needs_alloc = 1;
            free_tensor4(P);
        }
    }
    
    if(needs_alloc){
        LOG_VERBOSE("New allocation");
        (*P) = (tensor4_t*)init_tensor4(P_cols,P_rows,A->nmap,A->nbatch,ZEROS);
        (*Pooling_Mask) = calloc(A->flatten_size, sizeof(uint8_t));
    } else {
        LOG_VERBOSE("Previous allocation keeped");
    }

}

void getPadding(size_t *t, size_t *b, size_t *l, size_t *r, const tensor4_t *K, padding_t padding){
    
    REQUIRE(K != NULL, "Tensor K ptr is NULL");
    switch (padding){
        
    case VALID :
        *t = 0; *b = 0; *l = 0; *r = 0;
        break;
    case SAME :
        *t = K->row/2;*b = (K->row-1)/2; *l = K->col/2; *r = (K->col-1)/2;
        break;
    case FULL :
        *t = K->shape[1]-1; *b = K->shape[1]-1; *l = K->shape[0]-1; *r = K->shape[0]-1;
        break;
    default:
        break;
    }
}

void conv4(const tensor4_t *X, const tensor4_t *K, tensor4_t *Z, padding_t padding){
    assert(X != NULL && "Error conv_cumulate : NULL X parameter");
    assert(K != NULL && "Error during conv cumulate : NULL K");
    assert(Z != NULL && "Error during conv cumulate : NULL Z");

    //TODO calculer les offset en fonction du padding
    
    
    size_t pad_top, pad_bottom, pad_left, pad_right;
    getPadding(&pad_top, &pad_bottom, &pad_left, &pad_right,K,padding);
    
    
    LOG_DEBUG("%zu %zu %zu %zu",pad_top,pad_bottom, pad_left, pad_right);
    //allocZ(X,K,Z,padding);
    //LOG_DEBUG("Z : stide : %zu",(*Z)->strides[2]);

    //for each batche example
    size_t idX = 0;
    size_t idK = 0;
    size_t idZ = 0;


    //anti pattern pour du threading ?  oui
    size_t idxBatch = 0;

    //For each Batch
    for(size_t b = 0; b<X->shape[3]; b++){

        //For each filter
        for(size_t f = 0; f < K->shape[3]; f++){
            
            //For each fmap
            for(size_t m = 0; m < X->shape[2]; m++){
                
                LOG_DEBUG("Calling CONV BUFFER with : idX = %zu \t\t idK = %zu \t\t idZ = %zu \t\t idBatch = %zu",idX,idK,idZ,idxBatch);

                convBuffer(X,(X->datas + idX),
                            K, (K->datas + idK),
                            Z, (Z->datas + idZ),
                            pad_top,pad_bottom,pad_left,pad_right);
                
                idX += X->strides[2];
                idK += K->strides[2];
            }
            //idK += K->strides[3];
            idZ += Z->strides[2];
            idX = idxBatch;
        }
        idK = 0;
        idX += X->strides[3];
        idxBatch += X->strides[3];
    }
}

void kernelFlip(const tensor4_t *K, tensor4_t **Kflipped){
    REQUIRE(K != NULL, "Cannot flipp NULL Kernel");
    REQUIRE(Kflipped != NULL, "Cannot flipp NULL Kernel");
    
    uint8_t needs_alloc  = 0;

    //Check la nécessité d'allouer
    if((*Kflipped) == NULL){
        needs_alloc = 1;
    } else if(K->col != (*Kflipped)->col || K->row != (*Kflipped)->row ||  K->nmap != (*Kflipped)->nmap || K->nbatch != (*Kflipped)->nbatch){
        //Libère le précédent tenseur s'il existait déjà
        free_tensor4(Kflipped);
        needs_alloc = 1;
    }

    //Alloue
    if(needs_alloc){
        LOG_VERBOSE("Tensor Kflip allocated");
        (*Kflipped) = init_tensor4(K->col, K->row, K->nmap, K->nbatch,NOFILL);        
    }
    

    //Non thread friendly
    size_t idK = 0;
    //for each filter
    for(size_t f = 0; f<K->nbatch; f++){

        //for each featuremap
        for(size_t m = 0; m <K->nmap; m++){
            
            //Flip operation
            for(size_t i = 0; i< K->strides[2]; i++){
                
                (*Kflipped)->datas[idK+i] = K->datas[idK+K->strides[2]-i-1];
                //printf("%.4f ",K->datas[idK+K->strides[2]-i-1]);
                
            }
            //printf("\n ");
            idK += K->strides[2];
        }
    }
}

//A TESTER
void setBias(tensor4_t *Z, float *b){
    REQUIRE(Z != NULL, "Input ptr Z cannoc be NULL");
    REQUIRE(b != NULL, "b pointers unitialized");

    //Non thread friendly
    size_t idZ = 0;
    for(size_t nb = 0; nb <Z->nbatch; nb++){
        for(size_t m = 0; m<Z->nmap; m++){
            for(size_t i = 0; i< Z->strides[2]; i++){
                Z->datas[idZ+i] = b[m];
            }

            idZ += Z->strides[2];
        }
    }

}

//######################################
//######## TRAVAIL TEMPORAIRE #########
//######################################

//modify this to have tensor X as input
/* void matvec(const float *X, const tensor4_t *W, float **Z){
    //W is size(m,n), X is size(n,1) => Z is size 
    
    assert(X != NULL && W != NULL && "Error matvec : null parameter");
    assert(W->shape[2] == 1 && W->shape[3] == 1 && "Error, unmaching dimension for a matrix");
    
    if((*Z) == NULL){
        *Z = calloc(W->shape[1],sizeof(float));
    } else {
        memset(*Z,0,W->shape[1]*sizeof(float));
    }

    #pragma omp parallel for
    for(size_t m = 0; m < W->shape[1]; m++){
        for(size_t n = 0; n < W->shape[0]; n++){
            (*Z)[m] += get_t4_val(W,n,m,0,0)*X[n];
        }    
    }
} */
