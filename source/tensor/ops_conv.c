#include "tensor/ops_conv.h"

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


void convBuffer(const tensor4_t *X, const float *restrict dataX, const tensor4_t *K, const float *restrict dataK, const tensor4_t *restrict Z,
    float *dataZ, size_t pad_top, size_t pad_bottom, size_t pad_left, size_t pad_right) {

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

                const float *Krow = dataK + rK*K->strides[1];
                const float *Xrow = dataX + rX*X->strides[1];
                for(size_t cK = cKStart; cK < cKEnd ; cK++){
                    size_t cX = cK + cZ - pad_left;
                    acc +=Krow[cK]*Xrow[cX];
                }
            }
            dataZ[cZ + rZ*Z->strides[1]] += acc;
            
            
        }
    }
}

void conv4(const tensor4_t *X, const tensor4_t *K, tensor4_t *Z, padding_t padding){
    assert(X != NULL && "Error conv_cumulate : NULL X parameter");
    assert(K != NULL && "Error during conv cumulate : NULL K");
    assert(Z != NULL && "Error during conv cumulate : NULL Z");


    //          /!\ pas de vérif des bornes correctes ??
    
    size_t pad_top, pad_bottom, pad_left, pad_right;
    getPadding(&pad_top, &pad_bottom, &pad_left, &pad_right,K,padding);
    
    
    LOG_DEBUG("%zu %zu %zu %zu",pad_top,pad_bottom, pad_left, pad_right);

    #pragma omp parallel for
    for(size_t b = 0; b<X->nbatch; b++){
        
        size_t XStrideBatch = b*X->strides[3];
        size_t ZStrideBatch = b*Z->strides[3];
        //For each filter
        for(size_t f = 0; f < K->nbatch; f++){
            
            size_t KStrideFilter = f*K->strides[3];
            size_t ZStrideFilter = f*Z->strides[2];
            
            for(size_t m = 0; m < X->nmap; m++){
                
                size_t idX = XStrideBatch + m*X->strides[2];
                size_t idK = KStrideFilter + m*K->strides[2];
                size_t idZ = ZStrideBatch + ZStrideFilter;

                LOG_DEBUG("Calling CONV BUFFER with : idX = %zu \t\t idK = %zu \t\t idZ = %zu \t\t idBatch = %zu",idX,idK,idZ,b);

                convBuffer(X,(X->datas + idX),
                            K, (K->datas + idK),
                            Z, (Z->datas + idZ),
                            pad_top,pad_bottom,pad_left,pad_right);
                
            }
            
        }
    }
}

void setBias(tensor4_t *Z, const float *b){
    REQUIRE(Z != NULL, "Input ptr Z cannot be NULL");
    REQUIRE(b != NULL, "b pointers unitialized");

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


//Faire dépendre des dimensions de K plutôt ?
tensor4_dim_t get_conv_output_dim(tensor4_dim_t X_dim, tensor4_dim_t K_dim, padding_t pad){
    tensor4_dim_t A_dim = { .nmap = K_dim.nfilter, .nbatch = X_dim.nbatch };

    switch (pad) {
    case SAME:
        A_dim.col = X_dim.col;
        A_dim.row = X_dim.row;
        break;
    case VALID:
        REQUIRE(X_dim.col >= K_dim.col && X_dim.row >= K_dim.row, "Kernel too large for VALID conv");
        A_dim.col = X_dim.col - K_dim.col + 1;
        A_dim.row = X_dim.row - K_dim.row + 1;
        break;
    case FULL:
        A_dim.col = X_dim.col + K_dim.col - 1;
        A_dim.row = X_dim.row + K_dim.row - 1;
        break;
    default:
        LOG_ERROR("Incorrect padding");
    }

    return A_dim;
}

tensor4_dim_t get_maxpool_output_dim(tensor4_dim_t A_dim){
    tensor4_dim_t P_dim = {.col = (A_dim.row+1)/2, .row = (A_dim.row+1)/2,
        .nmap = A_dim.nmap, .nbatch = A_dim.nbatch};
    
    return P_dim;
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
    

    
    //for each filter
    #pragma omp parallel for
    for(size_t f = 0; f< K->nbatch; f++){
        size_t filterStride = f*K->strides[3];
        //for each featuremap
        for(size_t m = 0; m <K->nmap; m++){
            size_t fmapStride = m*K->strides[2];
            //Flip operation
            for(size_t i = 0; i< K->strides[2]; i++){
                
                size_t idK = filterStride+fmapStride;
                (*Kflipped)->datas[idK+i] = K->datas[idK+K->strides[2]-i-1];
                
            }
        }
    }
}

void MaxPool(const tensor4_t *A, tensor4_t *P, uint8_t *Pooling_Mask){
    assert(A != NULL);

    REQUIRE(A != NULL, "Tensor A ptr cannot be NULL");
    REQUIRE(P != NULL, "Tensor P ptr cannot be NULL");
    REQUIRE(Pooling_Mask != NULL, "Pooling mask ptr cannot be NULL");

   
    
    #pragma omp parallel for collapse(2)
    for(size_t batch = 0; batch < P->nbatch; batch++){
        
        size_t AStrideBatch = batch*A->strides[3];
        size_t PStrideBatch = batch*P->strides[3];

        for(size_t filter = 0; filter < P->nmap; filter++){
            size_t AStrideFilter = filter*A->strides[2]; 
            size_t PStrideFilter = filter*P->strides[2]; 
            
            for(size_t pRow= 0; pRow < P->row; pRow++){
                size_t AStrideRow = pRow*2*A->strides[1];
                
                for(size_t pCol = 0; pCol < P->col; pCol++){

                    float max_val;
                    size_t max_index;

                    size_t a = AStrideBatch + AStrideFilter + AStrideRow + pCol*2;
                    size_t b = a + 1;
                    size_t c = a + A->strides[1];
                    size_t d = c + 1;

                    if((AStrideRow + A->strides[1] >= A->strides[2]) && ((pCol*2)+1 >= A->strides[1])){
                        max_val = A->datas[a];
                        max_index = a;
                    }else if(AStrideRow + A->strides[1] >= A->strides[2]){
                        max_val = MAX(A->datas[a],A->datas[b]);
                        max_index = (A->datas[a] >= A->datas[b]) ? a : b;
                    } else if((pCol*2)+1 >= A->strides[1]){
                        max_val = MAX(A->datas[a],A->datas[c]);
                        max_index = (A->datas[a] >= A->datas[c]) ? a : c;
                    } else {
                        max_val = max4(A->datas[a], A->datas[b], A->datas[c], A->datas[d]);
                        max_index = maxindex4(A->datas[a], A->datas[b], A->datas[c], A->datas[d], a, b, c, d);
                    }
                    //Pooling_Mask[max_index] = (uint8_t)1;
                    
                    
                    //printf("\nBatch : %zu Fmap : %zu Row : %zu Col : %zu",idx_batch, idx_im, idx_row, idx_col);
                    size_t store_index = PStrideBatch + PStrideFilter + pRow*P->strides[1] + pCol;
                    //printf("\nstore index : %zu\n",store_index);
                    P->datas[store_index] = max_val;
                    //assert(store_index<10000);
                }
            }
        }
    }
    
}