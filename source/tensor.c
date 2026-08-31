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
    assert(d0 != 0 &&
            d1 != 0 &&
            d2  != 0 &&
            d3 != 0 && "Error, shapes must be >=1");
    
    
    tensor4_t *t = (tensor4_t*)malloc(sizeof(tensor4_t));
    t->flatten_size = d0*d1*d2*d3;
    t->datas= (float*)malloc(sizeof(float)*t->flatten_size);

    //Dimensions
    t->col = d0;
    t->row = d1;
    t->nmap = d2;
    t->nbatch = d3;         //<---For a Kernel or parameter tensor, nbatch actually represents the number of filter instead

    //Strides
    t->strides[0] = 1;
    t->strides[1] = d0;
    t->strides[2] = d0*d1;
    t->strides[3] = d0*d1*d2;

    size_t fan_in = d0 * d1 *d2;

    switch (type)
    {
    case NOFILL :
        //Garbage values
        break;

    case ZEROS:
        memset(t->datas, 0, t->flatten_size * sizeof(float));
        break;
    
    case UNIFORM :
        for(size_t i = 0; i<t->flatten_size; i++){
            t->datas[i] =  ((float)rand()/(float)RAND_MAX)*(float)2*HE_LIMIT(fan_in) - (float)HE_LIMIT(fan_in);
        }
        break;
    
    default :
        break;

    }
    return t;
}

void free_tensor4(tensor4_t **t){
    if (t == NULL || *t == NULL) {
        return;
    }
    free((*t)->datas);
    free(*t);
    *t = NULL;
}

void print_tensor4_shape(const tensor4_t *t){
    printf("(%zu, %zu, %zu, %zu) : (Cols, Rows, Nfmap, Nbatch/NFilter)\n",t->shape[0],t->shape[1],t->shape[2],t->shape[3]);
}

void print_tensor4_data(const tensor4_t *t){
    #if VERBOSE
        print_tensor4_shape(t);
    #endif
    for(size_t i = 0; i<t->flatten_size; i++){
        if(!(i%t->strides[1])){
            printf("\n");
        }
        if((t->shape[3] != 1) && !(i%t->strides[3]) ){
            printf("\n<--- Batch :  %zu --->\n",i/t->strides[3]);
        }
        if((t->shape[2] != 1) && !(i%t->strides[2]) ){
            printf("\nFeature map : %zu\n", ((i%t->strides[3])/t->strides[2]));
        }
        printf("%.4f\t",t->datas[i]);
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

void addBias(tensor4_t *t, const float *b){

}

void ReLU(tensor4_t *T)
{

    assert(T != NULL && "Error Null ptr buffer");
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


void MaxPool(const tensor4_t *A, tensor4_t **P, uint8_t **Pooling_Mask){
    assert(A != NULL);

    //Checks if P needs allocation
    uint8_t needs_realloc = (*P) == NULL || (*P)->shape[0] != (A->shape[0]+1)/2 ||            
    (*P)->shape[1] != (A->shape[1]+1)/2 || (*P)->shape[2] != A->shape[2] || (*P)->shape[3] != A->shape[3];

    //Proceed to P and pooling Mask reallocation
    if(needs_realloc){
        if((*P) != NULL){free_tensor4(P);}
        (*P) = (tensor4_t*)init_tensor4((A->shape[0]+1)/2,(A->shape[1]+1)/2,A->shape[2],A->shape[3],NOFILL);
        
        if((*Pooling_Mask) != NULL) free(*Pooling_Mask);
        (*Pooling_Mask) = calloc(A->flatten_size, sizeof(uint8_t));
    }


    size_t store_index = 0;    


    // ON DOIT ITERER SUR LE BATCH !!
    //for each image

    //foreach FEATURE MAP NOT IMAGE
    for(size_t idx_im = 0; idx_im <A->strides[3]; idx_im+= A->strides[2]){
        for(size_t idx_row = 0; idx_row < A->strides[2]; idx_row += 2*A->strides[1]){
            for(size_t idx_col = 0; idx_col < A->strides[1]; idx_col += 2){

                float max_val;
                size_t max_index;

                size_t a = idx_im + idx_row + idx_col;
                size_t b = a + 1;
                size_t c = a + A->strides[1];
                size_t d = c + 1;
                

                //Bound check
                if((idx_row + A->strides[1] >= A->strides[2]) && (idx_col+1 >= A->strides[1])){
                    max_val = A->datas[a];
                    max_index = a;

                } else if(idx_row + A->strides[1] >= A->strides[2]){
                    max_val = MAX(A->datas[a],A->datas[b]);
                    max_index = (A->datas[a] >= A->datas[b]) ? a : b;

                } else if((idx_col+1 >= A->strides[1])){
                    max_val = MAX(A->datas[a],A->datas[c]);
                    max_index = (A->datas[a] >= A->datas[c]) ? a :c;

                } else {
                    max_val = max4(A->datas[a], A->datas[b], A->datas[c], A->datas[d]);
                    max_index = maxindex4(A->datas[a], A->datas[b], A->datas[c], A->datas[d],a,b,c,d);
                }
                (*Pooling_Mask)[max_index] = (uint8_t)1;
                (*P)->datas[store_index++] = max_val;
                
            }
        }
    }

    /* t->shape[0] = (t->shape[0]+1)/2;
    t->shape[1] = (t->shape[1]+1)/2;

    t->strides[1] = t->shape[0];
    t->strides[2] = t->shape[0]*t->shape[1];
    t->strides[3] = t->shape[0]*t->shape[1]*t->shape[2];

    t->flatten_size = t->strides[3];


    t->datas = realloc(t->datas, sizeof(float)*t->flatten_size);
    assert(t->datas != NULL && "[MaxPool] realloc failed"); */
}



void outputConv(const tensor4_t *X, const tensor4_t *K, tensor4_t **Z, padding_t padding){
    
    if(*(Z) != NULL){
        LOG("Output tensor Z already allocated");
    }
    
    size_t Z_cols, Z_rows;

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
    (*Z) = (tensor4_t*)init_tensor4(Z_cols,Z_rows,1,1,NOFILL);
}


void allocZ(const tensor4_t *X, const tensor4_t *K, tensor4_t **Z, padding_t padding){
    
    if(*(Z) == NULL){
        //LOG("Allocation of the tensor");
        //mettre en verbose
    } else {
        return;                                 //Alloc only used one time
    }

    size_t Z_cols, Z_rows;

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
    printf("%zu %zu %zu %zu\n",Z_cols,Z_rows,K->nbatch,X->nbatch);
    (*Z) = (tensor4_t*)init_tensor4(Z_cols,Z_rows,K->shape[3],X->shape[3],ZEROS);
}


void getPadding(size_t *t, size_t *b, size_t *l, size_t *r, const tensor4_t *K, padding_t padding){
    switch (padding){
        
    case VALID :
        *t = 0; *b = 0; *l = 0; *r = 0;
        break;
    case SAME :
        //manage odd parity for kernel should be something like K->row/2  and other same (K->row-1)/2

        //I GUESS VRO
    case FULL :
        *t = K->shape[1]-1; *b = K->shape[1]-1; *l = K->shape[0]-1; *r = K->shape[0]-1;
    default:
        break;
    }
}

void conv4(const tensor4_t *X, const tensor4_t *K, tensor4_t **Z, padding_t padding){
    assert(X != NULL && "Error conv_cumulate : NULL X parameter");
    assert(K != NULL && "Error during conv cumulate : NULL K");
    assert(Z != NULL && "Error during conv cumulate : NULL Z");

    //TODO calculer les offset en fonction du padding
    size_t pad_top, pad_bottom, pad_left, pad_right;
    getPadding(&pad_top, &pad_bottom, &pad_left, &pad_right,K,padding);
    LOG_DEBUG("%zu %zu %zu %zu",pad_top,pad_bottom, pad_left, pad_right);
    allocZ(X,K,Z,padding);
    LOG_DEBUG("Z : stide : %zu",(*Z)->strides[2]);

    //for each batche example
    size_t idX = 0;
    size_t idK = 0;
    size_t idZ = 0;

    size_t idxBatch = 0;

    //For each Batch
    for(size_t b = 0; b<X->shape[3]; b++){

        //For each filter
        for(size_t f = 0; f < K->shape[3]; f++){
            
            //For each fmap
            for(size_t m = 0; m < X->shape[2]; m++){
                
                //Holly hardcodded padding
                LOG_DEBUG("Calling CONV BUFFER with : idX = %zu \t\t idK = %zu \t\t idZ = %zu \t\t idBatch = %zu",idX,idK,idZ,idxBatch);

                convBuffer(X,(X->datas + idX),
                            K, (K->datas + idK),
                            *Z, ((*Z)->datas + idZ),
                            pad_top,pad_bottom,pad_left,pad_right);
                
                idX += X->strides[2];
                idK += K->strides[2];
            }
            //idK += K->strides[3];
            idZ += (*Z)->strides[2];
            idX = idxBatch;
        }
        idK = 0;
        idX += X->strides[3];
        idxBatch += X->strides[3];
    }
    //Pour chaque image du batch
        //Pour chaque filtre
            //mon accumulateur
            //Pour chaque feature map
                //---> Convolution du Kernel par l'input
                //---> Cumul dans mon accumulateur
}



//######################################
//######## TRAVAIL TEMPORAIRE #########
//######################################

/* void conv_cumulate(const tensor4_t *X, const tensor4_t *K, const padding_t type, tensor4_t **Z)
{
    //X has dimension : x*y*n*m (n is the number of feature maps per image, m is the number of images in a batch)
    assert(X != NULL && "Error conv_cumulate : NULL X parameter");
    assert(K != NULL && "Error during conv cumulate : NULL K");
    assert(Z != NULL && "Error during conv cumulate : NULL Z");

    assert(X->shape[2] == K->shape[2]);

    size_t Z_cols = get_conv_dim(X,K,0,type);
    size_t Z_rows = get_conv_dim(X,K,1,type);    
    size_t Z_fmaps = K->shape[3];
    size_t Z_batch = X->shape[3];

    uint8_t needs_alloc =   (*Z) == NULL ||
                            (*Z)->shape[0] != Z_cols || 
                            (*Z)->shape[1] != Z_rows ||
                            (*Z)->shape[2] != Z_fmaps || 
                            (*Z)->shape[3] != Z_batch;
    if(needs_alloc){
        
        if (*Z != NULL) {free_tensor4(Z);}
        (*Z) = (tensor4_t*)init_tensor4(Z_cols, Z_rows, Z_fmaps, Z_batch, NOFILL);
        LOG("Output tensor allocated, calculation starts");

    } else {
        LOG("Output tensor already alocated, calculation starts");
    }

    //Allocates conv buffers
    size_t flat_conv_size = (*Z)->strides[2];
    float* acc =  calloc(flat_conv_size,sizeof(float));
    float* conv_buffer = calloc(flat_conv_size,sizeof(float));
    

    
    //#pragma omp parallel for

    //For each image in the batch
    for(size_t idx_batch = 0; idx_batch < X->shape[3]; idx_batch ++){

        //For each filter
        for(size_t idx_filter = 0; idx_filter < K->shape[3]; idx_filter++){
        
            //Accumulator reset
            acc = memset(acc,0,sizeof(float)*flat_conv_size);

        //For each Feature map
            for(size_t idx_fmap = 0; idx_fmap < K->shape[2]; idx_fmap++){
                memset(conv_buffer,0,sizeof(float)*flat_conv_size);
                conv(
                    X->datas + get_t4_idx(X,0,0,idx_fmap,idx_batch),          //Feature map indexée par la feature map et le numéro du batch
                    X->shape[0],
                    X->shape[1],

                    K->datas + get_t4_idx(K,0,0,idx_fmap,idx_filter),        //Kernel indexé par la feature map et le numéro du filtre
                    K->shape[0],
                    K->shape[1],
                    
                    conv_buffer,        
                    Z_cols,
                    Z_rows, 
                    type
                );
                cumulate(acc,conv_buffer, flat_conv_size);
            }
        memcpy((*Z)->datas+get_t4_idx((*Z),0,0,idx_filter,idx_batch), acc, flat_conv_size*sizeof(float));
        }
    }

    free(conv_buffer);
    free(acc);
} */




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



 //Doit être une conv intermédiaire
void convBuffer(const tensor4_t *X, const float *dataX, const tensor4_t *K, const float *dataK, const tensor4_t *Z, float *dataZ, size_t pad_top, size_t pad_bottom, size_t pad_left, size_t pad_right){

    //TENSOR Z ALREADY ALLOCATED
    if(Z == NULL){
        fprintf(stderr, "Error, output conv Z badalloc");
    }
    assert(X->nmap == K->nmap && "Error, input number of feature maps should match the number of feature maps inside a filter"); //<---- Here K->nbatch is the number of filter

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