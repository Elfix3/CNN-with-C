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

    t->shape[0] = d0;           //Cols
    t->shape[1] = d1;           //Rows
    t->shape[2] = d2;           //Number of feature maps
    t->shape[3] = d3;           //Number of filters


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
    printf("(%zu, %zu, %zu, %zu) : (Cols, Rows, Nfmap, Nfilter)\n",t->shape[0],t->shape[1],t->shape[2],t->shape[3]);
}

void print_tensor4_data(const tensor4_t *t){
    #if DEBUG
        print_tensor4_shape(t);
    #endif
    for(size_t i = 0; i<t->flatten_size; i++){
        if(!(i%t->strides[1])){
            printf("\n");
        }
        if((t->shape[3] != 1) && !(i%t->strides[3]) ){
            printf("\n<--- Filter number %zu --->\n",i/t->strides[3]);
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

void ReLU(tensor4_t *T){

    assert(T != NULL && "Error Null ptr buffer");
    for(size_t i = 0; i<T->flatten_size; i++){
        if (T->datas[i] < 0.0f) {
            T->datas[i] = 0.0f;
        }
    }
}

// bof bof
void addBias_buffer(float *buffer, const float *b, size_t size){
    
    assert(buffer != NULL && "Error Null ptr buffer");
    assert(b != NULL && "Error Null ptr bias");

    for(size_t i = 0; i<size; i++){
        buffer[i] += b[i];
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




void addBias(tensor4_t *t, const float *b){
    //We assume b is n_feature map === shape[2] sized.
    assert(b != NULL && "Error Null ptr bias");
    assert(t != NULL && "Error Null tensor");
    assert(t->shape[3] == 1 && "Error, wrong tensor shape for addbias, only one filter");
    
    size_t idx_image;
    size_t img_size;

    for(size_t i = 0; i<t->shape[2]; i++){
        idx_image = get_t4_idx(t,0,0,i,0);
        img_size = (t->shape[0]*t->shape[1]);
        for(size_t j = idx_image; j< (idx_image + img_size) ; j++){
            t->datas[j] += b[i];
        }
    }
}

//should use tensor4_t
/* void conv(      float *im, size_t im_cols, size_t im_rows,
                float *kernel, size_t k_cols, size_t k_rows,
                float *conv_result, size_t c_cols, size_t c_rows,
                padding_t type)
{
    assert(im != NULL && kernel != NULL && "Error image or kernel is NULL");
    
    memset(conv_result, 0, c_cols*c_rows*sizeof(float));
    
    switch (type){
    
    case SAME:
        
        for(size_t i = 0; i<c_cols*c_rows;i++){
            
            int x = i/c_cols;   //row number
            int y = i%c_cols;   //col number

            for(size_t k = 0; k <k_cols*k_rows; k++){
                int kx = k/k_cols- k_rows/2;   //row number difference
                int ky = k%k_cols- k_cols/2;   //col number difference

                short isOutbound = ((int)(x+kx) < 0) || ((int)(y+ky) < 0) || ((int)(x+kx) >= (int)c_rows) || ((int)(y+ky) >= (int)c_cols); 
                conv_result[i] += kernel[k]*(isOutbound ? 0.0f : im[y+ky + (x+kx)*c_cols]);
            }
            
        }

        break;

    case VALID:

        //#pragma omp parallel for
        for(size_t i = 0; i<c_cols*c_rows;i++){
            int x = i/c_cols;   //row number
            int y = i%c_cols;   //col number
            
            for(size_t k = 0; k<k_cols*k_rows; k++){

                int kx = k/k_cols;   //row number
                int ky = k%k_cols;   //col number   

                conv_result[i] +=kernel[k]*im[(x+kx)*im_cols + (y+ky)];
            }
        
        }

        break;

    default :
        break;
    }

} */

static void cumulate(float *acc, float *source, size_t cumulate_size){
    for(size_t i = 0; i< cumulate_size; i++){
        acc[i] += source[i];
    }
}




static size_t get_conv_dim(const tensor4_t *X, const tensor4_t *K, uint8_t axis, padding_t type){
    assert(axis < 2 && "Error : axis must be 0 (cols) or 1 (rows)");
    assert(K->shape[axis] <= X->shape[axis] && "Error : kernel bigger than input on this axis");
    return  ((type == VALID) ? (X->shape[0]- K->shape[0] + 1) : X->shape[0]);
}


void conv(const tensor4_t *X, const tensor4_t *K, tensor4_t **Z, padding_t padding){
    assert(X != NULL && "Error conv_cumulate : NULL X parameter");
    assert(K != NULL && "Error during conv cumulate : NULL K");
    assert(Z != NULL && "Error during conv cumulate : NULL Z");


    size_t Z_cols;
    size_t Z_rows;
    size_t Z_fmaps = K->shape[3];       //Is the number of filter in K
    size_t Z_batch = X->shape[3];

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

    if((*Z) == NULL){
        (*Z) = (tensor4_t*)init_tensor4(Z_cols, Z_rows, Z_fmaps, Z_batch, NOFILL);
    }


    //Pour chaque image du batch
        //Pour chaque filtre
            //mon accumulateur
            //Pour chaque feature map
                //---> Convolution du Kernel par l'input
                //---> Cumul dans mon accumulateur
}

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
void matvec(const float *X, const tensor4_t *W, float **Z){
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
}

void convNew(const tensor4_t *X, const tensor4_t *K, const float b, tensor4_t **Z, size_t pad_top, size_t pad_bottom, size_t pad_left, size_t pad_right){
    //size
    size_t Z_Cols = X->shape[0]-K->shape[0] + 1 + pad_left + pad_right;
    size_t Z_Rows = X->shape[1]-K->shape[1] + 1 + pad_top + pad_bottom;

    (*Z) = (tensor4_t*)init_tensor4(Z_Cols,Z_Rows,1,1,NOFILL);
    if(*Z == NULL){
        fprintf(stderr, "Error, output conv Z badalloc");
    }


    //Whatever go my #pragma OMP for
    for(size_t rZ = 0; rZ < (*Z)->shape[1]; rZ++){
        for(size_t cZ = 0; cZ < (*Z)->shape[0]; cZ++){
            
            float acc = 0.0f;
            for(size_t rK = 0; rK < K->shape[1]; rK++){
                for(size_t cK = 0; cK < K->shape[0]; cK++){
                    
                    int cX = cK + cZ - pad_left;
                    int rX = rK + rZ - pad_top;
                    //coûteux... on accède juste pas aux K qui sortent ?
                    short isOutBound = (cX < 0) ||(rX <0) || (cX > (int)X->shape[0]-1) ||(rX > (int)X->shape[1]-1); //upper bound check as well
                    acc += get_t4_val(K,cK,rK,0,0)*(isOutBound ? 0.0f : get_t4_val(X,cX,rX,0,0));
                    //printf("%.2f\n", get_t4_val(K,cK,rK,0,0));
                }
                //printf("%.2f\n",acc);
            }
            //---> Ajout biais
            set_t4_val((*Z),acc,cZ,rZ,0,0);
        }
    }

}


