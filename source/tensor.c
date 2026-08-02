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
tensor4_t *init_tensor4(size_t d0, size_t d1, size_t d2, size_t d3, init_type_t type){
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
        printf("%.2f\t",t->datas[i]);
    }
    printf("\n\n");
}

void print_tensor4_mask_data(const tensor4_mask_t *t){
    //assert(t->shape[0]);

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
        printf("%i\t",t->datas[i]);
    }
    printf("\n\n");
}

void ReLU(tensor4_t *t){
    //*mask = (uint8_t*)calloc(t->flatten_size,sizeof(uint8_t));

    for(size_t i = 0; i<t->flatten_size; i++){
        if(t->datas[i] < 0){
            t->datas[i] = 0;
        }
        /* else {
            (*mask)[i] = 1;
        } */
        
    }
}



void MaxPool(tensor4_t *t, tensor4_mask_t **m){
    *m =  (tensor4_mask_t*)malloc(sizeof(tensor4_mask_t));
    (*m)->datas = calloc(t->flatten_size,sizeof(uint8_t));
    
    for(size_t i = 0; i<4;i++){
        (*m)->shape[i] = t->shape[i];
        (*m)->strides[i] = t->strides[i];
    }
    (*m)->flatten_size = t->flatten_size;

    size_t store_index = 0;    
    for(size_t idx_im = 0; idx_im <t->strides[3]; idx_im+= t->strides[2]){
        for(size_t idx_row = 0; idx_row < t->strides[2]; idx_row += 2*t->strides[1]){
            for(size_t idx_col = 0; idx_col < t->strides[1]; idx_col += 2){
                

                float max_val;
                size_t max_index;


                size_t a = idx_im + idx_row + idx_col;
                size_t b = a + 1;
                size_t c = a + t->strides[1];
                size_t d = c + 1;
                

                //Bound check
                if((idx_row + t->strides[1] >= t->strides[2]) && (idx_col+1 >= t->strides[1])){
                    max_val = t->datas[a];
                    max_index = a;

                } else if(idx_row + t->strides[1] >= t->strides[2]){
                    max_val = MAX(t->datas[a],t->datas[b]);
                    max_index = (t->datas[a] >= t->datas[b]) ? a : b;

                } else if((idx_col+1 >= t->strides[1])){
                    max_val = MAX(t->datas[a],t->datas[c]);
                    max_index = (t->datas[a] >= t->datas[c]) ? a :c;

                } else {
                    max_val = max4(t->datas[a], t->datas[b], t->datas[c], t->datas[d]);
                    max_index = maxindex4(t->datas[a], t->datas[b], t->datas[c], t->datas[d],a,b,c,d);
                }

               
                
                
                (*m)->datas[max_index] = (uint8_t)1;
                t->datas[store_index++] = max_val;

                
            }
        }
    }

    t->shape[0] = (t->shape[0]+1)/2;
    t->shape[1] = (t->shape[1]+1)/2;

    t->strides[1] = t->shape[0];
    t->strides[2] = t->shape[0]*t->shape[1];
    t->strides[3] = t->shape[0]*t->shape[1]*t->shape[2];

    t->flatten_size = t->strides[3];


    t->datas = realloc(t->datas, sizeof(float)*t->flatten_size);
    assert(t->datas != NULL && "[MaxPool] realloc failed");
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


void conv(      float *im, size_t im_cols, size_t im_rows,
                float *kernel, size_t k_cols, size_t k_rows,
                float *conv_result, size_t c_cols, size_t c_rows,
                padding_type_t type)
{
    assert(im != NULL && kernel != NULL && "Error image or kernel is NULL");
    
    memset(conv_result,0,c_cols*c_rows);
    
    switch (type){
    
    case SAME:
        
        for(size_t i = 0; i<c_cols*c_rows;i++){
            
            int x = i/c_cols;   //row number
            int y = i%c_cols;   //col number

            for(size_t k = 0; k <k_cols*k_rows; k++){
                int kx = k/k_cols- k_rows/2;   //row number difference
                int ky = k%k_cols- k_cols/2;   //col number difference

                short isOutbound = ((int)(x+kx) < 0) || ((int)(y+ky) < 0) || ((int)(x+kx) >= c_rows) || ((int)(y+ky) >= c_cols); 
                conv_result[i] += kernel[k]*(isOutbound ? 0.0f : im[y+ky + (x+kx)*c_cols]);
            }
            
        }

        break;

    case VALID:


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

}

static void cumulate(float *acc, float *source, size_t cumulate_size){
    for(size_t i = 0; i< cumulate_size; i++){
        acc[i] += source[i];
    }
}


tensor4_t* conv_cumulate(tensor4_t *inputfmaps, tensor4_t *kernels, padding_type_t type){
    assert(inputfmaps != NULL && "Error during conv cumulate : NULL inputFmap");
    assert(kernels != NULL && "Error during conv cumulate : NULL kernels");
    assert(inputfmaps->shape[2] == kernels->shape[2] && "Error : for convolution operation, amount of input feature maps must be the same as amout of kernel for a filter\n");


    size_t outCols = ((type == VALID) ? (inputfmaps->shape[0]- kernels->shape[0] + 1) : inputfmaps->shape[0]);
    size_t outRows = ((type == VALID) ? (inputfmaps->shape[1]- kernels->shape[1] + 1) : inputfmaps->shape[1]);
    size_t flat_conv_size = outCols*outRows;
    size_t n_outputs = kernels->shape[3];       //N_filters is the number of output feature maps

    //attention, allocation non free si plusieurs forwards, penser à ca
    tensor4_t* outputfMaps = init_tensor4(outCols,outRows,n_outputs,1,NOFILL);


    //For each filter
    
    float* conv_buffer = calloc(flat_conv_size,sizeof(float));
    float* acc =  calloc(flat_conv_size,sizeof(float));

    for(size_t idx_filter = 0; idx_filter < kernels->shape[3]; idx_filter++){

        acc = memset(acc,0,sizeof(float)*flat_conv_size);

        for(size_t idx_fmap = 0; idx_fmap < kernels->shape[2]; idx_fmap++){
            
            conv_buffer = memset(conv_buffer,0,sizeof(float)*flat_conv_size);
            conv(
                inputfmaps->datas + get_t4_idx(inputfmaps,0,0,idx_fmap,0),
                inputfmaps->shape[0],   //Cols
                inputfmaps->shape[1],   //Rows
                
                kernels->datas + get_t4_idx(kernels,0,0,idx_fmap,idx_filter),
                kernels->shape[0],      //Cols
                kernels->shape[1],      //Rows
                
                conv_buffer,
                outCols,
                outRows,
                
                type
            );
            
           
            cumulate(acc,conv_buffer, outCols*outRows);
            
        }
        memcpy(outputfMaps->datas + get_t4_idx(outputfMaps,0,0,idx_filter,0) ,acc, flat_conv_size*sizeof(float));
          
    }
    free(conv_buffer);
    free(acc);

    return outputfMaps;
}

/* void print_mask(const uint8_t *mask, size_t d0, size_t d1, size_t d2){
    size_t flat_size = d0*d1*d2;
    for(size_t i = 0; i<flat_size; i++){
        if(!(i%d0)){
            printf("\n");
        }

        if((d2 != 1) && !(i%(d0*d1)) ){
            printf("\nFeature map : %zu\n", ((i%(d0*d1*d2))/(d0*d1)));
        }
        printf("%zu ",mask[i]);
    }
} */

/* float *SoftMax(float input, size_t size){
    return NULL;
}

float SoftMax(float *tab, size_t size, size_t target_index){

    float null_threshold = 0.000001f;
    float max = tab[0];
    float sumExp = 0.0f;

    //Trouve le max pour la stabilité Softmax(Zi-Zmax) = Softmax(Zi)
    for(size_t i = 1; i <size; i++){
        if(exp(tab[i])>max){
            max = tab[i];
        }
    }

    for(size_t i = 0; i<size; i++){
        sumExp += (float)exp(tab[i] - max);
    }

    return (float)exp(tab[target_index] - max) / (float)sumExp;
}
    
 */