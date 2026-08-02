#include "datatypes.h"



void init_floats(float *fp, size_t n_float, init_type_t type){
    
    switch (type)
    {
    case ZEROS:
        for(size_t i = 0; i <n_float; i++){
            fp[i] = 0.0f; //temporary, will be 0.0f
        }
        break;
    
    default:
        break;
    }
}

Array *init_array(size_t rows, size_t cols, init_type_t type, size_t fan_in)
{

    //###########################//
    //#### Error managmement ####//
    //###########################//

    assert((cols > 0 && rows >0) && "Error cols and rows must be superior to 0\n");
    

    Array *a = (Array *)malloc(sizeof(Array));
    a->cols = cols;
    a->rows = rows;
    a->datas = (float *)malloc(sizeof(float)*rows*cols);
    switch (type)
    {
    case ZEROS:
        for(size_t i = 0; i<cols*rows; i++){
            a->datas[i] =  0.0f;
        }
        break;
    case UNIFORM:
        assert((fan_in > 8) && "Error, fan_in must be  9 minmum\n");
        for(size_t i = 0; i<cols*rows; i++){
            a->datas[i] =  ((float)rand()/(float)RAND_MAX)*(float)2*HE_LIMIT(fan_in) - (float)HE_LIMIT(fan_in);
        }
         
        break;
    default:
        for(size_t i = 0; i<cols*rows; i++){
            a->datas[i] =  0.0f;
        }
        break;
    }
    return a;
}

void print_array(const Array *a){
    assert(a != NULL && "Error : array unitialized\n");
    assert((a->cols > 0 && a->rows >0) && "Error cols and rows must be superior to 0\n");
    
    for(size_t i = 0; i<a->cols*a->rows; i++){
        printf("%.6f\t\t",a->datas[i]);
        if((i%a->cols == a->cols-1)){
            printf("\n");
        }
    }
    printf("\n");
}

float get_val(Array *a ,size_t i, size_t j){
    assert(a != NULL && "Error array unitialized !\n");
    assert(j < a->cols && "Error out of index j on cols !\n");
    assert(i < a->rows && "Error out of index i on rows !\n");
    return a->datas[j+a->cols*i];
}

void free_array(Array **a){
    assert(a != NULL &&
        (*a) != NULL && "Error free_array : array unitialized\n");
    
    free((*a)->datas);
    free(*a);
    *a = NULL;
}

void ReLU(Array *a){
    assert(a != NULL && "Error ReLU : array unitialized\n");
    assert((a->cols > 0 && a->rows >0) && "Error cols and rows must be superior to 0\n");
    
    for(size_t i = 0; i<a->cols*a->rows; i++){
        a->datas[i] = (a->datas[i] >= 0 ? a->datas[i] : 0.0f);
    }
}

void set_array(Array *a, init_type_t type, size_t fan_in){

    size_t cols = a->cols;
    size_t rows = a->rows;

    assert((cols > 0 && rows >0) && "Error cols and rows must be superior to 0\n");
    assert((fan_in > 8) && "Error, fan_in must be  9 minmum\n");
    
    switch (type)
    {
    case ZEROS:
        for(size_t i = 0; i<cols*rows; i++){
            a->datas[i] =  0.0f;
        }
        break;
    case UNIFORM:
        for(size_t i = 0; i<cols*rows; i++){
            a->datas[i] =  ((float)rand()/(float)RAND_MAX)*(float)2*HE_LIMIT(fan_in) - (float)HE_LIMIT(fan_in);
        }
         
        break;
    default:
        for(size_t i = 0; i<cols*rows; i++){
            a->datas[i] =  0.0f;
        }
        break;
    }
}

Array *padded(Array *a, size_t n_padding){
    size_t new_rows = a->rows + 2 * n_padding;
    size_t new_cols = a->cols + 2 * n_padding;
    Array *p = init_array(new_rows, new_cols, ZEROS, 10);

    for (size_t i = 0; i < a->rows; i++) {
        float *padded_row = p->datas + (i + n_padding)*new_cols + n_padding;            //Source
        float *origin_row = a->datas + i*new_cols; 
        memcpy(padded_row, origin_row, a->cols*sizeof(float));
    }
    /* for(size_t i = 0; i<padded->cols * padded->rows; i++){
        //Condition bootom and top i>= n_padding*(n_padding+a->)
        short rowCondition = i>= n_padding*(n_padding*2 + a->cols) && i<(n_padding+a->rows)*(n_padding*2 + a->cols) ;
        short colCondition = i%(n_padding*2+a->cols) >= n_padding && i%(n_padding*2+a->cols) < n_padding+a->cols;  
        
       if(rowCondition && colCondition){
        padded->datas[i] = a->datas[aIndex++];
       }
       memcpy();

    } */
    return p;
}

void cumulate(Array *destination, Array **source){
    assert(destination!= NULL && "Error during cumulate : accumulator is undefined\n");
    assert(source!= NULL && "Error during cumulate : term is undefined\n");
    assert(destination->cols == (*source)->cols && "Error, accumulator and term must have same cols number\n");
    assert(destination->rows == (*source)->rows && "Error, accumulator and term must have same row number\n");
    
    for(size_t i = 0; i< destination->cols*destination->rows; i++){
        destination->datas[i]+= (*source)->datas[i];
    }
    free_array(source);
}
void MaxPooling(Array **a){
    //Specifications on max pooling : 2*2 stride of 2
    assert(*a != NULL && "Error : array unitialized\n");
    
    assert(((*a)->cols > 1 && (*a)->rows > 1) && "Error cols and rows must be superior to 1\n");
    
    size_t pcols = (*a)->cols/2;
    size_t prows = (*a)->rows/2;

    Array *a_pool = (Array *)malloc(sizeof(Array));
    assert(a_pool != NULL && "Error : badalloc\n");

    a_pool->cols = pcols;
    a_pool->rows = prows;
    a_pool->datas = (float *)malloc(sizeof(float)*prows*pcols);

    //for each pool operation
    for(size_t i = 0; i<prows*pcols; i++){
        
        //a_pool->datas[i];
        //a->datas[(i%a->cols)*2 + (i/a->cols)*2];

        //size_t index  = ((i*2)/a->cols)*a->cols*2;
        size_t index  =((i*2)%((*a)->cols)) + ((i*2)/(*a)->cols)*(*a)->cols*2;
        //printf("%.2f\n",a->datas[index]/*,index*/);

        float max = -(FLT_MAX);
        for(size_t j = 0; j<2*2 ; j++){
            
            if((*a)->datas[(index+j%2)+(j/2)*(*a)->cols] > max){
                max = (*a)->datas[(index+j%2)+(j/2)*(*a)->cols];
            }

        }
        //printf("%.2f\n", max);

        a_pool->datas[i] = max;
    }
    free_array(a);
    *a = a_pool;
}

Array *convold(Array *fmap, Array *kernel, padding_type_t padding){
    //print_array(fmap);
    //print_array(kernel);

    //Convolution dimensions
    if (kernel->cols > fmap->cols || kernel->rows > fmap->rows) {
        printf("Error, wrong kernel (%zu : %zu) feature map (%zu : %zu) dimensions\n",
        kernel->rows, kernel->cols, fmap->rows, fmap->cols);
    return NULL;
    }
    size_t c_col = fmap->cols - kernel->cols + 1;
    size_t c_rows = fmap->rows - kernel->rows + 1;

    //Structure build

    //replace by INIT ARAY
    Array *conv = malloc(sizeof(Array));
    conv->cols = (size_t) c_col;
    conv ->rows = (size_t) c_rows;
    conv->datas = (float*) malloc(sizeof(float)*c_rows * c_col );

    //For each kernel nested
    for(size_t i = 0; i <c_rows; i++){
        for(size_t j = 0; j<c_col; j++){
            float val = 0;
            //For each value of the kernel
            for(size_t n = 0; n<kernel->rows; n++){
                for(size_t m = 0; m<kernel->cols; m++){
                    val += get_val(kernel,n,m)*get_val(fmap,i+n,j+m);
                    //printf("%.2f * %.2f + ", get_val(kernel,n,m),get_val(fmap,i+n,j+m));
                }
            }
            //printf("\n");
            conv->datas[j+i*c_col] = val;

        }
    }

    return conv;
}

Array *conv(Array *fmap, Array *kernel, padding_type_t padding){
    //Kind of useless
    size_t output_cols;
    size_t output_rows;
    //We can keep this though
    Array *output = NULL;

    switch (padding){
        
    case SAME:
        //We keep same output dimensions
        output_cols = fmap->cols;
        output_rows = fmap->rows;

        output = init_array(output_rows, output_cols,ZEROS,0);
        output->cols = output_cols;
        output->rows = output_rows;
        
        //We iterate on the output map
        
        for(size_t i = 0; i<output_cols*output_rows;i++){
            
            int x = i/output_cols;   //row number
            int y = i%output_cols;   //col number
            
            
            for(size_t k = 0; k<kernel->cols*kernel->rows; k++){
                
                int kx = k/kernel->cols-kernel->rows/2;   //row number difference
                int ky = k%kernel->cols-kernel->cols/2;   //col number difference

                short isOutbound = ((int)(x+kx) < 0) || ((int)(y+ky) < 0) || ((int)(x+kx) >= output_rows) || ((int)(y+ky) >= output_cols); 
                output->datas[i] += kernel->datas[k]*(isOutbound ? 0.0f : fmap->datas[y+ky + (x+kx)*output_cols]);
                //printf("%.2f * %.2f +\t ",kernel->datas[k],(isOutbound ? 0.0f : fmap->datas[y+ky + (x+kx)*output_cols]));

            }
            //printf(" = %.2f\n",output->datas[i]);
        }
        
        break;

    case VALID :
        output_cols = fmap->cols - kernel->cols + 1;
        output_rows = fmap->rows - kernel->rows + 1;

        output = init_array(output_rows, output_cols,ZEROS,0);
        output->cols = output_cols;
        output->rows = output_rows;

        double debut_seq = omp_get_wtime();
        /* #ifdef USE_OMP
            #pragma omp parallel for
        #endif */
        for(size_t i = 0; i<output_cols*output_rows;i++){
            int x = i/output_cols;   //row number
            int y = i%output_cols;   //col number
            
            


            for(size_t k = 0; k<kernel->cols*kernel->rows; k++){

                int kx = k/kernel->cols;   //row number
                int ky = k%kernel->cols;   //col number   

                output->datas[i] +=kernel->datas[k]*fmap->datas[(x+kx)*fmap->cols + (y+ky)];
            }

        }
        /* double fin_seq = omp_get_wtime();
        printf("Sequentiel : %f secondes\n", fin_seq - debut_seq); */
        
        break;

    default:
        break;
    }

    return output;
}


void add_scalar(Array *output_map, float b){
    assert(output_map != NULL && "Error during add scalar : accumulator is undefined\n");
    for(size_t i = 0; i< output_map->cols*output_map->rows; i++){
        output_map->datas[i] += b;
    }
}

//###########################//
//####    Array of Array ####//
//###########################//

Array_Arr *init_array_arr(size_t n_array, size_t rows, size_t cols){
    Array_Arr *aa = (Array_Arr *)malloc(sizeof(Array_Arr));
    aa->n_array = n_array;
    aa->a = (Array **)malloc(sizeof(Array*)*n_array);
    for(size_t i = 0; i<n_array; i++){
        aa->a[i] = init_array(rows,cols, UNIFORM, 27);
    }
    return aa;
}

void free_array_arr(Array_Arr **aa){

    assert(aa != NULL &&
            *aa != NULL &&
            (*aa)->a != NULL &&
            "Error ReLU : array unitialized\n");

    for(size_t i = 0; i< (*aa)->n_array; i++){
        free_array(&(*aa)->a[i]);
    }
    free((*aa)->a);
    free(*aa);
    *aa = NULL;
}



void print_array_arr(const Array_Arr *aa){
    assert(aa != NULL && "Error null array arr");
    
    for(size_t i = 0; i< aa->n_array; i++){
        printf("Array %i: \n",i);
        print_array(aa->a[i]);
    }
    

}


//###########################//
//####    Array Bank     ####//
//###########################//

Array_Bank *init_array_bank(size_t n_arr_arr, size_t n_array, size_t rows, size_t cols){
    Array_Bank *ab = (Array_Bank*)malloc(sizeof(Array_Bank));
    ab->n_arr_arr = n_arr_arr;
    ab->aa = (Array_Arr**)malloc(sizeof(Array_Arr*)*n_arr_arr);
    for(size_t i = 0; i<n_arr_arr; i++){
        ab->aa[i] = init_array_arr(n_array, rows, cols);
    }
    return ab;
}

void free_array_bank(Array_Bank **ab){
    assert(ab != NULL &&
            *ab != NULL &&
            (*ab)->aa != NULL &&
            (*(*ab)->aa)->a != NULL &&
            "Error Array bank incorrectly initalized\n");
    for(size_t i = 0; i<(*ab)->n_arr_arr; i++){
        free_array_arr(&(*ab)->aa[i]);
    }
    free((*ab)->aa);
    free(*ab);
    *ab = NULL;
}

void print_array_bank(const Array_Bank *ab){
    for(size_t i = 0; i< ab->n_arr_arr;i++){
       printf("\n\n<---      Filter : %i     --->\n",i); 
        print_array_arr(ab->aa[i]);
    }
}

//###########################//
//####    Kernel bank    ####//
//###########################//

//AKA array of feature maps
/* KernelBank *init_kernel_bank(size_t n_featuremap){
    return 
}
 */