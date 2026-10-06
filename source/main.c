#include "stdio.h"
#include "time.h"
#include "omp.h"


#define         FAN_IN              (3*3*4)
#define         MAX(A,B)            (((A)>=(B)) ? (A) : (B) )
#define         MAX4(A,B,C,D)        MAX((MAX((A),(B))),MAX((C),(D)))

//My includes
//#include "datatypes.h"
//#include "layer.h"
#include "tensor/tensor.h"
#include "layers/layer.h"
#include "tensor/ops_laglebra.h"
#include "dataset.h"

int main(int argc, char **argv){

    //Franchement pas dingue, ca m'initalise beaucoup de valeurs négatives parfois -> gradient vanishing :(
    //srand(time(NULL);
    srand(1790321699);          //Pseudo aléatoire biaisé

    //-------------------------------------//
    //----      INFERENCE TEST         ----//
    //-------------------------------------//
    load_CIFAR();
    
    tensor4_t *Input = init_tensor_from_batch(batch1,0,64);
    init_conv_layer_parameter_t p1 = {.input_dimension = Input->dim, .kernel_size = 3, .n_filter = 256, .padding_type = SAME};
    Layer_Wrapper *L1 = make_conv_layer(p1);

    ConvLayer *CL1 = (ConvLayer*)L1->impl; 
    init_conv_layer_parameter_t p2 = {.input_dimension = CL1->P->dim, .kernel_size = 3, .n_filter = 128, .padding_type = SAME};
    Layer_Wrapper *L2 = make_conv_layer(p2);
    
    ConvLayer *CL2 = (ConvLayer*)L2->impl; 
    init_conv_layer_parameter_t p3 = {.input_dimension = CL2->P->dim, .kernel_size = 3, .n_filter = 32, .padding_type = SAME};
    Layer_Wrapper *L3 = make_conv_layer(p3);

    ConvLayer *CL3 = (ConvLayer*)L3->impl; 

    double start = omp_get_wtime();
    printf("Threads : %i\n",omp_get_max_threads());
    forward(L1, Input);
    forward(L2, CL1->P);
    forward(L3, CL2->P);
    double end = omp_get_wtime();
    
    printf("Forward time : %.4f\n",end-start);

    //Ecriture de la sortie (pour comparaison avec winmerge) :
    FILE *fptr;
    fptr = fopen("Results.bin", "w");
    if (fptr == NULL) {
        perror("fopen");
        return 1;
    }

    if(fwrite(CL3->P->datas,sizeof(float), CL3->P->flatten_size,fptr) != CL3->P->flatten_size){
        fclose(fptr);
        LOG_ERROR("Incorrect file writing");
    }
    fclose(fptr);
    LOG_COLOR(GREEN, "Successfull");


    //-------------------------------------//
    //----    END INFERENCE TEST       ----//
    //-------------------------------------//
    

   

    
   

    //-------------------------------------//
    //----       MATMULT TENSOR 2      ----//
    //-------------------------------------//
    /* tensor2_t *X = (tensor2_t*)init_tensor4(1024,1024,1,1,UNIFORM);
    tensor2_t *W = (tensor2_t*)init_tensor4(1024,1024,1,1,UNIFORM);
    
    tensor2_t *Z = NULL;
    allocMatMult(X,W,&Z);
    
    double nstart = omp_get_wtime();
    matmult(X,W,Z);
    double nend= omp_get_wtime();
    printf("Non threaded : %.6f\n",nend-nstart); 

    LOG_INFO("X input :");
    print_tensor4_data((tensor4_t*)X);

    LOG_INFO("W input :");
    print_tensor4_data((tensor4_t*)W);

    

    LOG_INFO("Z output W*X");
    print_tensor4_data((tensor4_t*)Z);  */
    

    //-------------------------------------//
    //----       MATMULT TENSOR 2      ----//
    //-------------------------------------//
    /* tensor4_t *Input = init_tensor4(8,8,4,5,UNIFORM);
    init_conv_layer_parameter_t p = {.input_dimension = get_t4_dim(Input),.kernel_size = 3, .n_filter = 7, .type = VALID};
    ConvLayer *cl = init_conv_layer(p);

    print_tensor4_shape(Input);
    print_tensor4_shape(cl->K);
    print_tensor4_shape(cl->A);
     */
    //assert(0);
    /* tensor4_t *Input = init_tensor4(8,8,4,5,UNIFORM);
    print_tensor4_shape(Input);
    tensor4_dim_t d = get_t4_dim(Input);
    print_t4_dim(&d);

    init_conv_layer_parameter_t p = {.input_dimension = Input->dim, .kernel_size = 3, .n_filter = 7, .padding_type = VALID};
    Layer_Wrapper *L1 = make_conv_layer(p);

    recap(L1);
    
    load_CIFAR();
    show_image(batch2,0); */


    return 0;
}

