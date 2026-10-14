#include "stdio.h"
#include "time.h"
#include "omp.h"


//moche a virer

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
    /* load_CIFAR();
    
    tensor4_t *Input = init_tensor_from_batch(batch1,0,4);
    init_conv_layer_parameter_t p1 = {.Input = Input, .kernel_size = 3, .n_filter = 5, .padding_type = SAME};
    layer_params_t PL =  {.type = CONV_LAYER, .conv = p1};
    Layer_Wrapper *L1 = createLayer(&PL);
/* 
    ConvLayer *CL1 = (ConvLayer*)L1->impl; 
    init_conv_layer_parameter_t p2 = {.Input = L1->P, .kernel_size = 3, .n_filter = 128, .padding_type = SAME};
    Layer_Wrapper *L2 = make_conv_layer(p2); */
    /* 
    ConvLayer *CL2 = (ConvLayer*)L2->impl; 
    init_conv_layer_parameter_t p3 = {.input_dimension = CL2->P->dim, .kernel_size = 3, .n_filter = 32, .padding_type = SAME};
    Layer_Wrapper *L3 = make_conv_layer(p3); */

    //ConvLayer *CL3 = (ConvLayer*)L3->impl; 

    /*double start = omp_get_wtime();
    printf("Threads : %i\n",omp_get_max_threads());
    forward(L1, Input); *
    /* forward(L2, CL1->P);
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
    LOG_COLOR(GREEN, "Successfull"); */


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

    //-------------------------------------//
    //----       Dense layer recap     ----//
    //-------------------------------------//
    /* DenseLayer *dl =  init_dense_layer(4,6,UNIFORM);
    print_t4_dim(dl->W); */



    /* tensor4_t *T = init_tensor4(5,5,2,4,UNIFORM);
    print_tensor4_data(T);
    
    reshape(T,(tensor4_dim_t){.col = 8, .row = 5, .nmap = 1, .nbatch = 5});

    print_tensor4_data(T); */


    /* tensor4_t *T = init_tensor4(5,5,2,4,UNIFORM);
    print_tensor4_data(T); */

    /* tensor4_t *A = init_tensor4(4,4,2,6,UNIFORM);
    tensor4_t *P = init_tensor4(2,2,2,6,NOFILL);
    uint8_t *Pmask = calloc(A->flatten_size, sizeof(uint8_t));
    MaxPool(A,P,Pmask);
    LOG_INFO("Original A tensor");
    print_tensor4_data(A);
    
    LOG_INFO("Pooled P tensor");
    print_tensor4_data(P);
    
    
    LOG_INFO("Pooling MASK");
    print_tensor4_mask(Pmask,A);
    assert(0); */
    
    //-------------------------------------//
    //----        MISC LAYER TEST      ----//
    //-------------------------------------//

    //tensor4_t *I = init_tensor4(4,4,2,1, UNIFORM);

    if(argc > 1){
        if(strcmp(argv[1],"test") == 0){
            goto TEST;
        }
    } 
    goto STOP;


    TEST :
    
    //On peut améliorer

    init_conv_layer_parameter_t p1 = {.kernel_size = 3, .n_filter = 7};
    init_conv_layer_parameter_t p2 = {.kernel_size = 3, .n_filter = 6};
    init_conv_layer_parameter_t p3 = {.kernel_size = 3, .n_filter = 5};
    //init_dense_layer_parameter_t p4 = {.m_neurons = 16, .activation_type = ReLU};
    //init_dense_layer_parameter_t p5 = {.m_neurons = 10, .activation_type = SOFTMAX};

    layer_params_t lp1 = {.type = CONV_LAYER, .conv = p1};
    layer_params_t lp2 = {.type = CONV_LAYER, .conv = p2};
    layer_params_t lp3 = {.type = CONV_LAYER, .conv = p3};
    //layer_params_t lp4 = {.type = DENSE_LAYER, .dense = p4};

    tensor4_t *I = init_tensor4(32,32,2,3, UNIFORM);
    


    Layer_Wrapper *L1  = createLayer(I->dim, lp1);
    //recap(L1);
    
    Layer_Wrapper *L2 = createLayer(output_dim(L1), lp2);
    //recap(L2);

    Layer_Wrapper *L3 = createLayer(output_dim(L2), lp3);
    //recap(L3);

    
    tensor4_t* A = init_tensor4(4,1,1,3,UNIFORM);
    print_tensor4_data(A);

    SoftMaxNew(A);
    print_tensor4_data(A);






    

    STOP :
    LOG_WARNING("Program end");
    return 0;
}

