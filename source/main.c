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
#include "layers/conv_layer.h"
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
    tensor4_t *Input = init_tensor_from_batch(batch1,0,64);
    ConvLayer *L1 = init_conv_layer(3,3,256,SAME);
    ConvLayer *L2 = init_conv_layer(3,256,128,SAME);
    ConvLayer *L3 = init_conv_layer(3,128,32,SAME);

    double start = omp_get_wtime();
    printf("Threads : %i\n",omp_get_max_threads());
    forward(L1, Input);
    forward(L2, L1->P);
    forward(L3, L2->P);
    double end = omp_get_wtime();
    
    printf("Forward time : %.4f\n",end-start);

    //Ecriture de la sortie (pour comparaison avec winmerge) :
    FILE *fptr;
    fptr = fopen("Results.bin", "w");
    if (fptr == NULL) {
        perror("fopen");
        return 1;
    }

    if(fwrite(L3->P->datas,sizeof(float), L3->P->flatten_size,fptr) != L3->P->flatten_size){
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
    tensor2_t *X = (tensor2_t*)init_tensor4(1024,1024,1,1,UNIFORM);
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
    print_tensor4_data((tensor4_t*)Z); 
    
    //assert(0);
    return 0;
}

