#include "stdio.h"
#include "time.h"
#include "omp.h"


#define         FAN_IN              (3*3*4)
#define         MAX(A,B)            (((A)>=(B)) ? (A) : (B) )
#define         MAX4(A,B,C,D)        MAX((MAX((A),(B))),MAX((C),(D)))

//My includes
//#include "datatypes.h"
//#include "layer.h"
#include "tensor.h"
#include "conv_layer.h"
#include "img_view.h"

int main(int argc, char **argv){

    //Franchement pas dingue, ca m'initalise beaucoup de valeurs négatives parfois -> gradient vanishing :(
    //srand(time(NULL);
    srand(1790321699);          //Pseudo aléatoire biaisé
    

    //-------------------------------------//
    //----      MAXPOOL THREADED       ----//
    //-------------------------------------//
    /* tensor4_t *t = init_tensor4(5,5,2,2,UNIFORM);
    tensor4_t *Pref = NULL;
    tensor4_t *Presult = NULL;
    uint8_t *MaskRef = NULL;
    uint8_t *MaskResult = NULL;

    allocP(t,&Pref, &MaskRef);
    allocP(t,&Presult, &MaskResult);
    
    print_tensor4_data(t);

    MaxPool(t,Pref,MaskRef);
    print_tensor4_data(Pref);
    MaxPoolThreaded(t,Presult,MaskResult);
    print_tensor4_data(Presult);
    

    if(compareTensor4(Pref, Presult)){
        LOG_COLOR(RED,"Error Different tensors");
    } else {
        LOG_COLOR(GREEN,"OK tensors are the same");
    } */
    //assert(0);
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
    //print_tensor4_data(L3->P);

    /* print_tensor4_shape(Input);
    print_tensor4_shape(L1->P);
    print_tensor4_shape(L2->P);
    print_tensor4_shape(L3->P); */

    //print_tensor4_data(L3->P);

    //-------------------------------------//
    //--- Threaded MAXPOOL COMPARAISON  ---//
    //-------------------------------------//
    //tensor4_t *X = init_tensor4(8,8,3,5,UNIFORM);
    /* tensor4_t *t = init_tensor4(128,128,128,128,UNIFORM);
    tensor4_t *Pref = NULL;
    tensor4_t *Presult = NULL;
    uint8_t *MaskRef = NULL;
    uint8_t *MaskResult = NULL;

    allocP(t,&Pref, &MaskRef);
    allocP(t,&Presult, &MaskResult);
    
    //print_tensor4_data(t);

    double start = omp_get_wtime();
    MaxPool(t,Pref,MaskRef);
    double end = omp_get_wtime();
    printf("Non threaded time : %.6f\n",end-start);
    
    //print_tensor4_data(Pref);


    start = omp_get_wtime();
    MaxPoolThreaded(t,Presult,MaskResult);
    end = omp_get_wtime();
    printf("Threaded time : %.6f\n",end-start);
    //print_tensor4_data(Presult);
    

    if(compareTensor4(Pref, Presult)){
        LOG_COLOR(RED,"Error Different tensors");
    } else {
        LOG_COLOR(GREEN,"OK tensors are the same");
    }
 */
    //-------------------------------------//
    //----  THREADED CONV4 COMPARAISON ----//
    //-------------------------------------//
    /* padding_t pad = FULL;
    
    tensor4_t *X = init_tensor4(8,8,3,5,UNIFORM);
    tensor4_t *W = init_tensor4(3,3,3,2,UNIFORM);

    tensor4_t *ZRef = NULL;
    tensor4_t *ZThreaded = NULL;
    
    allocZ(X,W,&ZRef,pad);
    allocZ(X,W,&ZThreaded,pad);
    
    conv4(X,W,ZRef,pad);
    conv4threaded(X,W,ZThreaded,pad);

    LOG_COLOR(YELLOW,"ZREF");
    print_tensor4_shape(ZRef);
    //print_tensor4_data(ZRef);

    LOG_COLOR(YELLOW,"ZTHREAD");
    print_tensor4_shape(ZThreaded);
    //print_tensor4_data(ZThreaded);

    if(compareTensor4(ZRef,ZThreaded)){
        LOG_COLOR(RED,"Error Different tensors");
    } else {
        LOG_COLOR(GREEN,"OK tensors are the same");
    } */


    //-------------------------------------//
    //----  Threaded kernel Flip       ----//
    //-------------------------------------//
    /* const tensor4_t *W = init_tensor4(256,256,128,128,NOFILL);
    tensor4_t *WFlipped = NULL;
    tensor4_t *WFlippedThreaded = NULL;
    
    LOG_INFO("K original  : ");
    //print_tensor4_data(W);

    double start = omp_get_wtime();
    kernelFlip(W,&WFlipped);
    double end = omp_get_wtime();
    printf("Non threaded time : %.6f\n",end-start);

    double nstart = omp_get_wtime();
    kernelFlipThreaded(W,&WFlippedThreaded);
    double nend = omp_get_wtime();
    printf("Threaded time : %.6f\n",nend-nstart);


    LOG_INFO("Kflipped ref : ");
    //print_tensor4_data(WFlipped);

    LOG_INFO("Kflipped threaded : ");
    //print_tensor4_data(WFlippedThreaded);
    
    if(compareTensor4(WFlipped, WFlippedThreaded)){
        LOG_COLOR(RED,"Error Different tensors");
    } else {
        LOG_COLOR(GREEN,"OK tensors are the same");
    } */


    //-------------------------------------//
    //----     Threaded setBias        ----//
    //-------------------------------------//
    /* tensor4_t *Z = init_tensor4(128,128,64,64,UNIFORM);
    tensor4_t *Zother = init_tensor4(128,128,64,64,UNIFORM);
    float *b = malloc(sizeof(float)*64*64);

    for(size_t i = 0; i <64*64; i++){
        b[i] = 6.0f;
    }
    
    double start = omp_get_wtime();
    setBias(Z,b);
    double end = omp_get_wtime();
    printf("Non threaded time : %.6f\n",end-start);

    double nstart = omp_get_wtime();
    setBiasThreaded(Zother,b);
    double nend = omp_get_wtime();
    printf("Threaded time : %.6f\n",nend-nstart);

    //print_tensor4_data(Zother);

    //print_tensor4_data(Z);
    if(compareTensor4(Z, Zother)){
        LOG_COLOR(RED,"Error Different tensors");
    } else {
        LOG_COLOR(GREEN,"OK tensors are the same");
    } */

    //-------------------------------------//
    //----        RESET TENSOR 4       ----//
    //-------------------------------------//

    /* tensor4_t *T = init_tensor4(128,128,256,256,NOFILL);
    //print_tensor4_data(T);
    double nstart = omp_get_wtime();
    reset_t4(T);
    double nend = omp_get_wtime();
    //print_tensor4_data(T);
    printf("Non threaded : %.6f\n",nend-nstart); */

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

/* 
    LOG_INFO("X input :");
    print_tensor4_data((tensor4_t*)X);

    LOG_INFO("W input :");
    print_tensor4_data((tensor4_t*)W);

    

    LOG_INFO("Z output W*X");
    print_tensor4_data((tensor4_t*)Z); */
    
    //assert(0);
    return 0;
}

