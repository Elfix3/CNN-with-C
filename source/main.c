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
    //-----     FORWARD IN LAYER      -----//
    //-------------------------------------//

    //----> INPUT
    /* tensor4_t *X = init_tensor4(5,5,4,2,UNIFORM);
    
    //----> LAYER
    ConvLayer *l = init_conv_layer(3,4,3,SAME);
    
    LOG_INFO("Input X :");
    print_tensor4_shape(X);
    print_tensor4_data(X);
    LOG_INFO("Layer Kernels :");
    print_tensor4_shape(l->K);
    print_tensor4_data(l->K);

    forward(l, X);

    LOG_INFO("OUTPUT :"); */
    //print_tensor4_data(l->A);
    //print_tensor4_shape(l->A);

    //-------------------------------------//
    //-----     TEST FOR POOLING      -----//
    //-------------------------------------//

    /* tensor4_t *A = init_tensor4(4,4,1,1,UNIFORM);
    tensor4_t *P = NULL;
    uint8_t *Pooling_Mask = NULL;

    MaxPool(A,&P,&Pooling_Mask);
    printf("A : \n");
    print_tensor4_data(A);

    printf("P : \n");
    print_tensor4_data(P);

    printf("Masque : \n");
    print_tensor4_mask(Pooling_Mask,A); */


    //-------------------------------------//
    //-----     TEST FOR MATVEC       -----//
    //-------------------------------------//

    /* float X[] = {-1.777, 3.0, 1.0, 2.3};            //n = 4
    
    tensor4_t *W = init_tensor4(4,3,1,1,UNIFORM);
    
    print_tensor4_data(W);
    float *Z = NULL;
  
    matvec(X,W,&Z);
    for(size_t i = 0; i< W->shape[1];i++){
        printf("%.3f \n",Z[i]);
    } */

    //-------------------------------------//
    //---   TEST FOR MATVEC PARALLEL    ---//
    //-------------------------------------//

    /* size_t n = 100000;
    size_t m = 100000;

    //INPUT X
    float *X = malloc(sizeof(float)*n);
    
    double start;
    double end;
    for(size_t i = 0; i < n; i++){X[i] = ((float)rand()/(float)RAND_MAX)-0.5;}
    
    
    //INPUT W
    tensor4_t *W = init_tensor4(n,m,1,1,NOFILL);

    //OUTPUT Z
    float *Z = NULL;

    start = omp_get_wtime();
    matvec(X,W,&Z);
    end = omp_get_wtime();
    printf("Temps : %f secondes\n", end - start);
 */

    //-------------------------------------//
    //-----     TEST FOR NEWCONV      -----//
    //-------------------------------------//
    /* 
    tensor4_t *X = init_tensor4(4,4,2,1, UNIFORM);
    tensor4_t *K = init_tensor4(2,2,2,2,UNIFORM);
    tensor4_t *Z = NULL;
    allocZ(X,K,&Z,SAME);
    
    
    //tensor4_t *ZZ;
    printf("\n\n########\tInput Tensor X :\t########\n\n");
    print_tensor4_data(X);
    printf("\n\n########\tKernel K :\t########\n\n");
    print_tensor4_data(K);
    
    //double start = omp_get_wtime();

    
    convBuffer(X,X->datas,
                K, K->datas,
                Z,Z->datas,
                1,0,1,0);

    print_tensor4_data(Z);     
        assert(0); */
    //Marche en ajoutant les strides !!
    /* convBuffer(X,(X->datas + X->strides[2]),
                K, (K->datas + K->strides[2]),
                Z, (Z->datas + Z->strides[2]),
                1,0,1,0);
 */
    
    //double end = omp_get_wtime();
    //printf("Time diff : %.7f\n",end-start);
    /* printf("\n\n########\tOutput tensor Z :\t########\n\n");
    print_tensor4_data(Z);
    
    tensor4_t *PoolX  = NULL;
    uint8_t *Pmask = NULL;
    MaxPool(Z,&PoolX,&Pmask); */
    //print_tensor4_data(PoolX);
    
    //print_tensor4_data(ZZ);

    //convNew(X,K,&Z,1,1,1,1);
    //print_tensor4_data(Z);

    //convNew(X,K,&Z,2,2,2,2);
    //print_tensor4_data(Z);



    /* convNew(X,K,&Z,1,0,1,0);
    print_tensor4_data(Z);

    convNew(X,K,&Z,0,1,0,1);
    print_tensor4_data(Z);


    convNew(X,K,&Z,1,1,1,1);
    print_tensor4_data(Z); */


    //-------------------------------------//
    //-----     TEST FOR CONV4        -----//
    //-------------------------------------//

    /* tensor4_t *X = init_tensor4(5,5,3,5, UNIFORM);      //<---INPUT
    tensor4_t *K = init_tensor4(3,3,3,2, UNIFORM);      //<---KERNELS
    

    tensor4_t *Z = NULL;                                //<---OUTPUT, should be 3*3*2*5
    free_tensor4(&Z);
    LOG("Tensor X :");
    print_tensor4_data(X);
    LOG("Tensor K :");
    print_tensor4_data(K);
    
    allocZ(X,K,&Z,VALID);
    
    conv4(X,K,&Z,VALID);
    
    LOG("Tensor Z Shape :");
    print_tensor4_shape(Z);

    LOG("Tensor Z :");
    print_tensor4_data(Z); */


    //-------------------------------------//
    //-----     TEST FOR CONV4        -----//
    //-------------------------------------//
    /* tensor4_t *K = init_tensor4(3,3,3,2, UNIFORM);
    LOG_INFO("K tensor :");
    print_tensor4_data(K);

    tensor4_t *KFlip = NULL;    
    
    kernelFlip(K,&KFlip);

    
    LOG_INFO("K flipped :");
    print_tensor4_data(KFlip); */
    
    //-------------------------------------//
    //-----    TEST FOR IMG VIEWER    -----//
    //-------------------------------------//

    //debug_show_image(r_channel, g_channel, b_channel, "input_batch_3_img_7");
   /*  load_CIFAR_Single_Batch("data_batch_2.bin");
    load_CIFAR();
    //load_CIFAR();

    for(int i = 122; i < 122+6; i++){
        show_image(batch1,i);
    } */

    /* show_image(batch2,8954);
    show_image(batch3,8856);
    show_image(batch4,8123);
    show_image(batch5,8442);
    show_image(test,7321); */


    //-------------------------------------//
    //----   LOAD TENSOR FROM BATCH    ----//
    //-------------------------------------//
    /* load_CIFAR();
    tensor4_t *t = init_tensor_from_batch(batch1,0,1);
    print_tensor4_data(t); */



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

    //assert(0);
    return 0;
}

