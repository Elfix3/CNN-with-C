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

int main(){
    srand(time(NULL));
    

    //-------------------------------------//
    //-----     FORWARD IN LAYER      -----//
    //-------------------------------------//
    /* tensor4_t *X = init_tensor4(8,8,4,1,UNIFORM);
    ConvLayer *l = init_conv_layer(3,4,3,SAME);
    
    print_tensor4_data(l->K);
    print_tensor4_data(X);
    
    forward(l, X); */


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
    
    /* tensor4_t *X = init_tensor4(4,4,2,1, UNIFORM);
    tensor4_t *K = init_tensor4(2,2,2,2,UNIFORM);
    tensor4_t *Z;
    allocZ(X,K,&Z,SAME);
    
    //tensor4_t *ZZ;
    printf("\n\n########\tInput Tensor X :\t########\n\n");
    print_tensor4_data(X);
    printf("\n\n########\tKernel K :\t########\n\n");
    print_tensor4_data(K); */
    
    //double start = omp_get_wtime();

    
    /* convBuffer(X,X->datas,
                K, K->datas,
                Z,Z->datas,
                1,0,1,0);

                
    //Marche en ajoutant les strides !!
    convBuffer(X,(X->datas + X->strides[2]),
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

    tensor4_t *X = init_tensor4(4,4,3,5, UNIFORM);      //<---INPUT
    tensor4_t *K = init_tensor4(2,2,3,2, UNIFORM);      //<---KERNELS
    

    tensor4_t *Z = NULL;                                //<---OUTPUT, should be 3*3*2*5

    LOG("Tensor X :");
    print_tensor4_data(X);
    LOG("Tensor K :");
    print_tensor4_data(K);
    conv4(X,K,&Z,VALID);
    


    LOG("Tensor Z Shape :");
    print_tensor4_shape(Z);

    LOG("Tensor Z :");
    print_tensor4_data(Z);
    return 0;
}

