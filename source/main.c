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
    
    tensor4_t *X = init_tensor4(1000,1000,1,1, UNIFORM);
    tensor4_t *K = init_tensor4(100,100,1,1,UNIFORM);
    tensor4_t *Z;
    tensor4_t *ZZ;
    //print_tensor4_data(X);
    //print_tensor4_data(K);
    
    double start = omp_get_wtime();
    convOld(X,K,0.0f,&Z,1,1,1,1);
    double end = omp_get_wtime();
    printf("Time diff : %.7f\n",end-start);
    //print_tensor4_data(Z);
    
    start = omp_get_wtime();
    convNew(X,K,0.0f,&ZZ,1,1,1,1);
    end = omp_get_wtime();
    printf("Time diff : %.7f\n",end-start);
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

    //printf("%i\n",(int)((size_t)6 - (size_t)12));

    return 0;
}

