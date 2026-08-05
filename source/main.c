#include "stdio.h"
#include "time.h"

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
    
    /* tensor4_t *X = init_tensor4(3,3,3,1,UNIFORM);
    tensor4_t *K = init_tensor4(2,2,3,2,UNIFORM);
    tensor4_t *Z = NULL;

    printf("Input X : \n");
    print_tensor4_data(X);

    printf("Kernels K : \n");
    print_tensor4_data(K);

    conv_cumulate(X,K,SAME,&Z);
    print_tensor4_data(Z); */


    /* tensor4_t *X2 = init_tensor4(3,3,3,1,UNIFORM);
    printf("Input X2 : \n");
    print_tensor4_data(X2);

    conv_cumulate(X2,K,VALID,&Z);
    print_tensor4_data(Z); */

    /* tensor4_t *X2 = init_tensor4(7,7,4,1,UNIFORM);

    conv_cumulate(X2,K,VALID,&Z);
    print_tensor4_data(Z); */
    
    //tensor4_t *im = init_tensor4(4,6,1,1,UNIFORM);
    //print_tensor4_data(im);
    //<--- Forward example --->//
   /*  ConvLayer *l = init_conv_layer(3,1,32,SAME);
    tensor4_t *mt = init_tensor4(28,28,1,1,UNIFORM);
    forward(l, mt);
    print_tensor4_shape(l->A);

    ConvLayer *y = init_conv_layer(3,l->A->shape[2],32,SAME);
    forward(y, l->A);
    print_tensor4_shape(y->A); */


    //-------------------------------------//
    //-----     FORWARD IN LAYER      -----//
    //-------------------------------------//
    tensor4_t *X = init_tensor4(8,8,4,1,UNIFORM);
    ConvLayer *l = init_conv_layer(3,4,3,SAME);
    
    print_tensor4_data(l->K);
    print_tensor4_data(X);
    
    forward(l, X);


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
    return 0;
}

