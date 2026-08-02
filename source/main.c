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
    ConvLayer *l = init_conv_layer(60,2,2,VALID);
    tensor4_t *mt = init_tensor4(1000,1000,2,1,UNIFORM);
    forward(l, mt);

    /* uint8_t *mask;

    tensor4_t *mt = init_tensor4(4,4,3,1,UNIFORM);
    print_tensor4_data(mt);

    size_t originalsz = mt->flatten_size;
    MaxPool(mt,&mask);
    print_mask(mask,4,4,3);
    print_tensor4_data(mt);
     */

    return 0;
}