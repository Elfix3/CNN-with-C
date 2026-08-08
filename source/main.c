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

