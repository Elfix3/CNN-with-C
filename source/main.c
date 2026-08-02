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
    

    tensor4_t *im = init_tensor4(4,6,1,1,UNIFORM);
    print_tensor4_data(im);
    //<--- Forward example --->//
    /* ConvLayer *l = init_conv_layer(2,6,2,SAME);
    tensor4_t *mt = init_tensor4(3,3,2,1,UNIFORM); */
    /* print_tensor4_data(l->InputFMaps);

    
    MaxPool(l->InputFMaps,&l->Pooling_Mask);
    print_tensor4_data(l->InputFMaps);
    print_mask(l->Pooling_Mask,3,3,2);

    forward(l, mt); */

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