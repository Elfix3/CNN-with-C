#include "tensor/ops_activation.h"

void ReLU(tensor4_t *T){
    REQUIRE(T != NULL, "Input tensor ptr is NULL");
    
    #pragma omp parallel for
    for(size_t i = 0; i<T->flatten_size; i++){
        if (T->datas[i] < 0.0f) {
            T->datas[i] = 0.0f;
        }
    }
}

void SoftMax(float *tab, size_t size){

    float max = tab[0];    
    for(size_t i = 1; i<size; i++){
        if(tab[i]>max){
            max = tab[i];
        }
    }
    
    float sumExp = 0;
    for(size_t i = 0; i<size; i++){
        tab[i] = expf(tab[i] - max);
        sumExp += tab[i];
    }

    for(size_t i = 0; i<size; i++){
        tab[i] /= sumExp;
    }
}