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


void SoftMax(tensor4_t *T){
    REQUIRE(T->row == 1 && T->nmap == 1,"Flatten tensor needed");

    //pour chaque batch
    for(size_t b = 0; b<T->nbatch; b++){
        size_t StrideBatch = b*T->strides[3];

        float max = T->datas[StrideBatch];
        for(size_t c = 1; c<T->col; c++){
            if(T->datas[StrideBatch+c] > max){
                max = T->datas[StrideBatch+c];
            }
        }
        float sumExp = 0.0f;
        for(size_t c = 0; c<T->col; c++){
            T->datas[StrideBatch+c] = expf(T->datas[StrideBatch+c] - max);
            sumExp += T->datas[StrideBatch+c];
        }

        for(size_t c = 0; c<T->col; c++){
            T->datas[StrideBatch+c] /= sumExp;
        }
    }
}