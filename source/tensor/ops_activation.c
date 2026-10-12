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


void SoftMaxOld(float *tab, size_t size){

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


void SoftMax(tensor4_t *T){
    REQUIRE(T->row == 1 && T->nmap == 1,"Flatten tensor needed");

    //pour chaque batch
    for(size_t b = 0; b<T->nbatch; b++){
        float max = get_t4_val(T,0,0,0,b);
        
        for(size_t c = 1; c<T->col; c++){
            if(T->datas[get_t4_idx(T,c,0,0,b)] > max){
                max = T->datas[get_t4_idx(T,c,0,0,b)];
            }
        }

        float sumExp = 0.0f;
        for(size_t c = 0; c<T->col; c++){
            T->datas[get_t4_idx(T,c,0,0,b)] = expf(T->datas[get_t4_idx(T,c,0,0,b)] - max);
            sumExp += T->datas[get_t4_idx(T,c,0,0,b)];
        }

        for(size_t c = 0; c<T->col; c++){
            T->datas[get_t4_idx(T,c,0,0,b)] /= sumExp;
        }
    }
        //softmaxer la ligne

}