#include "tensor/ops_laglebra.h"

void allocMatMult(const tensor4_t *X, const tensor4_t *W, tensor4_t **Z){
    // HORRIBLE MAIS ON DEVELOPPE LES ALGOS DABORD
    REQUIRE(X != NULL, "Null X tensor");
    REQUIRE(W != NULL, "Null W tensor");
    REQUIRE(Z != NULL, "Null Z tensor");

    checkT2(X);
    checkT2(W);

    if(*(Z) == NULL){
        (*Z) = (tensor4_t*)init_tensor4(X->col, W->row,1,1,NOFILL);
    }
    //printf("%zu %zu\n",(*Z)->col, (*Z)->row);
    checkT2(*Z);
}

void matmult(const tensor4_t *X, const tensor4_t *W, tensor4_t *Z){
    REQUIRE(X->row == W->col, "Dimensions of matmult X W dont match");
    REQUIRE(X->col == Z->col && W->row == Z->row, "Error output Z has wrong dimensions"); //Ca se duplique c'est moche 
    
    //flip les dimensiosn pour avoir du contigu sur les 2 tenseurs ??

    //Itère sur les rows

    #pragma omp parallel for
    for(size_t i = 0; i < Z->row; i++){
        size_t strideZ = i*Z->strides[1];

        for(size_t j = 0; j < Z->col; j++) Z->datas[strideZ + j*Z->strides[0]] = 0.0f;

        for(size_t k = 0; k < W->col; k++){
            float w_ik = W->datas[i*W->strides[1] + k*W->strides[0]];

            size_t strideX = k*X->strides[1];

            for(size_t j = 0; j < Z->col; j++){
                Z->datas[strideZ + j*Z->strides[0]] += w_ik * X->datas[strideX + j*X->strides[0]];
            }
        }
    }

}
