#include "layers/layer.h"

//---------------------//                      
//--  Conv Wrappers  --//
//---------------------//   

static void free_conv_w(void *impl){
    free_conv((ConvLayer*)impl);
}

static void clean_conv_w(void *impl){
    clean_conv((ConvLayer*)(impl));
}

static void recap_conv_w(const void *impl){
    recap_conv((const ConvLayer*)impl);
}

static tensor4_dim_t output_dim_conv_w(const void* impl){
    return output_dim_conv((const ConvLayer*)impl);
}

static const tensor4_t* forward_conv_w(void *impl, const tensor4_t *X){
    return forward_conv((ConvLayer*)impl, X);
}

static const tensor4_t* backward_conv_w(void *impl, const tensor4_t *dP){
    return backward_conv((ConvLayer*)impl, dP);
}

static void update_conv_w(void *impl){
    update_conv((ConvLayer*)impl);
}

//---------------------//                      
//--  Dense Wrappers  --//
//---------------------//  

static void free_dense_w(void *impl){
    free_dense((DenseLayer*)impl);
}

static void clean_dense_w(void *impl){
    clean_dense((DenseLayer*)(impl));
}

static void recap_dense_w(const void *impl){
    recap_dense((const DenseLayer*)impl);
}

static tensor4_dim_t output_dim_dense_w(const void* impl){
    return output_dim_dense((const DenseLayer*)impl);
}

static const tensor4_t* forward_dense_w(void *impl, const tensor4_t *X){
    return forward_dense((DenseLayer*)impl, X);
}

static const tensor4_t* backward_dense_w(void *impl, const tensor4_t *dA){
    return backward_dense((DenseLayer*)impl, dA);
}

static void update_dense_w(void *impl){
    update_dense((DenseLayer*)impl);
}
//not used anumore
/* Layer_Wrapper *make_conv_layer(init_conv_layer_parameter_t p){
    Layer_Wrapper *LW = malloc(sizeof(Layer_Wrapper));

    LW->impl = (void*)init_conv_layer(p);
    LW->recap = recap_conv_w;
    LW->forward = forward_conv_w;
    
    //LW->clean = clean_conv;
    //LW->backward = backward_conv;

    return LW;
}

Layer_Wrapper *make_dense_layer(void *p){
    return NULL;
}
 */
Layer_Wrapper *createLayer(tensor4_dim_t input_dim, const layer_params_t p){
    
    Layer_Wrapper *LW = malloc(sizeof(Layer_Wrapper));
    if(!LW){LOG_ERROR("Bad alloc");}

    switch(p.type){

        case DENSE_LAYER :
            //p.dense.Input
            LW->impl = (void*)init_dense_layer(input_dim,p.dense);
            
            LW->output_dim  = output_dim_dense_w;

            LW->free        = free_dense_w;
            LW->clean       = clean_dense_w;
            LW->recap       = recap_dense_w;
            
            LW->forward     = forward_dense_w;
            LW->backward    = backward_dense_w;
            LW->update      = update_dense_w;

        break;


        case CONV_LAYER :
            
            

            LW->impl        = (void*)init_conv_layer(input_dim,p.conv);
            ConvLayer *cl = LW->impl;
            assert(cl->P!= NULL);
            LW->output_dim  = output_dim_conv_w;
            
            LW->free        = free_conv_w;
            LW->clean       = clean_conv_w;
            LW->recap       = recap_conv_w;
            
            LW->forward     = forward_conv_w;
            LW->backward    = backward_conv_w;
            LW->update      = update_conv_w;

        break;

        
        default :
            LOG_ERROR("Unrecognized layer type");
        break;
    }

    return LW;
}

/* Layer_Wrapper *createLayer(layer_type_t *ltype, void *layerparameters)
{
    return nullptr;
}

Layer_Wrapper *createLayer(const layer_params_t *p){

    Layer_Wrapper *LW = malloc(sizeof(Layer_Wrapper));

    switch()

    return nullptr;
} */
