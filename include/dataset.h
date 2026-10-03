#ifndef DATASET_H
#define DATASET_H

#include "stdio.h"
#include "stdlib.h"
#include "string.h"
#include "debug.h"
#include "tensor/tensor.h"

#define IMG_DIM         32
#define IMG_SIZE        ((IMG_DIM)*(IMG_DIM))
#define IMG_PBATCH      10000

#define DATASETDIR      "cifar-10-binary"


typedef struct{
    unsigned char label;
    unsigned char pixels[3][IMG_SIZE];
} Image;


//va être renommé en dataset.h


extern Image* batch1;
extern Image* batch2;
extern Image* batch3;
extern Image* batch4;
extern Image* batch5;
extern Image* test;

extern const char *classes[10];

//Loads a cifar batch from cifar-10-binary directory, call with the file name with extension (for example "data_batch_1.bin")
void load_CIFAR_Single_Batch(char *filename);

//loads all the available batches in the CIFAR-10 dataset
void load_CIFAR();

//clears all the stored dataset
void clear_CIFAR();

//show an image and print its info
void show_image(Image *batch, size_t index);

//fonction d'init de tenseur à partir du batch à déplacer ici
tensor4_t* init_tensor_from_batch(Image *batch, size_t idxStart, size_t nimages);

void something();
void debug_show_image(unsigned char *r, unsigned char *g, unsigned char *b, const char *label);
#endif