#include "img_view.h"


Image *batch1 = NULL;
Image *batch2 = NULL;
Image *batch3 = NULL;
Image *batch4 = NULL;
Image *batch5 = NULL;
Image *test = NULL;

const char *classes[10] = {
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
};

void load_CIFAR_Single_Batch(char *filename){

    char path[512];
    snprintf(path, sizeof(path), "%s/%s", DATASETDIR, filename);
    FILE *f = fopen(path, "rb");
    if (!f) { perror("fopen"); return;}

    Image *images = malloc(IMG_PBATCH*sizeof(Image));
    if (!images) {
        LOG_ERROR("malloc failed");
        fclose(f);
        return;
    }
    LOG_INFO("Loading %s",path);

    for (size_t i = 0; i < IMG_PBATCH; i++) {
        //success
        if((fread(&images[i].label,1,1,f) != 1 ) || (fread(images[i].pixels,sizeof(unsigned char),IMG_SIZE*3,f) != IMG_SIZE*3)){
            LOG_ERROR("READ ERROR %zu",i);
            return;
        } else {
            //printf("%u\n",images[i].label);
        }
    }
    fclose(f);

    // WARNING !! CHECK REALLOCATION !!!!!!
    if(strcmp(filename,"data_batch_1.bin") == 0){
        batch1 = images;
    } else if(strcmp(filename,"data_batch_2.bin") == 0){
        batch2 = images;
    } else if(strcmp(filename,"data_batch_3.bin") == 0){
        batch3 = images;
    } else if(strcmp(filename,"data_batch_4.bin") == 0){
        batch4 = images;
    } else if(strcmp(filename,"data_batch_5.bin") == 0){
        batch5 = images;
    } else if(strcmp(filename,"test_batch.bin") == 0){
        test = images;
    } else {
        printf("%s",filename);
        LOG_ERROR("Invalid file name");
        free(images);
    }
}

void load_CIFAR(){
    char path[512];
    for(int i = 0; i < 5; i++){
        snprintf(path, sizeof(path), "data_batch_%c.bin", i + '1');
        load_CIFAR_Single_Batch(path);
    }

    load_CIFAR_Single_Batch("test_batch.bin");
}



//individual clear
static void clear_CIFAR_batch(Image **batch){
    if(batch!= NULL){
        free(*batch);
        *batch = NULL;
    } else {
        LOG_ERROR("Tried to clear a NULL batch");
    }
}

//global clear
void clear_CIFAR(){
    clear_CIFAR_batch(&batch1);
    clear_CIFAR_batch(&batch2);
    clear_CIFAR_batch(&batch3);
    clear_CIFAR_batch(&batch4);
    clear_CIFAR_batch(&batch5);
    clear_CIFAR_batch(&test);
}


void show_image(Image *batch, size_t index){
    FILE *f = fopen("debug_cifar.ppm", "wb");
    if (!f) { perror("fopen"); return; }

    fprintf(f, "P6\n%d %d\n255\n", IMG_DIM, IMG_DIM);
    for (int i = 0; i < IMG_SIZE; i++) {
        fputc(batch[index].pixels[0][i], f);
        fputc(batch[index].pixels[1][i], f);
        fputc(batch[index].pixels[2][i], f);
    }
    fclose(f);

    //irfanview should be in the path
    char cmd[256];
    snprintf(cmd, sizeof(cmd), "i_view64.exe %s", "debug_cifar.ppm");
    LOG_INFO("%s",classes[batch[index].label]);
    system(cmd);
}


//----------------------------------//            
//---       OLD TO DELETE        ---//
//----------------------------------//

void debug_show_image(unsigned char *r, unsigned char *g, unsigned char *b, const char *label){
    FILE *f = fopen("debug_cifar.ppm", "wb");
    if (!f) { perror("fopen"); return; }

    fprintf(f, "P6\n%d %d\n255\n", IMG_DIM, IMG_DIM);


    for (int i = 0; i < IMG_SIZE; i++) {
        fputc(r[i], f);
        fputc(g[i], f);
        fputc(b[i], f);
    }

    fclose(f);

    
    char cmd[256];
    snprintf(cmd, sizeof(cmd), "i_view64.exe %s", "debug_cifar.ppm");
    printf("%s\n",cmd);
    system(cmd);
}

void something(){
    FILE *p;
    unsigned char red[IMG_SIZE];
    unsigned char green[IMG_SIZE];
    unsigned char blue[IMG_SIZE];


    p = fopen("cifar-10-binary/data_batch_1.bin","rb");
    unsigned char label;
    fread(&label, 1, 1, p);   // consomme le label

    fread(red,sizeof(red),1,p);
    fread(green,sizeof(green),1,p);
    fread(blue,sizeof(blue),1,p);
    
    debug_show_image(red, green, blue, "Prout");    

}



