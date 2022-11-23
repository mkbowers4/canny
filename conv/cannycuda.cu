#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>
#include <string.h>
#include <math.h>
//#include <wb.h>

#define K1size 5
#define K2size 3

int IMGwidth, IMGheight;

typedef enum 
{
    NO_EDGE,
    WEAK_EDGE,
    STRONG_EDGE
} EDGE_TYPE;

int writePPM (uint8_t* arr, int size, char* FILENAME, int mode){
    //mode = 0 for color data, mode = 1 for grayscaled data
    int count;
    FILE *fp_wr = fopen(FILENAME, "wb"); // b - binary mode
    fprintf(fp_wr, "P6 %d %d 255\n", IMGwidth, IMGheight);

    if (mode == 1)
        count = fwrite(arr, 1, (size_t)(IMGwidth*IMGheight)*3, fp_wr);
    else{           //we have grayscale data
        uint8_t color[3];

        for (int i=0; i < IMGwidth*IMGheight; i++){
            color[2] = arr[i];
            color[1] = arr[i];
            color[0] = arr[i];
            fwrite(color, 1, 3, fp_wr);
        }
    }
    (void) fclose(fp_wr);

    if (mode == 1)
      return count;
    else
      return 0;
}



void readPPMHeader (FILE *fp){
    int maxval;
    char ch;
    fscanf(fp, "P%c%d%d%d\n", &ch, &IMGwidth, &IMGheight, &maxval);
    printf("PIXEL MAX = %d\n", maxval);
}

int readPPMData (FILE *fp, uint8_t* pixarr, int size){
    int count;
    count = fread(pixarr, 1, (size_t)size, fp);
    return count;
}



int main (int argc, char *argv[]) {

    int status;
    FILE *fp = fopen("img.ppm", "r");

    //get header data & read parameters back
    readPPMHeader(fp);
    printf("IMG WIDTH = %d\n", IMGwidth);
    printf("IMG HEIGHT = %d\n", IMGheight);
    int colorsize = (IMGwidth*IMGwidth)*3;      //a set of RGB values per pixel
    
    //gather pixel data from color img
    uint8_t* pixarr = (uint8_t*)malloc(colorsize);
    status = readPPMData(fp, pixarr, colorsize);
    printf("IMG COUNT = %d\n", status);
    (void) fclose(fp);

    //write back original color image with original pixel data
    char ppm_nochange[20] = "OUT0_unchanged.ppm";
    status = writePPM(pixarr, colorsize, ppm_nochange, 1);
    printf("OUTIMG COUNT = %d\n", status);

//START OF CUDA SETUP
    cudaError_t cudaStatus;
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    //CUDA ALLOCATIONS
    cudaStatus = cudaMemcpy(deviceInputImageData, pixarr, imageWidth * imageHeight * imageChannels * sizeof(uint8_t), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
        cudaStatus = cudaMemcpy(pixarr_grayed, deviceOutputImageData, imageWidth * imageHeight * sizeof(uint8_t), cudaMemcpyDeviceToHost);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMemcpy failed!");
            goto Error;
        }

    
    
















Error:
        cudaFree(deviceInputImageData);
        cudaFree(deviceOutputImageData);
        return cudaStatus;

//CUDA GRAYSCALE
    
}









































