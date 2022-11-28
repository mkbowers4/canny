#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>
#include <string.h>
#include <math.h>
#include <wb.h>

//#define PPM 
#define K1size 5
#define K2size 3
#define K1Mask_radius K1size / 2
#define K2Mask_radius K2size / 2
#define SKMR K2size / 2
#define TILE_WIDTH 16
#define GBlur_Div 159.0
#define clamp_0_255(x) (min(max((x), 0), 255))
#define clamp_175(x) (min(x, 175))
#define Width_padfor5x5 (TILE_WIDTH + K1size - 1)
#define Width_padfor3x3 (TILE_WIDTH + K2size - 1)
#define supTileWidth (TILE_WIDTH + K2size - 1)

#define highThreshold (uint8_t)(255 * 0.3)
#define lowThreshold (uint8_t)(highThreshold * 0.75)

//image widths and heights gathered from input PPM file
int IMGwidth=0, IMGheight=0;

//error indices for cudaStatus check functions
int cudaMemcpy_index = 0;
int cudaMalloc_index = 0;

//convolution kernels
__constant__ float dev_GaussKernel[K1size * K1size];
__constant__ int dev_xSobelKernel[K2size * K2size];
__constant__ int dev_ySobelKernel[K2size * K2size];

//function prototypes
void readPPMHeader (FILE *fp);
int writePPM (uint8_t* arr, int size, char* FILENAME, int mode);
int readPPMData (FILE *fp, uint8_t* pixarr, int size);

void check_cudaMemcpy(cudaError_t cudaStatus);
void check_cudaMalloc(cudaError_t cudaStatus);

//////////////////////////////////////////KERNELS//////////////////////////////////////////////////////
__global__ void toGrayScale(uint8_t* rgbImage, uint8_t* grayImage, int IMGwidth, int IMGheight)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    if (x < IMGwidth && y < IMGheight){
        int c = (y*IMGwidth)+x;
        int c_rgb = c*3;
        uint8_t r = rgbImage[c_rgb];
        uint8_t g = rgbImage[c_rgb+1];
        uint8_t b = rgbImage[c_rgb+2];
        grayImage[c] = 0.21f*r + 0.71f*g + 0.07f*b;
    }
}

__global__ void NMS_THRESHOLD(uint8_t *IN, uint8_t *XSOBEL, uint8_t *YSOBEL, float* DIR, uint8_t* OUT_NMS, uint8_t* OUT_CANNY, int IMGwidth, int IMGheight)
{
    

    //NO SHARED IMPLEMENTATION (JUST GLOBAL MEMORY READS)
    /*
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    if (x < IMGwidth && y < IMGheight){
        int c = (y*IMGwidth)+x;
        int NN = c - IMGwidth;
        int SS = c + IMGwidth;
        int WW = c - 1;
        int EE = c + 1;
        int NW = NN - 1;
        int NE = NN + 1;
        int SW = SS - 1;
        int SE = SS + 1;

        uint8_t xtemp = XSOBEL[c];
        uint8_t ytemp = YSOBEL[c];

        float DIR = atan2((double)ytemp,(double)ytemp);

        if 
        (  
            ((DIR < 0.3927) && (IN[c] >= IN[EE] && IN[c] >= IN[WW])) || 
            ((DIR > 0.3927) && (DIR < 1.1781) && ((IN[c] >= IN[NW] && IN[c] >= IN[SE]) || (IN[c] >= IN[NE] && IN[c] >= IN[SW]))) || 
            ((DIR > 1.1781) && (IN[c] >= IN[SS] && IN[c] >= IN[NN]))
        )
            OUT[c] = IN[c];
        else   
            OUT[c] = 0; 
    }
    */
    
    
    


    int ty = threadIdx.y; 
    int tx = threadIdx.x;

    //output indices
    int row_o = blockIdx.y*TILE_WIDTH + ty;
    int col_o = blockIdx.x*TILE_WIDTH + tx;

    //input indices
    int row_i = row_o - SKMR;
    int col_i = col_o - SKMR;

    __shared__ uint8_t SHARED_IN[supTileWidth][supTileWidth];
    __shared__ uint8_t OUTPUT_TILE[supTileWidth][supTileWidth];

    //All threads participate in loading shared parts
    if ((row_i >= 0 && row_i < IMGheight) && (col_i >= 0 && col_i < IMGwidth)){
        SHARED_IN[ty][tx] = IN[row_i * IMGwidth + col_i];
        OUTPUT_TILE[ty][tx] = 0;
    }
    else{
        SHARED_IN[ty][tx] = 0;
        OUTPUT_TILE[ty][tx] = 0;
    }
        
    __syncthreads();

    //only threads within tile participate in operation
    if (tx < TILE_WIDTH && ty < TILE_WIDTH){
        int c = IMGwidth * row_o + col_o;
        if (row_o < IMGheight-2 && col_o < IMGwidth-2 && row_o > 1 && col_o > 1)
        //if (row_o < IMGheight && col_o < IMGwidth)
        {
            uint8_t outputPixel = 0;
            uint8_t MAG = SHARED_IN[ty+SKMR][tx+SKMR];
            float dir = DIR[row_o * IMGwidth + col_o];
            //float dir = atan2((double)XSOBEL[row_o * IMGwidth + col_o],(double)YSOBEL[row_o * IMGwidth + col_o]);
            uint8_t NN = SHARED_IN[ty+SKMR-1][tx+SKMR];
            uint8_t SS = SHARED_IN[ty+SKMR+1][tx+SKMR];
            uint8_t WW = SHARED_IN[ty+SKMR][tx+SKMR-1];
            uint8_t EE = SHARED_IN[ty+SKMR][tx+SKMR+1];
            uint8_t NW = SHARED_IN[ty+SKMR-1][tx+SKMR-1];
            uint8_t SE = SHARED_IN[ty+SKMR+1][tx+SKMR+1];
            uint8_t NE = SHARED_IN[ty+SKMR-1][tx+SKMR+1];
            uint8_t SW = SHARED_IN[ty+SKMR+1][tx+SKMR-1];
            
            if 
            (
                
                ((dir < 0.3927) && (MAG >= EE && MAG >= WW)) || 
                ((dir > 0.3927) && (dir < 1.1781) && ( (MAG >= NW && MAG >= SE) || (MAG >= NE && MAG >= SW)))|| 
                ((dir > 1.1781) && (MAG >= SS && MAG >= NN))
            ){
                OUT_NMS[c] = MAG;
                outputPixel = MAG;
            }
                
            else   
                OUT_NMS[c] = 0; 




        //double thresholding

            
            //NOT EDGE PIXEL
            if (outputPixel < lowThreshold)
                OUTPUT_TILE[ty+SKMR][tx+SKMR] = 0;
            //STRONG EDGE PIXEL
            else if (outputPixel > highThreshold)
                OUTPUT_TILE[ty+SKMR][tx+SKMR] = 255;
            //WEAK EDGE PIXEL
            //else if (outputPixel <= highThreshold && outputPixel > lowThreshold)
            else
                OUTPUT_TILE[ty+SKMR][tx+SKMR] = 127;
            
            // __syncthreads();    
            
        //hysteresis

            if (OUTPUT_TILE[ty+SKMR][tx+SKMR] == 0)
                OUT_CANNY[c] = 0;
            else if 
            (
                OUTPUT_TILE[ty+SKMR][tx+SKMR] == 255 || 
                OUTPUT_TILE[ty+SKMR+1][tx+SKMR+1] == 255 ||  
                OUTPUT_TILE[ty+SKMR][tx+SKMR+1] == 255 ||  
                OUTPUT_TILE[ty+SKMR-1][tx+SKMR+1] == 255 ||  
                OUTPUT_TILE[ty+SKMR+1][tx+SKMR-1] == 255 ||  
                OUTPUT_TILE[ty+SKMR][tx+SKMR-1] == 255 ||  
                OUTPUT_TILE[ty+SKMR-1][tx+SKMR-1] == 255 ||  
                OUTPUT_TILE[ty+SKMR+1][tx+SKMR] == 255 ||  
                OUTPUT_TILE[ty+SKMR-1][tx+SKMR] == 255    
            )
                OUT_CANNY[c] = 255;
            else
                OUT_CANNY[c] = 0;
        }
        else
        {   
            OUT_NMS[c] = 0; 
            OUT_CANNY[c] = 0; 
         }   
                
    }

    

}

__global__ void SOBEL(uint8_t *IN, uint8_t* XSOBEL, uint8_t* YSOBEL, float* DIR, uint8_t *OUT, int IMGwidth, int IMGheight)
{
    int ty = threadIdx.y; 
    int tx = threadIdx.x;

    //output indices
    int row_o = blockIdx.y*TILE_WIDTH + ty;
    int col_o = blockIdx.x*TILE_WIDTH + tx;

    //input indices
    int row_i = row_o - K2Mask_radius;
    int col_i = col_o - K2Mask_radius;

    __shared__ uint8_t SHARED[Width_padfor3x3][Width_padfor3x3];

    //all threads participate in loading shared parts for each block
    if ((row_i >= 0 && row_i < IMGheight) && (col_i >= 0 && col_i < IMGwidth))
        SHARED[ty][tx] = IN[row_i * IMGwidth + col_i];
    else
        SHARED[ty][tx] = 0;
    
    __syncthreads();

    uint8_t IMAGEPIXEL;
    int xACCUM, yACCUM;
    uint32_t xACCUMt, yACCUMt;
    uint32_t prod;
    uint32_t sqrt_res;
    //uint8_t xtemp, ytemp;
    //int ftemp;
    int n, m;
    //only threads within tile participate in convolution operation
    if (tx < TILE_WIDTH && ty < TILE_WIDTH){

        xACCUM = 0;
        yACCUM = 0;
        
        //m = row index of kernel
        //m = col index of kernel
        for (int y = -K2Mask_radius; y <= K2Mask_radius; y++){ //row
            m = K2size - 1 - (y + K2Mask_radius);
            for (int x = -K2Mask_radius; x <= K2Mask_radius; x++){ //col
                n = K2size - 1 - (x + K2Mask_radius);
                IMAGEPIXEL = SHARED[ty+y+K2Mask_radius][tx+x+K2Mask_radius];
                xACCUM += (int)IMAGEPIXEL * dev_xSobelKernel[((K2size * n) + m)];
                yACCUM += (int)IMAGEPIXEL * dev_ySobelKernel[((K2size * n) + m)];
            }    
        }
        
        if (row_o < IMGheight && col_o < IMGwidth)
        {
        //Restricting max for each x and y to 175 because √(175^2 + 175^2) < 255. This is to reduce final sobel intensity
            xACCUMt = (uint32_t)abs(xACCUM);
            yACCUMt = (uint32_t)abs(yACCUM);
            xACCUMt = clamp_175(xACCUMt);
            yACCUMt = clamp_175(yACCUMt);

            //write xsobel and ysobel results to global memory
            XSOBEL[(row_o * IMGwidth) + col_o] = (uint8_t)xACCUMt;
            YSOBEL[(row_o * IMGwidth) + col_o] = (uint8_t)yACCUMt;

            //
            DIR[(row_o * IMGwidth) + col_o] = atan2((double)xACCUMt,(double)yACCUMt);

            prod = (xACCUMt*xACCUMt)+(yACCUMt*yACCUMt);
            sqrt_res = (uint32_t)(sqrt((double)prod));

            //write final sobel result (magnitude) to 
            if (sqrt_res > 255)
                OUT[(row_o * IMGwidth) + col_o] = 255;
            else 
                OUT[(row_o * IMGwidth) + col_o] = (uint8_t)sqrt_res;

        }
    }
}

__global__ void GaussBlur(uint8_t *IN, uint8_t *OUT, int IMGwidth, int IMGheight)
{
    int ty = threadIdx.y; 
    int tx = threadIdx.x;

    //output indices
    int row_o = blockIdx.y*TILE_WIDTH + ty;
    int col_o = blockIdx.x*TILE_WIDTH + tx;

    //input indices
    int row_i = row_o - K1Mask_radius;
    int col_i = col_o - K1Mask_radius;

    __shared__ uint8_t SHARED[Width_padfor5x5][Width_padfor5x5];

    //all threads participate in loading shared parts for each block
    if ((row_i >= 0 && row_i < IMGheight) && (col_i >= 0 && col_i < IMGwidth))
        SHARED[ty][tx] = IN[row_i * IMGwidth + col_i];
    else
        SHARED[ty][tx] = 0;
    
    __syncthreads();

    uint8_t IMAGEPIXEL=0;
    float ACCUM=0.0f;
    uint8_t temp=0;
    int tempi=0;
    int n, m;
    //only threads within tile participate in convolution operation
    if (tx < TILE_WIDTH && ty < TILE_WIDTH){
        ACCUM = 0.0f;

        for (int y = -K1Mask_radius; y <= K1Mask_radius; y++){ //row
            m = K1size - 1 - (y + K1Mask_radius);
            for (int x = -K1Mask_radius; x <= K1Mask_radius; x++){ //col
                n = K1size - 1 - (x + K1Mask_radius);
                IMAGEPIXEL = SHARED[ty+y+K1Mask_radius][tx+x+K1Mask_radius];
                ACCUM += (float)IMAGEPIXEL * dev_GaussKernel[((K1size * n) + m)];
                //ACCUM += (float)IMAGEPIXEL * (1.0/25.0);
            }    
        }
        
        if (row_o < IMGheight && col_o < IMGwidth)
        {
            tempi = (int)ACCUM;
            if (tempi > 255)
                tempi = 255;
            if (tempi < 0)
                tempi = 0;

            temp = (uint8_t)tempi;
            OUT[ (row_o * IMGwidth) + col_o] = temp;
        // if (row_o < IMGheight && col_o < IMGwidth)
        //     OUT[ (row_o * IMGwidth) + col_o] = SHARED[ty][tx]/2;
        }   
    }
}


int main (int argc, char *argv[]) {
    int status;
    FILE *fp = fopen("img.ppm", "r");

    uint8_t *dev_pixarr;
    uint8_t *dev_pixarr_grayed;
    uint8_t *dev_pixarr_gauss;
    uint8_t *dev_pixarr_xsobel;
    uint8_t *dev_pixarr_ysobel;
    float *dev_pixarr_dirsobel;
    uint8_t *dev_pixarr_finalsobel;
    uint8_t *dev_pixarr_NMS;
    uint8_t *dev_pixarr_canny;
    
    
    //get header data & read parameters back
    readPPMHeader(fp);
    printf("IMG WIDTH = %d  |  IMG HEIGHT = %d\n", IMGwidth, IMGheight);

    int colorsize = (IMGwidth*IMGheight)*3;      //a set of RGB values per pixel
    int graysize = IMGwidth*IMGheight;


    uint8_t* host_pixarr = (uint8_t*)malloc(colorsize);
    uint8_t* host_pixarr_grayed = (uint8_t*)malloc(graysize);
    uint8_t* host_pixarr_gauss = (uint8_t*)malloc(graysize);
    uint8_t* host_pixarr_finalsobel = (uint8_t*)malloc(graysize);
    uint8_t* host_pixarr_NMS = (uint8_t*)malloc(graysize);
    uint8_t* host_pixarr_canny = (uint8_t*)malloc(graysize);
    status = readPPMData(fp, host_pixarr, colorsize);
    printf("IMG COUNT = %d\n", status);
    (void) fclose(fp);

    //write back original color image with original pixel data
    char ppm_nochange[20] = "0_unchanged_cu.ppm";
    status = writePPM(host_pixarr, colorsize, ppm_nochange, 1);
    printf("OUTIMG COUNT (color) = %d\n", status);

//START OF CUDA SETUP
    cudaError_t cudaStatus;
    if (cudaSetDevice(0) != cudaSuccess) 
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");

    //CUDA ALLOCATIONS
    
    check_cudaMalloc( cudaMalloc((void**)&dev_pixarr, colorsize*sizeof(uint8_t)));
    check_cudaMalloc( cudaMalloc((void**)&dev_pixarr_grayed, graysize*sizeof(uint8_t)));
    check_cudaMalloc( cudaMalloc((void**)&dev_pixarr_gauss, graysize*sizeof(uint8_t)));
    check_cudaMalloc( cudaMalloc((void**)&dev_pixarr_xsobel, graysize*sizeof(uint8_t)));
    check_cudaMalloc( cudaMalloc((void**)&dev_pixarr_ysobel, graysize*sizeof(uint8_t)));
    check_cudaMalloc( cudaMalloc((void**)&dev_pixarr_dirsobel, graysize*sizeof(float)));
    check_cudaMalloc( cudaMalloc((void**)&dev_pixarr_finalsobel, graysize*sizeof(uint8_t)));
    check_cudaMalloc( cudaMalloc((void**)&dev_pixarr_NMS, graysize*sizeof(uint8_t)));
    check_cudaMalloc( cudaMalloc((void**)&dev_pixarr_canny, graysize*sizeof(uint8_t)));
/////////////////////////////////////////////////////////////////////////////////////
///// Step 1: Convert to GRAYSCALE //////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////

    // Copy host color image data to GPU
    check_cudaMemcpy( cudaMemcpy(dev_pixarr, host_pixarr, IMGwidth * IMGheight * 3 * sizeof(uint8_t), cudaMemcpyHostToDevice));

    dim3 dimGrid_gs( ceil((float)IMGwidth / 16), ceil((float)IMGheight / 16) );
    dim3 dimBlock_gs(16,16);

    toGrayScale<<<dimGrid_gs, dimBlock_gs>>>(
        dev_pixarr,
        dev_pixarr_grayed,
        IMGwidth, 
        IMGheight
    );
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) 
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
    
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess)
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
    
    check_cudaMemcpy(cudaMemcpy(host_pixarr_grayed, dev_pixarr_grayed, graysize*sizeof(uint8_t), cudaMemcpyDeviceToHost));
   

    char ppm_grayed[20] = "0_grayed_cu.ppm";
    status = writePPM(host_pixarr_grayed, graysize, ppm_grayed, 0);
    printf("OUTIMG COUNT (gray) = %d\n", status);

/////////////////////////////////////////////////////////////////////////////////////
///// Step 2: Perform Gaussian Blur /////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////
    float host_GaussKernel[K1size*K1size]={ 2.0, 4.0,  5.0,  4.0,  2.0, 
                                            4.0, 9.0,  12.0, 9.0,  4.0, 
                                            5.0, 12.0, 15.0, 12.0, 5.0, 
                                            4.0, 9.0,  12.0, 9.0,  4.0, 
                                            2.0, 4.0,  5.0,  4.0,  2.0 };
    for (int i = 0; i<K1size*K1size; i++)
        host_GaussKernel[i] = host_GaussKernel[i] / GBlur_Div;
        
    //copy gaussian blur convokernel to GPU constant memory
    check_cudaMemcpy(cudaMemcpyToSymbol(dev_GaussKernel, &host_GaussKernel, K1size*K1size*sizeof(float)));

    //CUDAKERNEL SETUP
    dim3 dimGrid_gblur( ceil((float)IMGwidth / 16), ceil((float)IMGheight / 16));
    dim3 dimBlock_gblur(Width_padfor5x5,Width_padfor5x5);

    //CUDAKERNEL LAUNCH
    GaussBlur<<<dimGrid_gblur, dimBlock_gblur>>>(
         dev_pixarr_grayed, 
         dev_pixarr_gauss,
         IMGwidth, 
         IMGheight
     );
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess)
        fprintf(stderr, "GaussBlur launch failed: %s\n", cudaGetErrorString(cudaStatus));

    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess)
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching GaussBlur\n", cudaStatus);

    //copy gaussian blur results from GPU to host (just to print ppm result)
    check_cudaMemcpy(cudaMemcpy(host_pixarr_gauss, dev_pixarr_gauss, graysize*sizeof(uint8_t), cudaMemcpyDeviceToHost));

    //write PPM of gaussian blur results
    char ppm_blurred[20] = "1_blurred_cu.ppm";
    status = writePPM(host_pixarr_gauss, graysize, ppm_blurred, 0);
    printf("OUTIMG COUNT (blur) = %d\n", status);

/////////////////////////////////////////////////////////////////////////////////////
///// Step 3: Perform Sobel Edge Detection //////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////

    int host_xSobelKernel[K2size*K2size] = {-1, 0, 1,
                                            -2, 0, 2, 
                                            -1, 0, 1};
    int host_ySobelKernel[K2size*K2size] = {1, 2, 1, 
                                            0, 0, 0, 
                                           -1,-2,-1};

    //copy sobel edge convokernels to GPU constant memory
    check_cudaMemcpy(cudaMemcpyToSymbol(dev_xSobelKernel, &host_xSobelKernel, K2size*K2size*sizeof(int)));
    check_cudaMemcpy(cudaMemcpyToSymbol(dev_ySobelKernel, &host_ySobelKernel, K2size*K2size*sizeof(int)));
    
    //CUDAKERNEL SETUP
    dim3 dimGrid_sobel( ceil((float)IMGwidth / 16), ceil((float)IMGheight / 16));
    dim3 dimBlock_sobel(Width_padfor3x3,Width_padfor3x3);

    //CUDAKERNEL LAUNCH
    SOBEL<<<dimGrid_sobel, dimBlock_sobel>>>(
         dev_pixarr_gauss, 
         dev_pixarr_xsobel,
         dev_pixarr_ysobel,
         dev_pixarr_dirsobel,
         dev_pixarr_finalsobel,
         IMGwidth, 
         IMGheight
    );
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) 
        fprintf(stderr, "Sobel launch failed: %s\n", cudaGetErrorString(cudaStatus));
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) 
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching Sobel!\n", cudaStatus);
        
    //copy final sobel edge results from GPU to host (just to write PPM results)
    check_cudaMemcpy(cudaMemcpy(host_pixarr_finalsobel, dev_pixarr_finalsobel, graysize*sizeof(uint8_t), cudaMemcpyDeviceToHost));
   
    //write PPM of final sobel results
    char ppm_sobel[20] = "2_sobel_cu.ppm";
    status = writePPM(host_pixarr_finalsobel, graysize, ppm_sobel, 0);
    printf("OUTIMG COUNT (sobel) = %d\n", status);

/////////////////////////////////////////////////////////////////////////////////////
///// Step 4: Perform Non-Maximum Suppression & Hysteresis Thresholding /////////////
/////////////////////////////////////////////////////////////////////////////////////

    //CUDAKERNEL
        //IN: xsobel and ysobel results
        //OUT: 
    dim3 dimGrid_NMS( ceil((float)IMGwidth / 16), ceil((float)IMGheight / 16));
    dim3 dimBlock_NMS(Width_padfor3x3,Width_padfor3x3);

    
    // //__global__ void NMS(uint8_t *IN, float* DIR, uint8_t* OUT, int IMGwidth, int IMGheight){
    // //__global__ void suppressionToOutput(uint8_t* finalSobel_pixarr, uint8_t* xSobelImage, uint8_t* ySobelImage, uint8_t* Output, int width, int height)
    // suppressionToOutput<<<dimGrid_NMS, dimBlock_NMS>>>( 
    //     dev_pixarr_finalsobel,
    //     dev_pixarr_xsobel,
    //     dev_pixarr_ysobel,    
    //     dev_pixarr_NMS,
    //     IMGwidth,
    //     IMGheight
    // );

    //__global__ void NMS(uint8_t *IN, uint8_t *XSOBEL, uint8_t *YSOBEL, float* DIR, uint8_t* OUT_NMS, uint8_t* OUT_CANNY, int IMGwidth, int IMGheight)

    NMS_THRESHOLD<<<dimGrid_NMS, dimBlock_NMS>>>(
        dev_pixarr_finalsobel,
        dev_pixarr_xsobel,
        dev_pixarr_ysobel,
        dev_pixarr_dirsobel,
        dev_pixarr_NMS,
        dev_pixarr_canny,
        IMGwidth,
        IMGheight
    );
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) 
        fprintf(stderr, "NMS launch failed: %s\n", cudaGetErrorString(cudaStatus));
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) 
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching NMS!\n", cudaStatus);
        

    //copy NMS results from GPU to host (just to write PPM results)
    check_cudaMemcpy(cudaMemcpy(host_pixarr_NMS, dev_pixarr_NMS, graysize*sizeof(uint8_t), cudaMemcpyDeviceToHost));
   
    //write PPM of final sobel results
    char ppm_NMS[20] = "3_NMS_cu.ppm";
    status = writePPM(host_pixarr_NMS, graysize, ppm_NMS, 0);
    printf("OUTIMG COUNT (NMS) = %d\n", status);


    //copy canny edge results from GPU to host (just to write PPM results)
    check_cudaMemcpy(cudaMemcpy(host_pixarr_canny, dev_pixarr_canny, graysize*sizeof(uint8_t), cudaMemcpyDeviceToHost));
   
    //write PPM of final sobel results
    char ppm_CANNY[20] = "4_CANNY_cu.ppm";
    status = writePPM(host_pixarr_canny, graysize, ppm_CANNY, 0);
    printf("OUTIMG COUNT (CANNY) = %d\n", status);


//frees

    free(host_pixarr);
    free(host_pixarr_grayed);
    free(host_pixarr_gauss);
    free(host_pixarr_finalsobel);
    free(host_pixarr_NMS);
    free(host_pixarr_canny);

    cudaFree(dev_pixarr);
    cudaFree(dev_pixarr_grayed);
    cudaFree(dev_pixarr_gauss);
    cudaFree(dev_pixarr_xsobel);
    cudaFree(dev_pixarr_ysobel);
    cudaFree(dev_pixarr_finalsobel);
    cudaFree(dev_pixarr_NMS);
    cudaFree(dev_pixarr_canny);

}





int writePPM (uint8_t* arr, int size, char* FILENAME, int mode){
    //mode = 0 for color data, mode = 1 for grayscaled data
    int count=0;
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
            count++;
        }
    }
    (void) fclose(fp_wr);
    return count;
    
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



void check_cudaMemcpy(cudaError_t cudaStatus){
    if (cudaStatus != cudaSuccess)
        fprintf(stderr, "cudaMemcpy %d failed!: Error: %s \n", cudaMemcpy_index, cudaGetErrorString(cudaStatus));
    cudaMemcpy_index++;
}

void check_cudaMalloc(cudaError_t cudaStatus){
    if (cudaStatus != cudaSuccess)
        fprintf(stderr, "cudaMalloc %d failed!: Error: %s \n", cudaMemcpy_index, cudaGetErrorString(cudaStatus));
    cudaMalloc_index++;
}

















