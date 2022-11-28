#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>
#include <string.h>
#include <math.h>
#include "tbb/tbb.h"

#define K1size 5
#define K2size 3
#define clamp_0_255(x) (min(max((x), 0), 255))
#define clamp_175(x) (min(x, 175))

int IMGwidth, IMGheight;
typedef enum
{
    NO_EDGE,
    WEAK_EDGE,
    STRONG_EDGE
} EDGE_TYPE;

//function prototypes
int CannySeq();
int CannyTBB();
int writePPM (uint8_t* arr, int size, char* FILENAME, int mode);
int doGrayscale (uint8_t* oldarr, uint8_t* newarr, int size);
void readPPMHeader (FILE *fp);
int readPPMData (FILE *fp, uint8_t* pixarr, int size);
int doConvoSeq(uint8_t* output, uint8_t *IMGarr, float* K, int IMGwidth, int IMGheight, int Ksize);
int doConvoTBB(uint8_t* output, uint8_t *IMGarr, float* K, int IMGwidth, int IMGheight, int Ksize);



//MAIN
int main (int argc, char *argv[]) {
    int status;
    if (argc != 2){
        printf("Follow Usage: ./canny MODE \n");
        printf("MODE: 0 = Sequential, 1 = TBB, 2 = Both\n");
        printf("T:    0 = No Times,   1 = Times\n");
        exit(-1);
    }
    int mode = atoi(argv[1]);
    //int doTimes = atoi(argv[2]);
    // if (doTimes != 0 && doTimes != 1){
    //     printf("ERROR: invalid T argument\n");
    //     exit(-1);
    // }
    if (mode == 0)
        status = CannySeq();
    else if (mode == 1)
        status = CannyTBB();
    else if (mode == 2){
        status = CannySeq();
        status = CannyTBB();
    }
    else{
        printf("ERROR: invalid MODE argument\n");
        exit(-1);
    }
    

    if (status != 0) printf("Not working!\n");
    else printf("Is Working!\n"); 

}





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

int doGrayscale (uint8_t* oldarr, uint8_t* newarr, int size){
    uint8_t r, g, b;
    int i;
    for (i=0; i<size; i++){
        r = oldarr[(3*i)];
        g = oldarr[(3*i)+1];
        b = oldarr[(3*i)+2];
        newarr[i] = 0.21f*r + 0.71f*g + 0.07f*b;
    }
    return i;
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

uint8_t doConvoTBB_inner(int i, int j, uint8_t* IMGarr, float* K, int IMGwidth, int IMGheight, int Ksize){
    float sum = 0.0;
    int sumtemp = 0;
    int kCX = Ksize/2;
    int kCY = Ksize/2;
    int m, n, mflip, nflip, iflip, jflip;

    //M and N are the local coordinates of the IMG element
    //Mflip and Nflip are the local coordinates of the K element
    //iflip and jflip are transformed coordinates to access IMG element in 1D array
    for (m = 0; m < Ksize; m++)
    {
        mflip = Ksize - 1 - m;
        for (n = 0; n < Ksize; n++)
        {
            nflip = Ksize - 1 - n;
            iflip = i + (kCY - mflip);
            jflip = j + (kCX - nflip);

            if (iflip >= 0 && iflip < IMGheight && jflip >= 0 && jflip < IMGwidth)
                sum += ((int)IMGarr[IMGwidth *iflip + jflip]) * K[Ksize * mflip + nflip];
        }
    }
    sumtemp = abs((int)sum);

    if (sumtemp < 0)
        sumtemp = 0;
    else if (sumtemp > 255)
        sumtemp = 255;

    return (uint8_t)sumtemp;
}

int doConvoTBB(uint8_t* output, uint8_t *IMGarr, float* K, int IMGwidth, int IMGheight, int Ksize){

    //i,j for data,
    //m,n for kernel,
    //mflip,mflip for flipped kernel
    
    //TBB lambda expressions
    tbb::parallel_for(int(0), int(IMGheight), [&] (int i)
    //for (i=0; i < IMGheight; i++)
    {
        tbb::parallel_for(int(0), int(IMGwidth), [&] (int j)
        //for (j = 0; j < IMGwidth; j++)
        {
            output[IMGwidth*i + j] = doConvoTBB_inner(i, j, IMGarr, K, IMGwidth, IMGheight, Ksize);
        });
    });
    return 1;
}

int doConvoSeq(uint8_t* output, uint8_t *IMGarr, float* K, int IMGwidth, int IMGheight, int Ksize){

    float sum;
    int sumtemp;
    int i, j, m, n, mflip, nflip, iflip, jflip;
    int kCX = Ksize/2;
    int kCY = Ksize/2;

    //M and N are the local coordinates of the IMG element
    //Mflip and Nflip are the local coordinates of the K element
    //iflip and jflip are transformed coordinates to access IMG element in 1D array

    for (i=0; i < IMGheight; i++)
    {
        for (j = 0; j < IMGwidth; j++)
        {
            sum = 0.0;
            for (m = 0; m < Ksize; m++)
            {
                mflip = Ksize - 1 - m;
                for (n = 0; n < Ksize; n++)
                {
                    nflip = Ksize - 1 - n;
                    iflip = i + (kCY - mflip);
                    jflip = j + (kCX - nflip);

                    if (iflip >= 0 && iflip < IMGheight && jflip >= 0 && jflip < IMGwidth)
                        sum += ((int)IMGarr[IMGwidth *iflip + jflip]) * K[Ksize * mflip + nflip];
                }
            }

            sumtemp = abs((int)sum);

            if (sumtemp < 0)
                sumtemp = 0;
            else if (sumtemp > 255)
                sumtemp = 255;

            output[IMGwidth*i + j] = (uint8_t)(sumtemp);
        }
    }
    return 1;
}

int CannySeq()
{
    int status; //used for some prints

//NEED a ppm file named img.ppm in the same directory, which is set up to be read from
    FILE *fp = fopen("img.ppm", "r"); 

//get header data from PPM input
    readPPMHeader(fp);
    printf("IMG WIDTH = %d\n", IMGwidth);
    printf("IMG HEIGHT = %d\n", IMGheight);

    int colorsize = (IMGwidth*IMGheight)*3;      //a set of RGB values per pixel
    int graysize = IMGheight*IMGwidth;

//all memory allocations
    uint8_t* pixarr = (uint8_t*)malloc(colorsize);
    uint8_t* grayed_pixarr = (uint8_t*)malloc(graysize);
    uint8_t* blurred_pixarr = (uint8_t*)malloc(graysize);
    uint8_t* xSobel_pixarr = (uint8_t*)malloc(graysize*sizeof(uint8_t));
    uint8_t* ySobel_pixarr = (uint8_t*)malloc(graysize*sizeof(uint8_t));
    uint8_t* finalSobel_pixarr = (uint8_t*)malloc(graysize*sizeof(uint8_t));
    EDGE_TYPE* PixelType = (EDGE_TYPE*)malloc(graysize*sizeof(EDGE_TYPE));
    uint8_t* afterNMS = (uint8_t*)calloc(graysize, sizeof(uint8_t));
    uint8_t* afterDthr = (uint8_t*)calloc(graysize, sizeof(uint8_t));
    uint8_t* afterDthr2 = (uint8_t*)calloc(graysize, sizeof(uint8_t));
    uint8_t* afterHyst = (uint8_t*)calloc(graysize, sizeof(uint8_t));

//get pixel data from PPM input
    status = readPPMData(fp, pixarr, colorsize);
    printf("IMG COUNT = %d\n", status);
    (void) fclose(fp);

//write back original color image with original pixel data
    char ppm_nochange[20] = "OUT0_unchanged.ppm";
    status = writePPM(pixarr, colorsize, ppm_nochange, 1);
    printf("OUTIMG COUNT = %d\n", status);
    printf("hello1\n");
/////////////////////////////////////////////////////////////////////////////////////
///// Step 1: Convert to GRAYSCALE //////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////
    status = doGrayscale(pixarr, grayed_pixarr, graysize);
    printf("hello1.5\n");
//print out grayscale image
    char ppm_grayed[20] = "OUT1_grayscale.ppm";
    status = writePPM(grayed_pixarr, graysize, ppm_grayed, 0);

    printf("hello2\n");


/////////////////////////////////////////////////////////////////////////////////////
///// Step 2: Perform Gaussian Blur /////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////
  
float KGaussF[K1size*K1size]= {2., 4.,  5.,  4.,  2., 
                               4., 9.,  12., 9.,  4., 
                               5., 12., 15., 12., 5., 
                               4., 9.,  12., 9.,  4., 
                               2., 4.,  5.,  4.,  2};     
//treat gauss
    for (int i = 0; i <K1size*K1size; i++)
        KGaussF[i] = KGaussF[i]/159.;
    for (int i=0; i<K1size*K1size; i++)
        printf("K[%d] = %f ",i, KGaussF[i]);

//GAUSSIAN BLUR CONVOLUTION
    status = doConvoSeq(blurred_pixarr, grayed_pixarr, KGaussF, IMGwidth, IMGheight, K1size);

//print blurred image
    char ppm_blurred[20] = "OUT2_blurred.ppm";
    status = writePPM(blurred_pixarr, graysize, ppm_blurred, 0);


/////////////////////////////////////////////////////////////////////////////////////
///// Step 3: Perform Sobel Edge Detection //////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////

    float KxSobel[K2size*K2size] = {-1., 0., 1., -2., 0., 2., -1., 0., 1.};
    float KySobel[K2size*K2size] = {1., 2., 1., 0., 0., 0., -1., -2., -1.};

//do X Sobel and print results
    char ppm_xSobel[20]= "OUT4_xSobel.ppm";
    status = doConvoSeq(xSobel_pixarr, blurred_pixarr, KxSobel, IMGwidth, IMGheight, K2size);
    status = writePPM(xSobel_pixarr, graysize, ppm_xSobel, 0);

//do Y Sobel and print results
    char ppm_ySobel[20] = "OUT5_ySobel.ppm";
    status = doConvoSeq(ySobel_pixarr, blurred_pixarr, KySobel, IMGwidth, IMGheight, K2size);
    status = writePPM(ySobel_pixarr, graysize, ppm_ySobel, 0);  

//suppress absurd highs
//this is to limit eventual hypot result (175^2 + 175^2) =~ 247
    for (int i=0; i < graysize; i++){
        if (ySobel_pixarr[i] > 175)
            ySobel_pixarr[i] = 175;
        if (xSobel_pixarr[i] > 175)
            xSobel_pixarr[i] = 175;
    }

//get final magnitude of Sobel

    for (int i=0; i < graysize; i++){
        uint32_t prod = ((uint32_t)ySobel_pixarr[i]*ySobel_pixarr[i])+((uint32_t)xSobel_pixarr[i]*xSobel_pixarr[i]);

            //saturating (some very strong edges were getting turned over
            if (sqrt(prod) > 255.0)
                finalSobel_pixarr[i] = 255;
            else
                finalSobel_pixarr[i] = (uint8_t)sqrt(prod);
    }

//FINAL SOBEL PRINT
        char ppm_finalSobel[20] = "OUT5_finalSobel.ppm";
        status = writePPM(finalSobel_pixarr, graysize, ppm_finalSobel, 0);

/////////////////////////////////////////////////////////////////////////////////////
///// Step 4: Perform Non-Maximum Suppression ///////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////
 /*

        - Find raw gradient direction, and confine to ranges of 0° (WW to EE), 45/135° (SW to NE, NW to SE), 90° (SS to NN)
        -if the gradient direction is in a particular location (0°, 45/135°, 90°), then check if the center pixel is smaller than its neighbors
        (checking for a local maximum. If it is a local maximum, we keep it)

                NW     NN     NE

                WW     C      EE

                SW     SS     SE
*/
    //ignore the border pixels, so we +1 and -1 on the count start & end, respectively
    for (int i=2; i < IMGheight-2; i++){
        for (int j=2; j < IMGwidth-2; j++){

            //take current center of pixel, and derive all neighboring indices from it
            int c = IMGwidth*i + j;
            int nn = c - IMGwidth;
            int ss = c + IMGwidth;
            int ww = c + 1;
            int ee = c - 1;
            int nw = nn + 1;
            int ne = nn - 1;
            int sw = ss + 1;
            int se = ss - 1;

            float dir = atan2(ySobel_pixarr[c], xSobel_pixarr[c]);

            if
            (
                ( (dir < 0.3927) && (finalSobel_pixarr[c] >= finalSobel_pixarr[ee] && finalSobel_pixarr[c] >= finalSobel_pixarr[ww])) ||
                ( (dir > 0.3927) && (dir < 1.1781) && (finalSobel_pixarr[c] >= finalSobel_pixarr[ss] && finalSobel_pixarr[c] >= finalSobel_pixarr[nn])) ||
                ( (dir > 1.1781) && ( (finalSobel_pixarr[c] >= finalSobel_pixarr[sw] && finalSobel_pixarr[c] >= finalSobel_pixarr[ne]) && (finalSobel_pixarr[c] >= finalSobel_pixarr[se] && finalSobel_pixarr[c] >= finalSobel_pixarr[nw])))
            )
                 afterNMS[c] = finalSobel_pixarr[c];
            else
                 afterNMS[c] = 0;
        }
    }

//print out NMS results for PPM 
    char ppm_nms[20] = "OUT_nms.ppm";
    status = writePPM(afterNMS, graysize, ppm_nms, 0);

/////////////////////////////////////////////////////////////////////////////////////
///// Step 5: Perform Double Thresholding ///////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////

//here we classify our edges if they're strong or not
    uint8_t highThreshold = (uint8_t)(255*0.3);
    uint8_t lowThreshold = (uint8_t)(highThreshold*0.75);
    for (int i=0; i < IMGheight; i++){
        for (int j=0; j < IMGwidth; j++){
            int c = IMGwidth*i + j;
            if (afterNMS[c] > highThreshold)
                PixelType[c] = STRONG_EDGE;
            else if ((afterNMS[c] < highThreshold) && (afterNMS[c] > lowThreshold))
            {
                PixelType[c] = WEAK_EDGE;
            }
            else
                PixelType[c] = NO_EDGE;
        }
    }

//optional print after threshold
    for (int i=0; i < IMGheight; i++){
        for (int j=0; j < IMGwidth; j++){
            int c = IMGwidth*i + j;
            if (PixelType[c] == NO_EDGE)
                afterDthr[c] = 0;
            else
                afterDthr[c] = afterNMS[c];
        }
    }
    char ppm_dthr[20] = "OUT_dthr.ppm";
    status = writePPM(afterDthr, graysize, ppm_dthr, 0);


    for (int i=0; i < IMGheight; i++){
        for (int j=0; j < IMGwidth; j++){
            int c = IMGwidth*i + j;
            if (PixelType[c] == NO_EDGE)
                afterDthr2[c] = 0;
            else if (PixelType[c] == WEAK_EDGE)
                afterDthr2[c] = 125;
            else
                afterDthr2[c] = 255;
        }
    }
    char ppm_dthr2[20] = "OUT_dthr2.ppm";
    status = writePPM(afterDthr2, graysize, ppm_dthr2, 0);

/////////////////////////////////////////////////////////////////////////////////////
///// Step 6: Perform Hysteresis Thresholding ///////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////

    for (int i=1; i < IMGheight-1; i++){
        for (int j=1; j < IMGwidth-1; j++){

            //take current center of pixel, and derive all neighboring indices from it
            int c = IMGwidth*i + j;
            int nn = c - IMGwidth;
            int ss = c + IMGwidth;
            int ww = c + 1;
            int ee = c - 1;
            int nw = nn + 1;
            int ne = nn - 1;
            int sw = ss + 1;
            int se = ss - 1;
            if (PixelType[c] == NO_EDGE)    //suppress no edges
                afterHyst[c] = 0;
            else if (PixelType[c] == STRONG_EDGE){
                //saturate strong edges
                afterHyst[c] = 255;
            }

            //now we need to consider weak edges. If any other surrounding edges are strong, make saturate the weak edge as you would a strong edge
            //else, we remove weak edge
            else if (
                (PixelType[nn] == STRONG_EDGE) || (PixelType[ss] == STRONG_EDGE) || (PixelType[ww] == STRONG_EDGE) || (PixelType[ee] == STRONG_EDGE) ||
                (PixelType[nw] == STRONG_EDGE) || (PixelType[ne] == STRONG_EDGE) || (PixelType[sw] == STRONG_EDGE) || (PixelType[se] == STRONG_EDGE)
            )
                afterHyst[c] = 255;
            else
                afterHyst[c] = 0;
        }
    }
    //print out PPM for NMS
    char ppm_hyst[20] = "OUT_hyst.ppm";
    status = writePPM(afterHyst, graysize, ppm_hyst, 0);
    printf("Done.");

    free(pixarr);
    free(grayed_pixarr);
    free(blurred_pixarr);
    free(xSobel_pixarr);
    free(ySobel_pixarr);
    free(finalSobel_pixarr);
    free(PixelType);
    free(afterNMS);
    free(afterDthr);
    free(afterDthr2);
    free(afterHyst);
    return 0;
}







// int CannySeq()
// {
//     int status; //used for some prints

// //NEED a ppm file named img.ppm in the same directory, which is set up to be read from
//     FILE *fp = fopen("img.ppm", "r"); 

// //get header data from PPM input
//     readPPMHeader(fp);
//     printf("IMG WIDTH = %d\n", IMGwidth);
//     printf("IMG HEIGHT = %d\n", IMGheight);

//     int colorsize = (IMGwidth*IMGheight)*3;      //a set of RGB values per pixel
//     int graysize = IMGheight*IMGwidth;

// //all memory allocations
//     uint8_t* pixarr = (uint8_t*)malloc(colorsize);
//     uint8_t* grayed_pixarr = (uint8_t*)malloc(graysize);
//     uint8_t* blurred_pixarr = (uint8_t*)malloc(graysize);
//     uint8_t* xSobel_pixarr = (uint8_t*)malloc(graysize*sizeof(uint8_t));
//     uint8_t* ySobel_pixarr = (uint8_t*)malloc(graysize*sizeof(uint8_t));
//     uint8_t* finalSobel_pixarr = (uint8_t*)malloc(graysize*sizeof(uint8_t));
//     EDGE_TYPE* PixelType = (EDGE_TYPE*)malloc(graysize*sizeof(EDGE_TYPE));
//     uint8_t* afterNMS = (uint8_t*)calloc(graysize, sizeof(uint8_t));
//     uint8_t* afterDthr = (uint8_t*)calloc(graysize, sizeof(uint8_t));
//     uint8_t* afterDthr2 = (uint8_t*)calloc(graysize, sizeof(uint8_t));
//     uint8_t* afterHyst = (uint8_t*)calloc(graysize, sizeof(uint8_t));

// //get pixel data from PPM input
//     status = readPPMData(fp, pixarr, colorsize);
//     printf("IMG COUNT = %d\n", status);
//     (void) fclose(fp);

// //write back original color image with original pixel data
//     char ppm_nochange[20] = "OUT0_unchanged.ppm";
//     status = writePPM(pixarr, colorsize, ppm_nochange, 1);
//     printf("OUTIMG COUNT = %d\n", status);
//     printf("hello1\n");
// /////////////////////////////////////////////////////////////////////////////////////
// ///// Step 1: Convert to GRAYSCALE //////////////////////////////////////////////////
// /////////////////////////////////////////////////////////////////////////////////////
//     status = doGrayscale(pixarr, grayed_pixarr, graysize);
//     printf("hello1.5\n");
// //print out grayscale image
//     char ppm_grayed[20] = "OUT1_grayscale.ppm";
//     status = writePPM(grayed_pixarr, graysize, ppm_grayed, 0);

//     printf("hello2\n");


// /////////////////////////////////////////////////////////////////////////////////////
// ///// Step 2: Perform Gaussian Blur /////////////////////////////////////////////////
// /////////////////////////////////////////////////////////////////////////////////////
  
// float KGaussF[K1size*K1size]= {2., 4.,  5.,  4.,  2., 
//                                4., 9.,  12., 9.,  4., 
//                                5., 12., 15., 12., 5., 
//                                4., 9.,  12., 9.,  4., 
//                                2., 4.,  5.,  4.,  2};     
// //treat gauss
//     for (int i = 0; i <K1size*K1size; i++)
//         KGaussF[i] = KGaussF[i]/159.;
//     for (int i=0; i<K1size*K1size; i++)
//         printf("K[%d] = %f ",i, KGaussF[i]);

// //GAUSSIAN BLUR CONVOLUTION
//     status = doConvoSeq(blurred_pixarr, grayed_pixarr, KGaussF, IMGwidth, IMGheight, K1size);

// //print blurred image
//     char ppm_blurred[20] = "OUT2_blurred.ppm";
//     status = writePPM(blurred_pixarr, graysize, ppm_blurred, 0);


// /////////////////////////////////////////////////////////////////////////////////////
// ///// Step 3: Perform Sobel Edge Detection //////////////////////////////////////////
// /////////////////////////////////////////////////////////////////////////////////////

//     float KxSobel[K2size*K2size] = {-1., 0., 1., -2., 0., 2., -1., 0., 1.};
//     float KySobel[K2size*K2size] = {1., 2., 1., 0., 0., 0., -1., -2., -1.};

// //do X Sobel and print results
//     char ppm_xSobel[20]= "OUT4_xSobel.ppm";
//     status = doConvoSeq(xSobel_pixarr, blurred_pixarr, KxSobel, IMGwidth, IMGheight, K2size);
//     status = writePPM(xSobel_pixarr, graysize, ppm_xSobel, 0);

// //do Y Sobel and print results
//     char ppm_ySobel[20] = "OUT5_ySobel.ppm";
//     status = doConvoSeq(ySobel_pixarr, blurred_pixarr, KySobel, IMGwidth, IMGheight, K2size);
//     status = writePPM(ySobel_pixarr, graysize, ppm_ySobel, 0);  

// //suppress absurd highs
// //this is to limit eventual hypot result (175^2 + 175^2) =~ 247
//     for (int i=0; i < graysize; i++){
//         if (ySobel_pixarr[i] > 175)
//             ySobel_pixarr[i] = 175;
//         if (xSobel_pixarr[i] > 175)
//             xSobel_pixarr[i] = 175;
//     }

// //get final magnitude of Sobel

//     for (int i=0; i < graysize; i++){
//         uint32_t prod = ((uint32_t)ySobel_pixarr[i]*ySobel_pixarr[i])+((uint32_t)xSobel_pixarr[i]*xSobel_pixarr[i]);

//             //saturating (some very strong edges were getting turned over
//             if (sqrt(prod) > 255.0)
//                 finalSobel_pixarr[i] = 255;
//             else
//                 finalSobel_pixarr[i] = (uint8_t)sqrt(prod);
//     }

// //FINAL SOBEL PRINT
//         char ppm_finalSobel[20] = "OUT5_finalSobel.ppm";
//         status = writePPM(finalSobel_pixarr, graysize, ppm_finalSobel, 0);

// /////////////////////////////////////////////////////////////////////////////////////
// ///// Step 4: Perform Non-Maximum Suppression ///////////////////////////////////////
// /////////////////////////////////////////////////////////////////////////////////////
//  /*

//         - Find raw gradient direction, and confine to ranges of 0° (WW to EE), 45/135° (SW to NE, NW to SE), 90° (SS to NN)
//         -if the gradient direction is in a particular location (0°, 45/135°, 90°), then check if the center pixel is smaller than its neighbors
//         (checking for a local maximum. If it is a local maximum, we keep it)

//                 NW     NN     NE

//                 WW     C      EE

//                 SW     SS     SE
// */
//     //ignore the border pixels, so we +1 and -1 on the count start & end, respectively
//     for (int i=2; i < IMGheight-2; i++){
//         for (int j=2; j < IMGwidth-2; j++){

//             //take current center of pixel, and derive all neighboring indices from it
//             int c = IMGwidth*i + j;
//             int nn = c - IMGwidth;
//             int ss = c + IMGwidth;
//             int ww = c + 1;
//             int ee = c - 1;
//             int nw = nn + 1;
//             int ne = nn - 1;
//             int sw = ss + 1;
//             int se = ss - 1;

//             float dir = atan2(ySobel_pixarr[c], xSobel_pixarr[c]);

//             if
//             (
//                 ( (dir < 0.3927) && (finalSobel_pixarr[c] >= finalSobel_pixarr[ee] && finalSobel_pixarr[c] >= finalSobel_pixarr[ww])) ||
//                 ( (dir > 0.3927) && (dir < 1.1781) && (finalSobel_pixarr[c] >= finalSobel_pixarr[ss] && finalSobel_pixarr[c] >= finalSobel_pixarr[nn])) ||
//                 ( (dir > 1.1781) && ( (finalSobel_pixarr[c] >= finalSobel_pixarr[sw] && finalSobel_pixarr[c] >= finalSobel_pixarr[ne]) && (finalSobel_pixarr[c] >= finalSobel_pixarr[se] && finalSobel_pixarr[c] >= finalSobel_pixarr[nw])))
//             )
//                  afterNMS[c] = finalSobel_pixarr[c];
//             else
//                  afterNMS[c] = 0;
//         }
//     }

// //print out NMS results for PPM 
//     char ppm_nms[20] = "OUT_nms.ppm";
//     status = writePPM(afterNMS, graysize, ppm_nms, 0);

// /////////////////////////////////////////////////////////////////////////////////////
// ///// Step 5: Perform Double Thresholding ///////////////////////////////////////////
// /////////////////////////////////////////////////////////////////////////////////////

// //here we classify our edges if they're strong or not
//     uint8_t highThreshold = (uint8_t)(255*0.3);
//     uint8_t lowThreshold = (uint8_t)(highThreshold*0.75);
//     for (int i=0; i < IMGheight; i++){
//         for (int j=0; j < IMGwidth; j++){
//             int c = IMGwidth*i + j;
//             if (afterNMS[c] > highThreshold)
//                 PixelType[c] = STRONG_EDGE;
//             else if ((afterNMS[c] < highThreshold) && (afterNMS[c] > lowThreshold))
//             {
//                 PixelType[c] = WEAK_EDGE;
//             }
//             else
//                 PixelType[c] = NO_EDGE;
//         }
//     }

// //optional print after threshold
//     for (int i=0; i < IMGheight; i++){
//         for (int j=0; j < IMGwidth; j++){
//             int c = IMGwidth*i + j;
//             if (PixelType[c] == NO_EDGE)
//                 afterDthr[c] = 0;
//             else
//                 afterDthr[c] = afterNMS[c];
//         }
//     }
//     char ppm_dthr[20] = "OUT_dthr.ppm";
//     status = writePPM(afterDthr, graysize, ppm_dthr, 0);


//     for (int i=0; i < IMGheight; i++){
//         for (int j=0; j < IMGwidth; j++){
//             int c = IMGwidth*i + j;
//             if (PixelType[c] == NO_EDGE)
//                 afterDthr2[c] = 0;
//             else if (PixelType[c] == WEAK_EDGE)
//                 afterDthr2[c] = 125;
//             else
//                 afterDthr2[c] = 255;
//         }
//     }
//     char ppm_dthr2[20] = "OUT_dthr2.ppm";
//     status = writePPM(afterDthr2, graysize, ppm_dthr2, 0);

// /////////////////////////////////////////////////////////////////////////////////////
// ///// Step 6: Perform Hysteresis Thresholding ///////////////////////////////////////
// /////////////////////////////////////////////////////////////////////////////////////

//     for (int i=1; i < IMGheight-1; i++){
//         for (int j=1; j < IMGwidth-1; j++){

//             //take current center of pixel, and derive all neighboring indices from it
//             int c = IMGwidth*i + j;
//             int nn = c - IMGwidth;
//             int ss = c + IMGwidth;
//             int ww = c + 1;
//             int ee = c - 1;
//             int nw = nn + 1;
//             int ne = nn - 1;
//             int sw = ss + 1;
//             int se = ss - 1;
//             if (PixelType[c] == NO_EDGE)    //suppress no edges
//                 afterHyst[c] = 0;
//             else if (PixelType[c] == STRONG_EDGE){
//                 //saturate strong edges
//                 afterHyst[c] = 255;
//             }

//             //now we need to consider weak edges. If any other surrounding edges are strong, make saturate the weak edge as you would a strong edge
//             //else, we remove weak edge
//             else if (
//                 (PixelType[nn] == STRONG_EDGE) || (PixelType[ss] == STRONG_EDGE) || (PixelType[ww] == STRONG_EDGE) || (PixelType[ee] == STRONG_EDGE) ||
//                 (PixelType[nw] == STRONG_EDGE) || (PixelType[ne] == STRONG_EDGE) || (PixelType[sw] == STRONG_EDGE) || (PixelType[se] == STRONG_EDGE)
//             )
//                 afterHyst[c] = 255;
//             else
//                 afterHyst[c] = 0;
//         }
//     }
//     //print out PPM for NMS
//     char ppm_hyst[20] = "OUT_hyst.ppm";
//     status = writePPM(afterHyst, graysize, ppm_hyst, 0);
//     printf("Done.");

//     free(pixarr);
//     free(grayed_pixarr);
//     free(blurred_pixarr);
//     free(xSobel_pixarr);
//     free(ySobel_pixarr);
//     free(finalSobel_pixarr);
//     free(PixelType);
//     free(afterNMS);
//     free(afterDthr);
//     free(afterDthr2);
//     free(afterHyst);
//     return 0;
// }


































