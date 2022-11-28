#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>
#include <string.h>
#include <math.h>
#include "tbb/tbb.h"

#define USE_TIMES

#define KGaussSize 5
#define KSobelSize 3
#define clamp_0_255(x) (min(max((x), 0), 255))
#define clamp_175(x) (fmin(x, 175))

int IMGwidth, IMGheight;
typedef enum
{
    NO_EDGE,
    WEAK_EDGE,
    STRONG_EDGE
} EDGE_TYPE;

//function prototypes
int CannySeq(int printPPMs);
void doConvoSeq(uint8_t* output, uint8_t *IMGarr, float* K, int IMGwidth, int IMGheight, int Ksize);
int doGrayscaleSeq (uint8_t* oldarr, uint8_t* newarr, int size);

int CannyTBB(int printPPMs);
void doConvoTBB(uint8_t* output, uint8_t *IMGarr, float* K, int IMGwidth, int IMGheight, int Ksize);
uint8_t doNMS_tbb(uint8_t* finalSobel_pixarr, uint8_t* xSobel_pixarr, uint8_t* ySobel_pixarr, int i, int j);
uint8_t doConvoTBB_inner(int i, int j, uint8_t* IMGarr, float* K, int IMGwidth, int IMGheight, int Ksize);
uint8_t doGrayscaleTBB (uint8_t* oldarr, int i);
uint8_t getSobelMagnitude_TBB(uint8_t* xSobel_pixarr, uint8_t* ySobel_pixarr, int i);
uint8_t doThresh_tbb(uint8_t* afterNMS, int i, int j, uint8_t highThreshold, uint8_t lowThreshold);

int writePPM (uint8_t* arr, int size, char* FILENAME, int mode);
void readPPMHeader (FILE *fp);
int readPPMData (FILE *fp, uint8_t* pixarr, int size);

//MAIN
int main (int argc, char *argv[]) {
    int status;
    if (argc != 3){
        printf("Follow Usage: ./canny MODE PPMs\n");
        printf("MODE: 0 = Sequential, 1 = TBB, 2 = Both\n");
        printf("PPMs: 0 = Print only canny PPM, 1 = Print all PPMs");
        printf("T:    0 = No Times,   1 = Times\n");
        exit(-1);
    }
    int mode = atoi(argv[1]);
    int printPPMs = atoi(argv[2]);

    if (printPPMs != 0 && printPPMs != 1){
         printf("ERROR: invalid PPMs argument\n");
         exit(-1);
    }
    if (mode == 0)
        status = CannySeq(printPPMs);
    else if (mode == 1)
        status = CannyTBB(printPPMs);
    else if (mode == 2){
        status = CannySeq(printPPMs);
        status = CannyTBB(printPPMs);
    }
    else{
        printf("ERROR: invalid MODE argument\n");
        exit(-1);
    }
    
    if (status != 0) printf("Not working!\n");
    else printf("Done!");

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

int doGrayscaleSeq (uint8_t* oldarr, uint8_t* newarr, int size){
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

void doConvoSeq(uint8_t* output, uint8_t *IMGarr, float* K, int IMGwidth, int IMGheight, int Ksize){

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
}

int CannySeq(int printPPMs)
{
    int status; 
//NEED a ppm file named img.ppm in the same directory, which is set up to be read from
    FILE *fp;
    if ( !(fp = fopen("img.ppm", "r"))){
        printf("ERROR: Need a .ppm (P6) file named 'img.ppm' in same directory as executable\n");
        exit(-1);
    }

//get header data (width and height of image) from PPM input
    readPPMHeader(fp);
    printf("IMG WIDTH = %d\n", IMGwidth);
    printf("IMG HEIGHT = %d\n", IMGheight);

//PPM uses 1R, 1G, 1B per pixel
//only keep track of one pixel for grayscale calculations
    int colorsize = (IMGwidth*IMGheight)*3;      
    int graysize = IMGheight*IMGwidth;

//all memory allocations
//x and y sobel are split up to print separate .PPMs for each
    uint8_t* pixarr = (uint8_t*)malloc(colorsize);
    uint8_t* grayed_pixarr = (uint8_t*)malloc(graysize);
    uint8_t* blurred_pixarr = (uint8_t*)malloc(graysize);
    uint8_t* xSobel_pixarr = (uint8_t*)malloc(graysize*sizeof(uint8_t));
    uint8_t* ySobel_pixarr = (uint8_t*)malloc(graysize*sizeof(uint8_t));
    uint8_t* finalSobel_pixarr = (uint8_t*)malloc(graysize*sizeof(uint8_t));
    uint8_t* afterNMS = (uint8_t*)calloc(graysize, sizeof(uint8_t));
    uint8_t* afterHyst = (uint8_t*)calloc(graysize, sizeof(uint8_t));
    
    EDGE_TYPE* PixelType = (EDGE_TYPE*)malloc(graysize*sizeof(EDGE_TYPE));
    uint8_t* afterDthr = (uint8_t*)calloc(graysize, sizeof(uint8_t));
    uint8_t* afterDthr2 = (uint8_t*)calloc(graysize, sizeof(uint8_t));
    
//get pixel data from PPM input
    status = readPPMData(fp, pixarr, colorsize);
    printf("IMG COUNT = %d\n", status);
    (void) fclose(fp);

//write back original color image to new PPM
    if (printPPMs){
        char ppm_nochange[20] = "0_original_s.ppm";
        status = writePPM(pixarr, colorsize, ppm_nochange, 1);
        printf("OUTIMG COUNT = %d\n", status);
    }

//timing 
    #ifdef USE_TIMES
        struct timeval start, end; 
        long t_us, t_us_total = 0;
    #endif

/////////////////////////////////////////////////////////////////////////////////////
///// Step 1: Convert to GRAYSCALE //////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////
    gettimeofday(&start, NULL);
    status = doGrayscaleSeq(pixarr, grayed_pixarr, graysize);
    gettimeofday(&end, NULL);

//print out timing results for sequential grayscale 
    printf("Grayscale_seq Run:\n");
    printf("    Start: %ld us    End: %ld us\n", start.tv_usec, end.tv_usec);
    t_us = (end.tv_sec - start.tv_sec)*1000000 + end.tv_usec - start.tv_usec;
    printf("    Elapsed time: %ld us\n", t_us);
    t_us_total += t_us;

//write out ppm of grayscale image
    if (printPPMs){
        char ppm_grayed[20] = "1_grayscale_s.ppm";
        status = writePPM(grayed_pixarr, graysize, ppm_grayed, 0);
    }

    /////////////////////////////////////////////////////////////////////////////////////
    ///// Step 2: Perform Gaussian Blur /////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////
    gettimeofday(&start, NULL);

    float KGaussF[KGaussSize*KGaussSize]= {2., 4.,  5.,  4.,  2., 
                                4., 9.,  12., 9.,  4., 
                                5., 12., 15., 12., 5., 
                                4., 9.,  12., 9.,  4., 
                                2., 4.,  5.,  4.,  2};     
    //treat gauss
        for (int i = 0; i <KGaussSize*KGaussSize; i++)
            KGaussF[i] = KGaussF[i]/159.;

//GAUSSIAN BLUR CONVOLUTION
    doConvoSeq(blurred_pixarr, grayed_pixarr, KGaussF, IMGwidth, IMGheight, KGaussSize);

    gettimeofday(&end, NULL);
        printf("GaussBlur_seq Run:\n");
        printf("    Start: %ld us    End: %ld us\n", start.tv_usec, end.tv_usec);
        t_us = (end.tv_sec - start.tv_sec)*1000000 + end.tv_usec - start.tv_usec;
        printf("    Elapsed time: %ld us\n", t_us);
        t_us_total += t_us;

    if (printPPMs){
    //print blurred image
        char ppm_blurred[20] = "2_blurred_s.ppm";
        status = writePPM(blurred_pixarr, graysize, ppm_blurred, 0);
    }
    /////////////////////////////////////////////////////////////////////////////////////
    ///// Step 3: Perform Sobel Edge Detection //////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////
        gettimeofday(&start, NULL);

        float KxSobel[KSobelSize*KSobelSize] = {-1., 0., 1., 
                                                -2., 0., 2., 
                                                -1., 0., 1.};

        float KySobel[KSobelSize*KSobelSize] = {1., 2., 1., 
                                                0., 0., 0., 
                                            -1.,-2.,-1.};

    //do X Sobel convolution
        doConvoSeq(xSobel_pixarr, blurred_pixarr, KxSobel, IMGwidth, IMGheight, KSobelSize);
        doConvoSeq(ySobel_pixarr, blurred_pixarr, KySobel, IMGwidth, IMGheight, KSobelSize);

    if (printPPMs){
        //print results
        char ppm_xSobel[20]= "3_xSobel_s.ppm";
        status = writePPM(xSobel_pixarr, graysize, ppm_xSobel, 0);
        char ppm_ySobel[20] = "4_ySobel_s.ppm";
        status = writePPM(ySobel_pixarr, graysize, ppm_ySobel, 0);  
    }

    //this is to limit eventual hypot result (175^2 + 175^2) =~ 247
    //this helps reduce saturation & intensity of the sobel result
        for (int i=0; i < graysize; i++){
        ySobel_pixarr[i] = clamp_175(ySobel_pixarr[i]);
        xSobel_pixarr[i] = clamp_175(xSobel_pixarr[i]);
        uint32_t prod = (ySobel_pixarr[i]*ySobel_pixarr[i])+(xSobel_pixarr[i]*xSobel_pixarr[i]);
        finalSobel_pixarr[i] = (uint8_t)sqrt(prod);
        }

        gettimeofday(&end, NULL);
        printf("Sobel_seq Run:\n");
        printf("    Start: %ld us    End: %ld us\n", start.tv_usec, end.tv_usec);
        t_us = (end.tv_sec - start.tv_sec)*1000000 + end.tv_usec - start.tv_usec;
        printf("    Elapsed time: %ld us\n", t_us);
        t_us_total += t_us;
        
    if (printPPMs){
    //FINAL SOBEL PRINT
            char ppm_finalSobel[20] = "5_finalSobel_s.ppm";
            status = writePPM(finalSobel_pixarr, graysize, ppm_finalSobel, 0);
    }

    /////////////////////////////////////////////////////////////////////////////////////
    ///// Step 4: Perform Non-Maximum Suppression ///////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////
    /*
            Find raw gradient direction, and confine to ranges of 0deg (WW to EE), 45/135deg (SW to NE, NW to SE), 90deg (SS to NN).
            - Anything less than 22.5 degrees (0.3927 rad) is considered to be in the 0deg/180deg (WW/EE) direction
            - Anything in between 22.5 degrees(0.3927 rad) and 67.5 (1.1781 rad) is considered to be in the 45/135degdirection(SW to NE, NW to SE)
            - Anything more than 67.5 degrees (1.1781 rad) is considered to be in the 90deg/270deg (NN/SS) direction
            If a pixel is in that direction, and it is the local max to its neighbors in that direction, we keep that pixel

                    NW    NN    NE

                    WW    C     EE

                    SW    SS    SE
    */
        
        gettimeofday(&start, NULL);

        //because of the side effect of gaussian blur darkening edges of image, we will ignore the border pixels.
        for (int i=2; i < IMGheight-2; i++){
            for (int j=2; j < IMGwidth-2; j++){

                //take current center of pixel, and derive all neighboring indices from it
                int center = IMGwidth*i + j;
                int NN = center - IMGwidth;
                int SS = center + IMGwidth;
                int WW = center + 1;
                int EE = center - 1;
                int NW = NN + 1;
                int NE = NN - 1;
                int SW = SS + 1;
                int SE = SS - 1;

                float dir = atan2(ySobel_pixarr[center], xSobel_pixarr[center]);

                if
                (
                    /*
                    ( (dir < 0.3927) && (finalSobel_pixarr[center] >= finalSobel_pixarr[EE] && finalSobel_pixarr[center] >= finalSobel_pixarr[WW])) ||
                    ( (dir > 0.3927) && (dir < 1.1781) && (finalSobel_pixarr[center] >= finalSobel_pixarr[SS] && finalSobel_pixarr[center] >= finalSobel_pixarr[NN])) ||
                    ( (dir > 1.1781) && ((finalSobel_pixarr[center] >= finalSobel_pixarr[SW] && finalSobel_pixarr[center] >= finalSobel_pixarr[NE]) && (finalSobel_pixarr[center] >= finalSobel_pixarr[SE] && finalSobel_pixarr[center] >= finalSobel_pixarr[NW])))
                    */
                    
                    ( (dir < 0.3927) && (finalSobel_pixarr[center] >= finalSobel_pixarr[EE] && finalSobel_pixarr[center] >= finalSobel_pixarr[WW])) ||
                    ( (dir > 0.3927) && (dir < 1.1781) && ( (finalSobel_pixarr[center] >= finalSobel_pixarr[SW] && finalSobel_pixarr[center] >= finalSobel_pixarr[NE]) || (finalSobel_pixarr[center] >= finalSobel_pixarr[SE] && finalSobel_pixarr[center] >= finalSobel_pixarr[NW]))) ||
                    ( (dir > 1.1781) && (finalSobel_pixarr[center] >= finalSobel_pixarr[SS] && finalSobel_pixarr[center] >= finalSobel_pixarr[NN]))
                )
                    afterNMS[center] = finalSobel_pixarr[center];
                else
                    afterNMS[center] = 0;
            }
        }

        gettimeofday(&end, NULL);
        printf("NMS_seq Run:\n");
        printf("    Start: %ld us    End: %ld us\n", start.tv_usec, end.tv_usec);
        t_us = (end.tv_sec - start.tv_sec)*1000000 + end.tv_usec - start.tv_usec;
        printf("    Elapsed time: %ld us\n", t_us);
        t_us_total += t_us;

    //after NMS, we need to classify pixels as strong, weak or non-border pixels
    //pixels above high threshold are STRONG border pixels
    //pixels in between high and low thresholds are WEAK pixels
    //pixels below low threshold are NOT border pixels
        uint8_t highThreshold = (uint8_t)(255*0.3);
        uint8_t lowThreshold = (uint8_t)(highThreshold*0.75);

    if (printPPMs){
        //print out NMS results for PPM 
        char ppm_nms[20] = "6_nms_s.ppm";
        status = writePPM(afterNMS, graysize, ppm_nms, 0);

        //here we classify our edges if they're strong or not
        for (int i=0; i < IMGheight; i++){
            for (int j=0; j < IMGwidth; j++){
                int c = IMGwidth*i + j;
                if (afterNMS[c] > highThreshold)
                    PixelType[c] = STRONG_EDGE;
                else if ((afterNMS[c] <= highThreshold) && (afterNMS[c] > lowThreshold))
                    PixelType[c] = WEAK_EDGE;
                else
                    PixelType[c] = NO_EDGE;
            }
        }
        for (int i=0; i < IMGheight; i++){
            for (int j=0; j < IMGwidth; j++){
                int c = IMGwidth*i + j;
                if (PixelType[c] == NO_EDGE)
                    afterDthr[c] = 0;
                else
                    afterDthr[c] = afterNMS[c];
            }
        }
        //this displays all pixels that aren't NO_EDGE pixels
        //char ppm_dthr[20] = "7_Strong_s.ppm";
        //status = writePPM(afterDthr, graysize, ppm_dthr, 0);
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
        //this displays weak edges as gray, strong pixels as white
        char ppm_dthr2[20] = "7_dualthresh_s.ppm";
        status = writePPM(afterDthr2, graysize, ppm_dthr2, 0);
    }
    /////////////////////////////////////////////////////////////////////////////////////////////////////
    ///// Step 5: Perform Double Thresholding & Hysteresis Thresholding /////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////////////////////

        gettimeofday(&start, NULL);

        //pixels above high threshold are STRONG border pixels
        //pixels in between high and low thresholds are WEAK pixels
        //pixels below low threshold are NOT border pixels
        
        for (int i=2; i < IMGheight-2; i++){
            for (int j=2; j < IMGwidth-2; j++){

                //take current center of pixel, and derive all neighboring indices from it
                int center = IMGwidth*i + j;
                int NN = center - IMGwidth;
                int SS = center + IMGwidth;
                int WW = center + 1;
                int EE = center - 1;
                int NW = NN + 1;
                int NE = NN - 1;
                int SW = SS + 1;
                int SE = SS - 1;
                if (afterNMS[center] < lowThreshold)        //suppress no edges
                    afterHyst[center] = 0;

                //now we need to consider others edges. If any other surrounding edges are strong, saturate the weak edge as you would a strong edge
                else if (
                    (afterNMS[center] > highThreshold) || (afterNMS[NN] > highThreshold) || (afterNMS[SS] > highThreshold) || 
                    (afterNMS[EE] > highThreshold) || (afterNMS[WW] > highThreshold) ||
                    (afterNMS[NE] > highThreshold) || (afterNMS[NW] > highThreshold) || 
                    (afterNMS[SE] > highThreshold) || (afterNMS[SW] > highThreshold)
                )
                    afterHyst[center] = 255;

                //else, we remove weak edge
                else
                    afterHyst[center] = 0;
            }
        }


        gettimeofday(&end, NULL);
        printf("Thresh&Hysteresis_seq Run:\n");
        printf("    Start: %ld us    End: %ld us\n", start.tv_usec, end.tv_usec);
        t_us = (end.tv_sec - start.tv_sec)*1000000 + end.tv_usec - start.tv_usec;
        printf("    Elapsed time: %ld us\n", t_us);
        t_us_total += t_us;
        printf("Full Sequential Run:\n");
        printf("    Total Elapsed time: %ld us\n", t_us_total);

        //print out PPM for NMS
        char ppm_hyst[20] = "OUT_canny_s.ppm";
        status = writePPM(afterHyst, graysize, ppm_hyst, 0);

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

uint8_t doThresh_tbb(uint8_t* afterNMS, int i, int j, uint8_t highThreshold, uint8_t lowThreshold){
    int center = IMGwidth*i + j;
    int NN = center - IMGwidth;
    int SS = center + IMGwidth;
    int WW = center + 1;
    int EE = center - 1;
    int NW = NN + 1;
    int NE = NN - 1;
    int SW = SS + 1;
    int SE = SS - 1;
    if (afterNMS[center] < lowThreshold)        //suppress no edges
        return 0;
    else if (afterNMS[center] > highThreshold) //saturate strong edges
        return 255;

    //now we need to consider weak edges. If any other surrounding edges are strong, saturate the weak edge as you would a strong edge
    else if (
        (afterNMS[NN] > highThreshold) || (afterNMS[SS] > highThreshold) || 
        (afterNMS[EE] > highThreshold) || (afterNMS[WW] > highThreshold) ||
        (afterNMS[NE] > highThreshold) || (afterNMS[NW] > highThreshold) || 
        (afterNMS[SE] > highThreshold) || (afterNMS[SW] > highThreshold)
    )
        return 255;
    //else, we remove weak edge
    else
        return 0;

}


/*
gettimeofday(&start, NULL);

        //because of the side effect of gaussian blur darkening edges of image, we will ignore the border pixels.
        for (int i=2; i < IMGheight-2; i++){
            for (int j=2; j < IMGwidth-2; j++){

                //take current center of pixel, and derive all neighboring indices from it
                int center = IMGwidth*i + j;
                int NN = center - IMGwidth;
                int SS = center + IMGwidth;
                int WW = center + 1;
                int EE = center - 1;
                int NW = NN + 1;
                int NE = NN - 1;
                int SW = SS + 1;
                int SE = SS - 1;

                float dir = atan2(ySobel_pixarr[center], xSobel_pixarr[center]);

                if
                (
                    ( (dir < 0.3927) && (finalSobel_pixarr[center] >= finalSobel_pixarr[EE] && finalSobel_pixarr[center] >= finalSobel_pixarr[WW])) ||
                    ( (dir > 0.3927) && (dir < 1.1781) && (finalSobel_pixarr[center] >= finalSobel_pixarr[SS] && finalSobel_pixarr[center] >= finalSobel_pixarr[NN])) ||
                    ( (dir > 1.1781) && ( (finalSobel_pixarr[center] >= finalSobel_pixarr[SW] && finalSobel_pixarr[center] >= finalSobel_pixarr[NE]) && (finalSobel_pixarr[center] >= finalSobel_pixarr[SE] && finalSobel_pixarr[center] >= finalSobel_pixarr[NW])))
                )
                    afterNMS[center] = finalSobel_pixarr[center];
                else
                    afterNMS[center] = 0;
            }
        }

*/

uint8_t doNMS_tbb(uint8_t* finalSobel_pixarr, uint8_t* xSobel_pixarr, uint8_t* ySobel_pixarr, int i, int j){
    int center = IMGwidth*i + j;
    int NN = center - IMGwidth;
    int SS = center + IMGwidth;
    int WW = center + 1;
    int EE = center - 1;
    int NW = NN + 1;
    int NE = NN - 1;
    int SW = SS + 1;
    int SE = SS - 1;
    float dir = atan2(ySobel_pixarr[center], xSobel_pixarr[center]);
    if
    (
        ( (dir < 0.3927) && (finalSobel_pixarr[center] >= finalSobel_pixarr[EE] && finalSobel_pixarr[center] >= finalSobel_pixarr[WW])) ||
        ( (dir > 0.3927) && (dir < 1.1781) && ( (finalSobel_pixarr[center] >= finalSobel_pixarr[SW] && finalSobel_pixarr[center] >= finalSobel_pixarr[NE]) && (finalSobel_pixarr[center] >= finalSobel_pixarr[SE] && finalSobel_pixarr[center] >= finalSobel_pixarr[NW]))) ||
        ( (dir > 1.1781) && finalSobel_pixarr[center] >= finalSobel_pixarr[SS] && finalSobel_pixarr[center] >= finalSobel_pixarr[NN])
    )
        return finalSobel_pixarr[center];
    else
        return 0;   
}

uint8_t getSobelMagnitude_TBB(uint8_t* xSobel_pixarr, uint8_t* ySobel_pixarr, int i){
    uint8_t x = clamp_175(xSobel_pixarr[i]);
    uint8_t y = clamp_175(xSobel_pixarr[i]);  
    uint32_t prod = (y*y)+(x*x);
    return (uint8_t)sqrt(prod);
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

void doConvoTBB(uint8_t* output, uint8_t *IMGarr, float* K, int IMGwidth, int IMGheight, int Ksize){

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
}

uint8_t doGrayscaleTBB (uint8_t* oldarr, int i){
    uint8_t r = oldarr[(3*i)];
    uint8_t g = oldarr[(3*i)+1];
    uint8_t b = oldarr[(3*i)+2];
    uint8_t newval = 0.21f*r + 0.71f*g + 0.07f*b;
    return newval;
}


int CannyTBB(int printPPMs)
{
        int status; //used for some prints
        //NEED a ppm file named img.ppm in the same directory, which is set up to be read from
        FILE *fp;
        if ( !(fp = fopen("img.ppm", "r"))){
            printf("ERROR: Need a .ppm (P6) file named 'img.ppm' in same directory as executable\n");
            exit(-1);
        }

        //get header data (width and height of image) from PPM input
        readPPMHeader(fp);
        printf("IMG WIDTH = %d\n", IMGwidth);
        printf("IMG HEIGHT = %d\n", IMGheight);

        //PPM uses 1R, 1G, 1B per pixel
        //only keep track of one pixel for grayscale calculations
        int colorsize = (IMGwidth*IMGheight)*3;      
        int graysize = IMGheight*IMGwidth;

        //all memory allocations
        //x and y sobel are split up to print separate .PPMs for each
        uint8_t* pixarr = (uint8_t*)malloc(colorsize);
        uint8_t* grayed_pixarr = (uint8_t*)malloc(graysize);
        uint8_t* blurred_pixarr = (uint8_t*)malloc(graysize);
        uint8_t* xSobel_pixarr = (uint8_t*)malloc(graysize*sizeof(uint8_t));
        uint8_t* ySobel_pixarr = (uint8_t*)malloc(graysize*sizeof(uint8_t));
        uint8_t* finalSobel_pixarr = (uint8_t*)malloc(graysize*sizeof(uint8_t));
        uint8_t* afterNMS = (uint8_t*)calloc(graysize, sizeof(uint8_t));
        uint8_t* afterHyst = (uint8_t*)calloc(graysize, sizeof(uint8_t));
        
        EDGE_TYPE* PixelType = (EDGE_TYPE*)malloc(graysize*sizeof(EDGE_TYPE));
        uint8_t* afterDthr = (uint8_t*)calloc(graysize, sizeof(uint8_t));
        uint8_t* afterDthr2 = (uint8_t*)calloc(graysize, sizeof(uint8_t));
        

        //get pixel data from PPM input
        status = readPPMData(fp, pixarr, colorsize);
        printf("IMG COUNT = %d\n", status);
        (void) fclose(fp);

    if (printPPMs){
        //write back original color image with original pixel data
        char ppm_nochange[20] = "0_original_p.ppm";
        status = writePPM(pixarr, colorsize, ppm_nochange, 1);
        printf("OUTIMG COUNT = %d\n", status);
    }

    #ifdef USE_TIMES
        struct timeval start, end; 
        long t_us, t_us_total = 0;
    #endif


    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    ///// Step 1: Convert to GRAYSCALE ////////////////////////////////////////////////////////////////////////////////
    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        gettimeofday(&start, NULL);
            tbb::parallel_for(int(0), int(graysize), [&] (int i){
                grayed_pixarr[i] = doGrayscaleTBB(pixarr, i);
            });
        gettimeofday(&end, NULL);
        printf("Grayscale_TBB Run:\n");
        printf("    Start: %ld us    End: %ld us\n", start.tv_usec, end.tv_usec);
        t_us = (end.tv_sec - start.tv_sec)*1000000 + end.tv_usec - start.tv_usec;
        printf("    Elapsed time: %ld us\n", t_us);
        t_us_total += t_us;

    if (printPPMs){
        //print out grayscale image
        char ppm_grayed[20] = "1_grayscale_p.ppm";
        status = writePPM(grayed_pixarr, graysize, ppm_grayed, 0);
    }


    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    ///// Step 2: Perform Gaussian Blur ///////////////////////////////////////////////////////////////////////////////
    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        gettimeofday(&start, NULL);

    float KGaussF[KGaussSize*KGaussSize]= {2., 4.,  5.,  4.,  2., 
                                4., 9.,  12., 9.,  4., 
                                5., 12., 15., 12., 5., 
                                4., 9.,  12., 9.,  4., 
                                2., 4.,  5.,  4.,  2};     
    //treat gauss
        for (int i = 0; i <KGaussSize*KGaussSize; i++)
            KGaussF[i] = KGaussF[i]/159.;

    //GAUSSIAN BLUR CONVOLUTION
        doConvoTBB(blurred_pixarr, grayed_pixarr, KGaussF, IMGwidth, IMGheight, KGaussSize);

        gettimeofday(&end, NULL);
        printf("GaussBlur_TBB Run:\n");
        printf("    Start: %ld us    End: %ld us\n", start.tv_usec, end.tv_usec);
        t_us = (end.tv_sec - start.tv_sec)*1000000 + end.tv_usec - start.tv_usec;
        printf("    Elapsed time: %ld us\n", t_us);
        t_us_total += t_us;

    if (printPPMs){
    //print blurred image
        char ppm_blurred[20] = "2_blurred_p.ppm";
        status = writePPM(blurred_pixarr, graysize, ppm_blurred, 0);
    }
    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    ///// Step 3: Perform Sobel Edge Detection ////////////////////////////////////////////////////////////////////////
    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        gettimeofday(&start, NULL);

        float KxSobel[KSobelSize*KSobelSize] = {-1., 0., 1., 
                                                -2., 0., 2., 
                                                -1., 0., 1.};

        float KySobel[KSobelSize*KSobelSize] = {1., 2., 1., 
                                                0., 0., 0., 
                                            -1.,-2.,-1.};

    //do X Sobel convolution
        doConvoTBB(xSobel_pixarr, blurred_pixarr, KxSobel, IMGwidth, IMGheight, KSobelSize);
        doConvoTBB(ySobel_pixarr, blurred_pixarr, KySobel, IMGwidth, IMGheight, KSobelSize);

    if (printPPMs){
        //print results
        char ppm_xSobel[20]= "3_xSobel_p.ppm";
        status = writePPM(xSobel_pixarr, graysize, ppm_xSobel, 0);
        char ppm_ySobel[20] = "4_ySobel_p.ppm";
        status = writePPM(ySobel_pixarr, graysize, ppm_ySobel, 0);  
    }


    //this is to limit eventual hypot result (175^2 + 175^2) =~ 247
    //this helps reduce saturation & intensity of the sobel result
        tbb::parallel_for(int(0), int(graysize), [&] (int i){
            finalSobel_pixarr[i] = getSobelMagnitude_TBB(xSobel_pixarr, ySobel_pixarr, i);
        });
        // for (int i=0; i < graysize; i++){
        // ySobel_pixarr[i] = clamp_175(ySobel_pixarr[i]);
        // xSobel_pixarr[i] = clamp_175(xSobel_pixarr[i]);
        // uint32_t prod = (ySobel_pixarr[i]*ySobel_pixarr[i])+(xSobel_pixarr[i]*xSobel_pixarr[i]);
        // finalSobel_pixarr[i] = (uint8_t)sqrt(prod);
        // }

        gettimeofday(&end, NULL);
        printf("Sobel_TBB Run:\n");
        printf("    Start: %ld us    End: %ld us\n", start.tv_usec, end.tv_usec);
        t_us = (end.tv_sec - start.tv_sec)*1000000 + end.tv_usec - start.tv_usec;
        printf("    Elapsed time: %ld us\n", t_us);
        t_us_total += t_us;
        
    if (printPPMs){
    //FINAL SOBEL PRINT
            char ppm_finalSobel[20] = "5_finalSobel_p.ppm";
            status = writePPM(finalSobel_pixarr, graysize, ppm_finalSobel, 0);
    }

    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    ///// Step 4: Perform Non-Maximum Suppression /////////////////////////////////////////////////////////////////////
    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    /*
            Find raw gradient direction, and confine to ranges of 0deg (WW to EE), 45/135deg (SW to NE, NW to SE), 90deg (SS to NN).
            - Anything less than 22.5 degrees (0.3927 rad) is considered to be in the 0deg/180deg (WW/EE) direction
            - Anything in between 22.5 degrees(0.3927 rad) and 67.5 (1.1781 rad) is considered to be in the 45/135degdirection(SW to NE, NW to SE)
            - Anything more than 67.5 degrees (1.1781 rad) is considered to be in the 90deg/270deg (NN/SS) direction
            If a pixel is in that direction, and it is the local max to its neighbors in that direction, we keep that pixel

                    NW    NN    NE

                    WW    C     EE

                    SW    SS    SE
    */

        gettimeofday(&start, NULL);
        int IMGheight_trunc = IMGheight-2;
        int IMGwidth_trunc = IMGwidth-2;
        tbb::parallel_for(int(2), int(IMGheight_trunc), [&] (int i){
            tbb::parallel_for(int(2), int(IMGwidth_trunc), [&] (int j){
                afterNMS[IMGwidth*i + j] = doNMS_tbb(finalSobel_pixarr, xSobel_pixarr, ySobel_pixarr, i, j);
            });    
        });


        gettimeofday(&end, NULL);
        printf("NMS_tbb Run:\n");
        printf("    Start: %ld us    End: %ld us\n", start.tv_usec, end.tv_usec);
        t_us = (end.tv_sec - start.tv_sec)*1000000 + end.tv_usec - start.tv_usec;
        printf("    Elapsed time: %ld us\n", t_us);
        t_us_total += t_us;

    //after NMS, we need to classify pixels as strong, weak or non-border pixels
    //pixels above high threshold are STRONG border pixels
    //pixels in between high and low thresholds are WEAK pixels
    //pixels below low threshold are NOT border pixels
        uint8_t highThreshold = (uint8_t)(255*0.3);
        uint8_t lowThreshold = (uint8_t)(highThreshold*0.75);

    if (printPPMs){
        //print out NMS results for PPM 
        char ppm_nms[20] = "6_nms_p.ppm";
        status = writePPM(afterNMS, graysize, ppm_nms, 0);

        //here we classify our edges if they're strong or not
        for (int i=0; i < IMGheight; i++){
            for (int j=0; j < IMGwidth; j++){
                int c = IMGwidth*i + j;
                if (afterNMS[c] > highThreshold)
                    PixelType[c] = STRONG_EDGE;
                else if ((afterNMS[c] <= highThreshold) && (afterNMS[c] > lowThreshold))
                    PixelType[c] = WEAK_EDGE;
                else
                    PixelType[c] = NO_EDGE;
            }
        }
        for (int i=0; i < IMGheight; i++){
            for (int j=0; j < IMGwidth; j++){
                int c = IMGwidth*i + j;
                if (PixelType[c] == NO_EDGE)
                    afterDthr[c] = 0;
                else
                    afterDthr[c] = afterNMS[c];
            }
        }
        //this displays all pixels that aren't NO_EDGE pixels
        //char ppm_dthr[20] = "7_Strong_s.ppm";
        //status = writePPM(afterDthr, graysize, ppm_dthr, 0);
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
        //this displays weak edges as gray, strong pixels as white
        char ppm_dthr2[20] = "7_dualthresh_p.ppm";
        status = writePPM(afterDthr2, graysize, ppm_dthr2, 0);
    }
    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    ///// Step 5: Perform Double Thresholding & Hysteresis Thresholding ///////////////////////////////////////////////////
    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


        gettimeofday(&start, NULL);

        tbb::parallel_for(int(2), int(IMGheight_trunc), [&] (int i){
            tbb::parallel_for(int(2), int(IMGwidth_trunc), [&] (int j){
                afterHyst[IMGwidth*i + j] = doThresh_tbb(afterNMS, i, j, highThreshold, lowThreshold);
            });    
        });

        gettimeofday(&end, NULL);
        printf("Thresh&Hysteresis_tbb Run:\n");
        printf("    Start: %ld us    End: %ld us\n", start.tv_usec, end.tv_usec);
        t_us = (end.tv_sec - start.tv_sec)*1000000 + end.tv_usec - start.tv_usec;
        printf("    Elapsed time: %ld us\n", t_us);
        t_us_total += t_us;
        printf("Full Parallel(tbb) Run:\n");
        printf("    Total Elapsed time: %ld us\n", t_us_total);

        //print out PPM for NMS
        char ppm_hyst[20] = "OUT_canny_p.ppm";
        status = writePPM(afterHyst, graysize, ppm_hyst, 0);

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
