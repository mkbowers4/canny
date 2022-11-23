#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>
#include <string.h>
#include <math.h>

#define K1size 5
#define K2size 3

//#define doPPMprints

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

int writePGM (uint16_t* arr, int size, char* FILENAME, int mode){

    //mode = 0 for color data, mode = 1 for grayscaled data

    int count;
    FILE *fp_wr = fopen(FILENAME, "wb"); // b - binary mode
    fprintf(fp_wr, "P5 %d %d 65535\n", IMGwidth, IMGheight);

    uint16_t temp;
        for (int i=0; i < IMGwidth*IMGheight; i++){
            temp = arr[i];
            fwrite(&temp, 2, 1, fp_wr);
        }
    (void) fclose(fp_wr);

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


int doConvo(uint8_t* output, uint8_t *IMGarr, float* K, int IMGwidth, int IMGheight, int Ksize){

  float sum;
  int sumtemp;
  int i, j, m, n, mflip, nflip, iflip, jflip;
  //i,j for data,
  //m,n for kernel,
  //mflip,mflip for flipped kernel
  int kCX = Ksize/2;
  int kCY = Ksize/2;

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

int main (int argc, char *argv[]) {

    //time stuff
    struct timeval start, end;
    long t_us;

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


    //CONVERSION TO GRAYSCALE
    int graysize = IMGwidth*IMGheight;
    uint8_t* grayed_pixarr = (uint8_t*)malloc(graysize);
    status = doGrayscale(pixarr, grayed_pixarr, graysize);

#ifdef doPPMprints
    //print out grayscale image
    char ppm_grayed[20] = "OUT1_grayscale.ppm";
    status = writePPM(grayed_pixarr, graysize, ppm_grayed, 0);
#endif

//GAUSSIAN BLUR

    //int KGauss[K1size*K1size] = {1, 4, 6, 4, 1, 4, 16, 24, 16, 4, 6, 24, 36, 24, 6, 4, 16, 24, 16, 4, 1, 4, 6, 4, 1};   //use 256
    int KGauss[K1size*K1size] = {2, 4, 5, 4, 2, 4, 9, 12, 9, 4, 5, 12, 15, 12, 5, 4, 9, 12, 9, 4, 2, 4, 5, 4, 2};     //use 1se 159
    float KGaussF[K1size*K1size];

    //treat gauss
    for (int i = 0; i <K1size*K1size; i++)
        KGaussF[i] = ((float)KGauss[i]/159.);


gettimeofday (&start, NULL);


    //GAUSSIAN BLUR CONVOLUTION
    uint8_t* blurred_pixarr = (uint8_t*)malloc(graysize);
    status = doConvo(blurred_pixarr, grayed_pixarr, KGaussF, IMGwidth, IMGheight, K1size);

    //print blurred image
#ifdef doPPMprints
    char ppm_blurred[20] = "OUT2_blurred.ppm";
    status = writePPM(blurred_pixarr, graysize, ppm_blurred, 0);
#endif
//SOBEL

    //float KxSobel[K2size*K2size] = {-3., 0., 3., -10., 0., 10., -3., 0., 3.};
    //float KySobel[K2size*K2size] = {3., 10., 3., 0., 0., 0., -3., -10., -3.};
    float KxSobel[K2size*K2size] = {-1., 0., 1., -2., 0., 2., -1., 0., 1.};
    float KySobel[K2size*K2size] = {1., 2., 1., 0., 0., 0., -1., -2., -1.};
    uint8_t* xSobel_pixarr = (uint8_t*)malloc(graysize*sizeof(uint8_t));
    uint8_t* ySobel_pixarr = (uint8_t*)malloc(graysize*sizeof(uint8_t));
    uint8_t* finalSobel_pixarr = (uint8_t*)malloc(graysize*sizeof(uint8_t));

    //XSOBEL + print

        status = doConvo(xSobel_pixarr, blurred_pixarr, KxSobel, IMGwidth, IMGheight, K2size);
#ifdef doPPMprints
        char ppm_xSobel[20]= "OUT4_xSobel.ppm";
        status = writePPM(xSobel_pixarr, graysize, ppm_xSobel, 0);
#endif
    //YSOBEL + print

        status = doConvo(ySobel_pixarr, blurred_pixarr, KySobel, IMGwidth, IMGheight, K2size);
#ifdef doPPMprints
        char ppm_ySobel[20] = "OUT5_ySobel.ppm";
        status = writePPM(ySobel_pixarr, graysize, ppm_ySobel, 0);
#endif

    //suppress absurd highs
    for (int i=0; i < graysize; i++){
        if (ySobel_pixarr[i] > 175)
            ySobel_pixarr[i] = 175;   
        if (xSobel_pixarr[i] > 175)
            xSobel_pixarr[i] = 175; 
    }

    for (int i=0; i < graysize; i++){
        uint32_t prod = ((uint32_t)ySobel_pixarr[i]*ySobel_pixarr[i])+((uint32_t)xSobel_pixarr[i]*xSobel_pixarr[i]);

            //saturating (some very strong edges were getting turned over
            if (sqrt(prod) > 255.0)
                finalSobel_pixarr[i] = 255;
            else
                finalSobel_pixarr[i] = (uint8_t)sqrt(prod);
    }

    //simpler adding of sobel results
    /*
    for (int i=0; i < graysize; i++){
        if (ySobel_pixarr[i] + xSobel_pixarr[i] > 255)
            finalSobel_pixarr[i] = 255;
        else
            finalSobel_pixarr[i] = ySobel_pixarr[i] + xSobel_pixarr[i];
    }
    */
#ifdef doPPMprints
    //FINAL SOBEL PRINT
        char ppm_finalSobel[20] = "OUT5_finalSobel.ppm";   
        status = writePPM(finalSobel_pixarr, graysize, ppm_finalSobel, 0);
#endif

//NON-MAXIMUM SUPPRESSION

    /*

        - Find raw gradient direction, and confine to ranges of 0° (WW to EE), 45/135° (SW to NE, NW to SE), 90° (SS to NN)
        -if the gradient direction is in a particular location (0°, 45/135°, 90°), then check if the center pixel is smaller than its neighbors
        (checking for a local maximum. If it is a local maximum, we keep it)
            
                NW     NN     NE
                    
                WW     C      EE

                SW     SS     SE

         //OLD!!!!!!!
        gradient mapping numbers:
            -22.5/337.5° to 22.5° OR 157.5° to 202.5° ==> 0/180°
                (-0.3927rad/5.8905rad to 0.3927rad) OR (2.7489rad to 3.5343rad)
                                                                       -2.74889
            22.5° to 67.5° OR 202.5° to 247.5° ==> 45°
                (0.3927rad to 1.1781rad) OR (3.5343rad to 4.3197rad)
                                            -2.74889        -1.96549
            67.5° to 112.5° OR  247.5° to 292.5° ==> 90°
                (1.1781rad to 1.9635rad) OR (4.3197rad to 5.1051rad)
                                            -1.96549   to  -1.18168
            112.5° to 157.5° OR 292.5° to -22.5/337.5° ==> 135°
                (1.9635rad to 2.7489rad) OR (5.1051rad to -0.3927rad/5.8905rad)
                                            -1.18168      -0.3927rad
     */



    uint8_t* afterNMS = (uint8_t*)calloc(graysize, sizeof(uint8_t));

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

            //OLD (SIGNED CONVERSION)
            //int8_t xtemp = xSobel_pixarr[c] - 128;
            //int8_t ytemp = ySobel_pixarr[c] - 128;
            //float dir = atan2(ytemp, xtemp);            //range of -pi (-3.141593...) to pi (3.141593...)
            float dir = atan2(ySobel_pixarr[c], xSobel_pixarr[c]);            
            
            if 
             
                ( 

                ( (dir < 0.3927) && (finalSobel_pixarr[c] >= finalSobel_pixarr[ee] && finalSobel_pixarr[c] >= finalSobel_pixarr[ww])) || 
                ( (dir > 0.3927) && (dir < 1.1781) && (finalSobel_pixarr[c] >= finalSobel_pixarr[ss] && finalSobel_pixarr[c] >= finalSobel_pixarr[nn])) ||
                ( (dir > 1.1781) && ( (finalSobel_pixarr[c] >= finalSobel_pixarr[sw] && finalSobel_pixarr[c] >= finalSobel_pixarr[ne]) && (finalSobel_pixarr[c] >= finalSobel_pixarr[se] && finalSobel_pixarr[c] >= finalSobel_pixarr[nw])))

                //OLD (SIGNED CONVERSION)
                // ( ((dir > -0.3927 && dir < 0.3927) || (dir > 2.7489 || dir < -2.7489)) && (finalSobel_pixarr[c] >= finalSobel_pixarr[ee] && finalSobel_pixarr[c] >= finalSobel_pixarr[ww])) ||
                // ( ((dir > 0.3927 && dir < 1.1781) || (dir > -2.7489 && dir < -1.9655)) && (finalSobel_pixarr[c] >= finalSobel_pixarr[nw] && finalSobel_pixarr[c] >= finalSobel_pixarr[se])) ||
                // ( ((dir > 1.1781 && dir < 1.9635) || (dir > -1.9655 && dir < -1.1817)) && (finalSobel_pixarr[c] >= finalSobel_pixarr[nn] && finalSobel_pixarr[c] >= finalSobel_pixarr[ss])) ||
                // ( ((dir > 1.9635 && dir < 2.7489) || (dir > -1.1817 && dir < -0.3927)) && (finalSobel_pixarr[c] >= finalSobel_pixarr[ne] && finalSobel_pixarr[c] >= finalSobel_pixarr[sw]))  
            )
                 afterNMS[c] = finalSobel_pixarr[c];
            else 
                 afterNMS[c] = 0; 
        }
    }

#ifdef doPPMprints
    //print out PPM for NMS
    char ppm_nms[20] = "OUT_nms.ppm";
    status = writePPM(afterNMS, graysize, ppm_nms, 0);
#endif
   
//DOUBLE THRESHOLDING
    //here we classify our edges if they're strong or not 

    EDGE_TYPE* PixelType = (EDGE_TYPE*)malloc(graysize*sizeof(EDGE_TYPE));
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

#ifdef doPPMprints
//////////////JUST FOR PRINTING
    uint8_t* afterDthr = (uint8_t*)calloc(graysize, sizeof(uint8_t));

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

    uint8_t* afterDthr2 = (uint8_t*)calloc(graysize, sizeof(uint8_t));

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

/////////////////JUST FOR PRINTING
#endif




//HYSTERESIS
    uint8_t* afterHyst = (uint8_t*)calloc(graysize, sizeof(uint8_t));

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
gettimeofday (&end, NULL);
    //print out PPM for NMS
    char ppm_hyst[20] = "OUT_hyst.ppm";
    status = writePPM(afterHyst, graysize, ppm_hyst, 0);
    printf("done.");

printf ("start: %ld us\n", start.tv_usec); // start.tv_sec
printf ("end: %ld us\n", end.tv_usec);    // end.tv_sec;
t_us = (end.tv_sec - start.tv_sec)*1000000 + end.tv_usec - start.tv_usec; // for ms: define t_ms as double and divide by 1000.0
// gettimeofday: returns current time. So, when the secs increment, the us resets to 0.
printf ("Elapsed time: %ld us\n", t_us);



    free(pixarr);
    free(PixelType);
    free(afterNMS);
    free(afterHyst);
    free(grayed_pixarr);
    free(blurred_pixarr);
    free(xSobel_pixarr);
    free(ySobel_pixarr);
    free(finalSobel_pixarr);
}









































