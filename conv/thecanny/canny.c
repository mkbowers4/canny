#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>
#include <string.h>
#include <math.h>

#define K1size 5
#define K2size 3

int IMGwidth, IMGheight;

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

int doGrayscale (uint8_t* oldarr, uint8_t* newarr, int newsize){
    uint8_t r, g, b;
    int i;
    for (i=0; i<newsize; i++){
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

    //hardcoded kernels

    //create kernels
    int status;
    uint8_t* pixarr;
    uint8_t* grayed_pixarr;
    uint8_t* blurred_pixarr;
    FILE *fp = fopen("img.ppm", "r");

    //get header data & read parameters back
    readPPMHeader(fp);
    printf("IMG WIDTH = %d\n", IMGwidth);
    printf("IMG HEIGHT = %d\n", IMGheight);

    int size = (IMGwidth*IMGwidth)*3;      //a set of RGB values per pixel
    pixarr = (uint8_t*)malloc(size);

    //get pixel data
    status = readPPMData(fp, pixarr, size);
    printf("IMG COUNT = %d\n", status);
    (void) fclose(fp);

    //write back original image with original pixel data
    char ppm_nochange[20] = "OUT0_unchanged.ppm";
    status = writePPM(pixarr, size, ppm_nochange, 1);
    printf("OUTIMG COUNT = %d\n", status);

    int newsize = IMGwidth*IMGwidth;
    grayed_pixarr = (uint8_t*)malloc(newsize);



    //CONVERSION TO GRAYSCALE

    status = doGrayscale(pixarr, grayed_pixarr, newsize);

    free(pixarr);

    char ppm_grayed[20] = "OUT1_grayscale.ppm";
    status = writePPM(grayed_pixarr, newsize, ppm_grayed, 0);

    //GAUSSIAN BLUR
    //int KGauss[K1size*K1size] = {1, 4, 6, 4, 1, 4, 16, 24, 16, 4, 6, 24, 36, 24, 6, 4, 16, 24, 16, 4, 1, 4, 6, 4, 1};   //use 256
    int KGauss[K1size*K1size] = {2, 4, 5, 4, 2, 4, 9, 12, 9, 4, 5, 12, 15, 12, 5, 4, 9, 12, 9, 4, 2, 4, 5, 4, 2};     //use 1se 159
    float KGaussTreated[K1size*K1size];

    for (int i = 0; i <K1size*K1size; i++)
        KGaussTreated[i] = ((float)KGauss[i]/159.);

    for (int i=0; i<K1size*K1size; i++)
      printf("K[%d] = %f ",i, KGaussTreated[i]);

    blurred_pixarr = (uint8_t*)malloc(newsize);
    status = doConvo(blurred_pixarr, grayed_pixarr, KGaussTreated, IMGwidth, IMGheight, K1size);
    char ppm_blurred[20] = "OUT2_blurred.ppm";
    status = writePPM(blurred_pixarr, newsize, ppm_blurred, 0);

    //SOBEL

    //float KxSobel[K2size*K2size] = {-3., 0., 3., -10., 0., 10., -3., 0., 3.};
    //float KySobel[K2size*K2size] = {3., 10., 3., 0., 0., 0., -3., -10., -3.};
    float KxSobel[K2size*K2size] = {-1., 0., 1., -2., 0., 2., -1., 0., 1.};
    float KySobel[K2size*K2size] = {1., 2., 1., 0., 0., 0., -1., -2., -1.};
    uint8_t* xSobel_pixarr = (uint8_t*)malloc(newsize*sizeof(uint8_t));
    uint8_t* ySobel_pixarr = (uint8_t*)malloc(newsize*sizeof(uint8_t));
    uint8_t* finalSobel_pixarr = (uint8_t*)malloc(newsize*sizeof(uint8_t));
    float* sobelDirection = (float*)malloc(newsize*sizeof(float));


    //XSOBEL
        char ppm_xSobel[20]= "OUT4_xSobel.ppm";
        status = doConvo(xSobel_pixarr, blurred_pixarr, KxSobel, IMGwidth, IMGheight, K2size);
        status = writePPM(xSobel_pixarr, newsize, ppm_xSobel, 0);

    //YSOBEL
        char ppm_ySobel[20] = "OUT5_ySobel.ppm";
        status = doConvo(ySobel_pixarr, blurred_pixarr, KySobel, IMGwidth, IMGheight, K2size);
        status = writePPM(ySobel_pixarr, newsize, ppm_ySobel, 0);

    //FINAL SOBEL
        char ppm_finalSobel[20] = "OUT5_finalSobel.ppm";
        uint32_t prod;


    for (int i=0; i < newsize; i++){
        prod = ((uint32_t)ySobel_pixarr[i]*ySobel_pixarr[i])+((uint32_t)xSobel_pixarr[i]*xSobel_pixarr[i]);

            //saturating (some very strong edges were getting turned over
            if (sqrt(prod) > 255.0)
                finalSobel_pixarr[i] = 255;
            else
                finalSobel_pixarr[i] = (uint8_t)sqrt(prod);

            //getting direction of edge
             sobelDirection[i] = atan2(xSobel_pixarr[i], ySobel_pixarr[i]);
        }

    //simpler adding of sobel results
    /*
    for (int i=0; i < newsize; i++){
        if (ySobel_pixarr[i] + xSobel_pixarr[i] > 255)
            finalSobel_pixarr[i] = 255;
        else
            finalSobel_pixarr[i] = ySobel_pixarr[i] + xSobel_pixarr[i];
    }
    */

    status = writePPM(finalSobel_pixarr, newsize, ppm_finalSobel, 0);


    //NON-MAXIMUM SUPPRESSION


    free(grayed_pixarr);
    free(blurred_pixarr);
    free(xSobel_pixarr);
    free(ySobel_pixarr);
    free(finalSobel_pixarr);
    free(sobelDirection);







}









































