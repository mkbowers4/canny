#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>
#include <string.h>
#include <math.h>

#define K1size 5
#define K2size 3

// //generating parallel vectored (tbb+see) brot
//     FILE *fp_vp = fopen("vpbrot.ppm", "wb"); // b - binary mode
//     (void) fprintf(fp_vp, "P6 %d %d 255\n", MB.width, MB.height);
//                 //fprintf(stream, parameters
//     char cfactor_vp = 200; // This is a multiplier to change saturation
//     char color_vp[3];
//     for(i=0; i < SIZE; i++){
//         color_vp[2] = ((MB.pixarr_vp[i] & 0xff) >> 3 )*cfactor_vp;
//         color_vp[1] = ((MB.pixarr_vp[i] & 0xff) >> 6 )*cfactor_vp;
//         color_vp[0] = ((MB.pixarr_vp[i] & 0xff) >> 0 )*cfactor_vp;
//         (void) fwrite(color_vp, 1, 3, fp_vp);
//
//     }
//     (void) fclose(fp_vp);

int IMGwidth, IMGheight;


// typedef struct {
//     int SX;
//     int SY;
//     int* K;
// } CONVO_args;


int writePPM (uint8_t* arr, int size, char* FILENAME, int mode){

    //mode = 0 for color data, mode = 1 for grayscaled data

    int count;
    FILE *fp_wr = fopen(FILENAME, "wb"); // b - binary mode
    fprintf(fp_wr, "P6 %d %d 255\n", IMGwidth, IMGheight);


    if (mode == 1){
        count = fwrite(arr, 1, (size_t)(IMGwidth*IMGheight)*3, fp_wr);
    }
    else{           //we have grayscale data

        //uint8_t cfactor = 200; // This is a multiplier to change saturation
        uint8_t color[3];

        for (int i=0; i < IMGwidth*IMGheight; i++){
//             color[2] = ((arr[i] & 0xff) >> 0 )*cfactor;
//             color[1] = ((arr[i] & 0xff) >> 0 )*cfactor;
//             color[0] = ((arr[i] & 0xff) >> 0 )*cfactor;
            //color[3] = arr[i];
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

//doConvo(newpixarr, width, height, KGauss);
int doConvo(uint8_t* output, uint8_t *IMGarr, float* K, int IMGwidth, int IMGheight, int Ksize){

  float sum;
  int sumtemp;
  int i, j, m, n, mm, nn, i_I, j_I;
  //i,j for data, m,n for kernel, mm,nn for flipped kernel
  int kCX = Ksize/2;
  int kCY = Ksize/2;

  for (i=0; i < IMGheight; i++)
  {
      for (j = 0; j < IMGwidth; j++)
      {
          sum = 0.0;
          for (m = 0; m < Ksize; m++)
          {
              mm = Ksize - 1 - m;
              for (n = 0; n < Ksize; n++)
              {
                  nn = Ksize - 1 - n;
                  i_I = i + (kCY - mm);
                  j_I = j + (kCX - nn);

                  //if out of bounds, inputs are set to 0
                  if (i_I >= 0 && i_I < IMGheight && j_I >= 0 && j_I < IMGwidth)
					sum += ((float)IMGarr[IMGwidth * i_I + j_I]) * K[Ksize * mm + nn];

              }
          }
         if(sum >= 0.0)
            {sum = sum + 0.5;}    //rounding
         else
            {sum = sum - 0.5;}            //rounding  if(sum >= 0.0)

         sumtemp = (int)sum;

          output[IMGwidth*i + j] = (uint8_t)(sumtemp);
      }
  }
   return 1;
}


//doConvo(newpixarr, width, height, KGauss);
int doConvoEdge(uint16_t* output, uint8_t *IMGarr, float* K, int IMGwidth, int IMGheight, int Ksize){

  int sum;
  int sumtemp;
  int i, j, m, n, mm, nn, i_I, j_I;
  //i,j for data, m,n for kernel, mm,nn for flipped kernel
  int kCX = Ksize/2;
  int kCY = Ksize/2;

  for (i=0; i < IMGheight; i++)
  {
      for (j = 0; j < IMGwidth; j++)
      {
          sum = 0;
          for (m = 0; m < Ksize; m++)
          {
              mm = Ksize - 1 - m;
              for (n = 0; n < Ksize; n++)
              {
                  nn = Ksize - 1 - n;
                  i_I = i + (kCY - mm);
                  j_I = j + (kCX - nn);

                  //if out of bounds, inputs are set to 0
                  if (i_I >= 0 && i_I < IMGheight && j_I >= 0 && j_I < IMGwidth)
					sum += (int)((IMGarr[IMGwidth * i_I + j_I]) * K[Ksize * mm + nn]);

              }
          }
          if (sum < 0) sum = 0;

          output[IMGwidth*i + j] = (uint16_t)(sum);
      }
  }
   return 1;
}

//doConvo(newpixarr, width, height, KGauss);
int doConvoEdge2(uint8_t* output, uint8_t *IMGarr, float* K, int IMGwidth, int IMGheight, int Ksize){

  int sum;
  int sumtemp;
  int mintrack=0;
  int i, j, m, n, mm, nn, i_I, j_I;
  //i,j for data, m,n for kernel, mm,nn for flipped kernel
  int kCX = Ksize/2;
  int kCY = Ksize/2;

  for (i=0; i < IMGheight; i++)
  {
      for (j = 0; j < IMGwidth; j++)
      {
          sum = 0;
          for (m = 0; m < Ksize; m++)
          {
              mm = Ksize - 1 - m;
              for (n = 0; n < Ksize; n++)
              {
                  nn = Ksize - 1 - n;
                  i_I = i + (kCY - mm);
                  j_I = j + (kCX - nn);

                  //if out of bounds, inputs are set to 0
                  if (i_I >= 0 && i_I < IMGheight && j_I >= 0 && j_I < IMGwidth)
					sum += (int)((IMGarr[IMGwidth * i_I + j_I]) * K[Ksize * mm + nn]);

              }
          }
          if (sum < 0)
              sum = 0;
          else if (sum > 255)
              sum = 255;

          output[IMGwidth*i + j] = (uint8_t)(sum);
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
    uint8_t* xSobel_pixarr = (uint8_t*)malloc(newsize*sizeof(uint8_t));;
    uint8_t* ySobel_pixarr = (uint8_t*)malloc(newsize*sizeof(uint8_t));;
    uint8_t* finalSobel_pixarr = (uint8_t*)malloc(newsize*sizeof(uint8_t));
    //uint8_t* xSobel_pixarr2 = (uint8_t*)malloc(newsize*sizeof(uint8_t));
    //uint16_t* xSobel_pixarr16 = (uint16_t*)malloc(newsize*sizeof(uint16_t));




/*
    char ppm_xSobel[20] = "OUT3_xSobel.ppm";
    status = doConvo(xSobel_pixarr2, blurred_pixarr, KxSobel, IMGwidth, IMGheight, K2size);
    status = writePPM(xSobel_pixarr2, newsize, ppm_xSobel, 0);*/

//     char pgm_xSobel16[20]= "OUT3_xSobel.pgm";
//     status = doConvoEdge(xSobel_pixarr16, blurred_pixarr, KxSobel, IMGwidth, IMGheight, K2size);
//     status = writePGM(xSobel_pixarr16, newsize, pgm_xSobel16, 0);

    char ppm_xSobel2[20]= "OUT4_xSobel.ppm";
    status = doConvoEdge2(xSobel_pixarr, blurred_pixarr, KxSobel, IMGwidth, IMGheight, K2size);
    status = writePPM(xSobel_pixarr, newsize, ppm_xSobel2, 0);

    char ppm_ySobel[20] = "OUT5_ySobel.ppm";
    status = doConvoEdge2(ySobel_pixarr, blurred_pixarr, KySobel, IMGwidth, IMGheight, K2size);
    status = writePPM(ySobel_pixarr, newsize, ppm_ySobel, 0);

    char ppm_finalSobel[20] = "OUT5_finalSobel.ppm";
    uint32_t prod;
    // final sobel = sqrt( x^2 + y^2 );
    for (int i=0; i < newsize; i++){
        prod = ((uint32_t)ySobel_pixarr[i]*ySobel_pixarr[i])+((uint32_t)xSobel_pixarr[i]*xSobel_pixarr[i]);
        finalSobel_pixarr[i] = (uint8_t)sqrt((double)prod);

    }
    status = writePPM(finalSobel_pixarr, newsize, ppm_finalSobel, 0);

    free(grayed_pixarr);
    free(xSobel_pixarr);
    //free(xSobel_pixarr2);
    //free(xSobel_pixarr16);
    free(ySobel_pixarr);
    free(finalSobel_pixarr);

    free(blurred_pixarr);






    //image processing






}




















































// /*
//
//
// ***************************************************************
//  *
//  * ppm.c
//  *
//  * Read and write PPM files.  Only works for "raw" format.
//  *
//  * AF970205
//  *
//  ****************************************************************/
//
//
// #include <stdlib.h>
// #include <stdio.h>
// #include <ctype.h>
// #include "ppm.h"
//
// /************************ private functions ****************************/
//
// /* die gracelessly */
//
// static void
// die(char *message)
// {
//   fprintf(stderr, "ppm: %s\n", message);
//   exit(1);
// }
//
//
// /* check a dimension (width or height) from the image file for reasonability */
//
// static void
// checkDimension(int dim)
// {
//   if (dim < 1 || dim > 4000)
//     die("file contained unreasonable width or height");
// }
//
//
// /* read a header: verify format and get width and height */
//
// static void
// readPPMHeader(FILE *fp, int *width, int *height)
// {
//   char ch;
//   int  maxval;
//
//   if (fscanf(fp, "P%c\n", &ch) != 1 || ch != '6')
//     die("file is not in ppm raw format; cannot read");
//
//   /* skip comments */
//   ch = getc(fp);
//   while (ch == '#')
//     {
//       do {co
// 	ch = getc(fp);
//       } while (ch != '\n');	/* read to the end of the line */
//       ch = getc(fp);            /* thanks, Elliot */
//     }
//
//   if (!isdigit(ch)) die("cannot read header information from ppm file");
//
//   ungetc(ch, fp);		/* put that digit back */
//
//   /* read the width, height, and maximum value for a pixel */
//   fscanf(fp, "%d%d%d\n", width, height, &maxval);
//
//   if (maxval != 255) die("image is not true-color (24 bit); read failed");
//
//   checkDimension(*width);
//   checkDimension(*height);
// }
//
// /************************ exported functions ****************************/
//
// Image *
// ImageCreate(int width, int height)
// {
//   Image *image = (Image *) malloc(sizeof(Image));
//
//   if (!image) die("cannot allocate memory for new image");
//
//   image->width  = width;
//   image->height = height;
//   image->data   = (u_char *) malloc(width * height * 3);
//
//   if (!image->data) die("cannot allocate memory for new image");
//
//   return image;
// }


// Image *
// ImageRead(char *filename)
// {
//   int width, height, num, size;
//   u_char *p;
//
//   Image *image = (Image *) malloc(sizeof(Image));
//   FILE  *fp    = fopen(filename, "r");
//
//   if (!image) die("cannot allocate memory for new image");
//   if (!fp)    die("cannot open file for reading");
//
//   readPPMHeader(fp, &width, &height);
//
//   size          = width * height * 3;
//   image->data   = (u_char *) malloc(size);
//   image->width  = width;
//   image->height = height;
//
//   if (!image->data) die("cannot allocate memory for new image");
//
//   num = fread((void *) image->data, 1, (size_t) size, fp);
//
//   if (num != size) die("cannot read image data from file");
//
//   fclose(fp);
//
//   return image;
// }
//
//
// void ImageWrite(Image *image, char *filename)
// {
//   int num;
//   int size = image->width * image->height * 3;
//
//   FILE *fp = fopen(filename, "w");
//
//   if (!fp) die("cannot open file for writing");
//
//   fprintf(fp, "P6\n%d %d\n%d\n", image->width, image->height, 255);
//
//   num = fwrite((void *) image->data, 1, (size_t) size, fp);
//
//   if (num != size) die("cannot write image data to file");
//
//   fclose(fp);
// }
//
//
// int
// ImageWidth(Image *image)
// {
//   return image->width;
// }
//
//
// int
// ImageHeight(Image *image)
// {
//   return image->height;
// }

/*
void
ImageClear(Image *image, u_char red, u_char green, u_char blue)
{
  int i;
  int pix = image->width * image->height;

  u_char *data = image->data;

  for (i = 0; i < pix; i++)
    {
      *data++ = red;
      *data++ = green;
      *data++ = blue;
    }
}

void
ImageSetPixel(Image *image, int x, int y, int chan, u_char val)
{
  int offset = (y * image->width + x) * 3 + chan;

  image->data[offset] = val;
}


u_char
ImageGetPixel(Image *image, int x, int y, int chan)
{
  int offset = (y * image->width + x) * 3 + chan;

  return image->data[offset];
}*/

