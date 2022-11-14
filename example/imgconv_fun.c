#include "imgconv_fun.h"
#include <math.h>
#include <string.h>

int binstr2int (char *str) {
// it converts a 32-bit string to a signed integer value.
// we go from str[0] to str[31], to str[32] = '\0'. str[0]: MSB, str[31]: LSB

	int i, j, x; // -2^31 to (2^31 -1)
        unsigned int p31=1; // 0 to 2^32 -1 

	//printf("%s\n",str);
        x = 0;
        if (str[0] == '1') {
            for (j = 0; j < 31; j++) p31 = p31*2;
	    x = -p31; 
	    str++; // so that str[0] is not included in the power of 2 computation.
        }
	
        i = 0;
	while (*str != '\0' && *str != '\n' && *str != '\r') {   // string term or newline?
	   if (*str != '0' && *str != '1') // binary digit?
	      { printf("(binstr2int) Not a binary digit\n"); exit(1); }
	   i = i*2 + (*str++ & 1); // power of 2 formula
	}
	return i+x;
}

int int2binstr (unsigned char *bit_str, int numb) {
// it converts a 32-bit signed integer number represented in decimal into a bit string.
// It also gets # of minimum required bits for a signed integer number
// numb: -2^(32-1) to 2^(32-1)-1
// bit_str[0]: MSB, bit_str[31]: LSB
    int k, b, i;

   // we need unsigned representation to use our 2C formula correctly
      unsigned int p31,d; // 0 to (2^32-1)
      p31 = 1;

      if (numb < 0) {
           d = -numb; // we need to get the 2C representation of '-d' using
                     // the formula K = (2^n-1) - B +1. K,B: unsigned #s. B = d in our example. 
	   for (i = 0; i < 31; i++) p31 = p31*2; // p31 = 2^31
	   d = (p31*2 -1) + (-d) + 1; // 2C formula. 'd' bits represent (in 2C) a negative number
                                    // In the equation below, 'd' is treated as an unsigned in order to
                                    //   get the bits.
           //printf("(get_nb) d = %d\n",d);
       }
       else
	{
	   for (i=0; i < 32; i++) bit_str[i] = 0;
 	   d = numb;
	}

       k = 31;
       // 'd' here is treated as unsigned for the purposes of getting the bits.
       do {         // 32 bits max available.
	   b = d % 2; d = d / 2;
	   bit_str[k] = b; k--; // this is problematic bit_str[k] is char.
       } while (d!=0);

  return k+1; // index from where to start (k+1 up to bit 31).
              // if numb < 0, then k=-1 and k+1=0

// we have
// 0  1  2  3  4  5  6  7 .... 25 26 27 28 29 30 31
//                             b6 b5 b4 b3 b2 b1 b0
// k+1 = 25.. need to start from 25.

}


int read_binfile (CONV_args convo_data, char *in_file, int typ)
{
  // Reads binary file that contains the binary image: raster scan fashion
  // Data is placed on convo.data.I
  // type: data type of each element:
  //    type = 0: each element is unsigned 8-bit integer. ==> 'unsigned char'
  //    type = 1: each element is a signed integer (32 bits) ==> 'int'
  
  FILE *file_i;
  int i;
  size_t result, ELEM_SIZE;
  
  if (typ != 0 && typ != 1) { printf ("(read_binfile) Wrong modifier (only 0, 1 accepted)\n"); return -1; }
  
  file_i = fopen(in_file,"rb");
  if (file_i == NULL) { printf ("(read_binfile) Error opening file!\n"); return -1; } // check that the file was actually opened
      
  if (typ == 0) // each element (pixel) is an unsigned integer of 8 bits
  {
      unsigned char *IM;
      IM = (unsigned char *) calloc (convo_data.SX*convo_data.SY, sizeof(unsigned char));

      ELEM_SIZE = sizeof(unsigned char);
      result = fread (IM, sizeof(unsigned char), convo_data.SX*convo_data.SY, file_i);
            
      for (i = 0; i < convo_data.SX*convo_data.SY; i++)  convo_data.I[i] = (int) IM[i];
	  // This conversion transform elements of type 'unsigned char' to 'int'
	  // IMPORTANT: DO NOT use 'char' for input data. When transforming char to int, it will assume your data to be
	  //      signed, and it will screw up everything. Use unsigned char such that it transform from unsigned into signed integer.
      
      free (IM);
  }
  else //  if (typ == 1) // each element (pixel) is a signed integer (32 bits, 4 bytes)
  {
      int *IM;
      IM = (int *) calloc (convo_data.SX*convo_data.SY, sizeof(int));

      ELEM_SIZE = sizeof(int);
      result = fread (IM, sizeof(int), convo_data.SX*convo_data.SY, file_i);
          
      for (i = 0; i < convo_data.SX*convo_data.SY; i++)  convo_data.I[i] = IM[i];
      
      free (IM);
  }
    
  fclose (file_i);

  printf ("(read_binfile) Size of each element: %ld bytes\n", ELEM_SIZE); // Size of each element
  printf ("(read_binfile) Input binary file: # of elements read = %ld\n", result); // Total # of elements successfully read
  
  return 0;
}   
  

int conv2D (CONV_args convo_data)
{
   // it computes convolution (the centered part)
   // It performs the same job as MATLAB conv2(I,K,'same')
   int sX, sY, kX, kY;
   int *I, *K, *O; // these are provided as linear arrays
 
   I = convo_data.I; O = convo_data.O; sX = convo_data.SX; sY = convo_data.SY;
   K = convo_data.K; kKxX = convo_data.KX; kY = convo_data.KY;

   // convolution 2d, but provided as linear array.
   int i, j, m, n, mm, nn;
   int kCX, kCY; // center index of kernel
   int sum; // temp accumulation buffer
   int i_I, j_I;

   // check validity of parameters
   if (!I || !O || !K) return 0;
   if (sX <= 0 || kX <= 0) return 0;

   // find center position of kernel (half of kernel size, flooring)
   kCX = kX / 2; kCY = kY / 2;

   // O(i,j) = sum { sum { K(mm,nn) x I(i+kCY-mm, j+kCX-nn) } }. mm = kY-1-m, nn = kX-1-n
   //    it uses the portion of I of the same size as K. The portion of I is centered
   //    based on the kernel center position kCX, kCY
   /* Example: kX=kY=3. kCX = kCY = 1
      Kernel: K(m,n)			Flipped Kernel: K(mm,nn)
      K(0,0) K(0,1) K(0,2)		K(2,2) K(2,1) K(2,0)	
      K(1,0) K(1,1) K(1,2)		K(1,2) K(1,1) K(1,0)
      K(2,0) K(2,1) K(2,2)		K(0,2) K(0,1) K(0,0)
				
					Input image region that multiplies the flipped kernel
      					I(i-1,j-1) I(i-1,j) I(i-1,j+1)
      					I(  i,j-1) I(  i,j) I(  i,j+1)
     			 		I(i+1,j-1) I(i+1,j) I(i+1,j+1)
    */
   
   // Convolution:
   // The computation with i=0:sY-1, j=0:sX-1 calculates the centered part of the convolution.
   //    We can cover the entire convolution: sX+kX-1, sY+kY-1
   //      We'd need to zero-pad I so that it is of size sX+kX-1,sY+kY-1
   //        --> 'I' would be placed in the 'center': ( floor(kY/2) to sY-1 + floor(kY/2), floor(kX/2) to sX-1 + floor(kX/2) )
   //        * Another option is to use (i,j) but with indices:
   //                i = -floor(kY/2) to sY-1 + ceil(kY/2)-1
   //                j = -floor(kX/2) to sX-1 + ceil(kX/2)-1
   for (i = 0; i < sY; i++) // Output matrix: rows
   {
	for (j = 0; j < sX; j++) // Output matrix: columns
	{
		sum = 0; // init to 0 before sum
		for (m = 0; m < kY; m++) // Kernel: rows
		{
			mm = kY - 1 - m; // Flipped kernel: index for rows
			for (n = 0; n < kX; n++) // Kernel: columns
			{
				nn = kX - 1 - n; // Flipped kernel: index for columns
				i_I = i + (kCY - mm);
				j_I = j + (kCX - nn);

				// out-of-bounds input samples are set to 0
				if (i_I >= 0 && i_I < sY && j_I >= 0 && j_I < sX)
					sum += I[sX * i_I + j_I] * K[kX * mm + nn];
			}
		}
		//out[dataSizeX * i + j] = (unsigned char)((float)fabs(sum) + 0.5f);
		O[sX*i + j] = sum; // fabs work for float?
	}
    }

    return 1;
}



