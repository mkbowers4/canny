#include <stdio.h>
#include <stdlib.h>

typedef struct {
    int SX;
    int SY;
    int KX;
    int KY;
    int *I;
    int *K;
    int *O;
} CONV_args;

int binstr2int (char *str);
int int2binstr (unsigned char *bit_str, int dig);
int conv2D (CONV_args convo_data);
int read_binfile (CONV_args convo_data, char *in_file, int typ);
