#include "data.c"
#include <stdio.h>
int main (int argc, char *argv[])
{
    
    for (size_t i = 0; i < rows; i++) {
        for (size_t j = 0; j < cols; j++) {
            printf("%f ", data[i*cols+j]);
        }
        printf("\n");
    }
    return 0;
}
