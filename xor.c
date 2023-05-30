#define NN_IMPLEMENTATION
#include "nn.h"
#include "raylib.h"
//---------------------------------------------------
//---------------- HYPER PARAMETERS -----------------
//---------------------------------------------------
#define WIN_WIDTH_RATIO 16
#define WIN_LENGTH_RATIO 9
#define MAX_EPOCHS 10*1000
//---------------------------------------------------
//---------------------------------------------------
//---------------------------------------------------

int main() {

    size_t arch[] = {2, 2, 1};
    NN nn = nn_alloc(arch, ARRAY_LEN(arch));
    nn_rand(nn, -1, 1);

    float tai[] = {
        0, 0,
        0, 1,
        1, 0,
        1, 1,
    };
    float tao[] = {
        0, 
        1, 
        1, 
        0, 
    };

    Matrix ti = matrix_alloc(4 , 2);
    ti.ele = tai;
    Matrix to = matrix_alloc(4 , 1);
    to.ele = tao;
    

    for (size_t i = 0; i < MAX_EPOCHS; ++i) {
        nn_backprop(nn, ti, to, 1.f);
        printf("C = %f\n", nn_cost(nn, ti, to));
    }
    
    for (size_t i = 0; i < ti.rows; ++i) {
        matrix_copy(NN_INPUT(nn), matrix_row(ti, i));
        nn_forward(nn);
        printf("%f %f -> %f\n", MAT_ELE(ti, i, 0), MAT_ELE(ti, i, 1), MAT_ELE(NN_OUTPUT(nn), 0, 0));
    }

	return 0;
}
