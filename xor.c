#include <assert.h>
#include <math.h>
#include <raylib.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdio.h>
#define NN_IMPLEMENTATION
#define NN_GYM
#include "nn.h"

//---------------------------------------------------
//---------------- HYPER PARAMETERS -----------------
//---------------------------------------------------
size_t arch[] = {2, 2, 1};

#define FPS 60
#define WIN_WIDTH_RATIO 16
#define WIN_LENGTH_RATIO 9
#define WIN_SCALE 100
#define MAX_EPOCHS 10*1000
#define EPOCHS_PER_FRAME 3
#define RATE 1.f
size_t chosen_input = 1;
int paused = false;
//---------------------------------------------------
//---------------------------------------------------
//---------------------------------------------------

int main() {

    assert((chosen_input>=0)&&(chosen_input<arch[0]));
    
    NN nn = nn_alloc(arch, ARRAY_LEN(arch));

    srand(time(NULL));
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

    Mat ti = matrix_alloc(4 , 2);
    ti.ele = tai;
    Mat to = matrix_alloc(4 , 1);
    to.ele = tao;

    float win_width = WIN_WIDTH_RATIO*WIN_SCALE;
    float win_lenth = WIN_LENGTH_RATIO*WIN_SCALE;
    
    //SetConfigFlags(FLAG_WINDOW_RESIZABLE);
    InitWindow(win_width, win_lenth, "Test");

    SetTargetFPS(FPS);
    
    size_t epoch = 0;
    while (!WindowShouldClose()) {
        if (IsKeyPressed(KEY_SPACE)) {
            paused = !paused;
        }

        if (IsKeyPressed(KEY_R)) {
            epoch = 0;
            srand(time(NULL));
            nn_rand(nn, -1, 1);
        }
        
        for (size_t i = 0 ; i < EPOCHS_PER_FRAME && epoch<MAX_EPOCHS && paused; ++i) {
            nn_backprop(nn, ti, to, RATE);
            epoch +=1;
        }

        float box_w = win_width*0.3;
        float box_l = win_lenth*0.3;
        float xoffset = win_width*0.2;
        float yoffset = win_lenth*0.09;
        float box_xpadl = (win_width-box_w)/2 + xoffset;
        float box_ypadt = (win_lenth-box_l)/2 + yoffset;
        float line_thick = 2;

        float circ_r = win_lenth*0.04;
        float neuron_distx = box_w/(nn.n_layers);
        Vector2 neuron_center;
        Vector2 pneuron_center;
        Color neuron_color_low = {0x00, 0x00, 0x00, 0xFF};
        Color neuron_color_high = {0xFF, 0xFF, 0xFF, 0xFF};
        Color w_color_low = {0xED, 0x53, 0x53, 0xFF};
        Color w_color_high = {0x36, 0xBE, 0x7c, 0xFF};
        //Color w_color_low = {0xFF, 0x00, 0xFF, 0xFF};
        //Color w_color_high = {0x00, 0xFF, 0x00, 0xFF};
        char buf[256];

        BeginDrawing();

            DrawText("XOR Gate", win_width*0.05, win_lenth*0.07, 60, (Color){0xE1, 0x12, 0x99, 0xFF});  
            matrix_copy(NN_INPUT(nn), matrix_row(ti, chosen_input));
            nn_forward(nn);


            ClearBackground((Color){0x19, 0x19, 0x19, 0xFF});
        
            for (size_t l = 0 ; l < nn.n_layers+1; ++l) {
                size_t n = nn.as[l].rows;
                for (size_t i = 0; i < n; ++i) {
                    if (n==1) {
                        neuron_center.x = box_xpadl+l*neuron_distx;
                        neuron_center.y = box_ypadt+box_l/2;
                    }
                    else {
                        neuron_center.x = box_xpadl+l*neuron_distx;
                        neuron_center.y = box_ypadt+i*box_l/(n-1);
                    }

                    if (l>0 ) {
                        size_t np = nn.as[l-1].rows;
                        for (size_t j = 0 ; j < np; ++j) {
                            w_color_high.a = floorf(255.f*sigmoid(MAT_ELE(nn.ws[l-1], i, j)));
                            if (np==1) {
                                pneuron_center.x = box_xpadl+(l-1)*neuron_distx;
                                pneuron_center.y = box_ypadt+box_l/2;
                            }
                            else {
                                pneuron_center.x = box_xpadl+(l-1)*neuron_distx;
                                pneuron_center.y = box_ypadt+j*box_l/(np-1);
                            }

                            DrawLineEx(neuron_center, pneuron_center, box_l*0.015 , ColorAlphaBlend(w_color_low, w_color_high, WHITE));
                        }
                    } 
                }
            }

            for (size_t l = 0 ; l < nn.n_layers+1; ++l) {
                size_t n = nn.as[l].rows;
                for (size_t i = 0; i < n; ++i) {
                    neuron_color_high.a = floorf(255.f*MAT_ELE(nn.as[l], i, 0));

                    if (n==1) {
                        neuron_center.x = box_xpadl+l*neuron_distx;
                        neuron_center.y = box_ypadt+box_l/2;
                    }
                    else {
                        neuron_center.x = box_xpadl+l*neuron_distx;
                        neuron_center.y = box_ypadt+i*box_l/(n-1);
                    }
                    
                    if (l>0 ) {
                        DrawCircleV(neuron_center, circ_r, ColorAlphaBlend(neuron_color_low, neuron_color_high, WHITE));
                        DrawRing(neuron_center, circ_r, 1.07*circ_r, 0 , 360, 1 , (Color){0x90, 0x90, 0x90, 0xFF});
                        snprintf(buf, sizeof(buf), "%.2f", MAT_ELE(nn.as[l], i, 0));
                        DrawText(buf, neuron_center.x - circ_r*0.45, neuron_center.y - circ_r*0.25, 20, RED);
                    }
                    else {
                        DrawCircleV(neuron_center, circ_r, ColorAlphaBlend(neuron_color_low, neuron_color_high, WHITE));
                        DrawRing(neuron_center, circ_r, 1.07*circ_r, 0 , 360, 1 , (Color){0x90, 0x90, 0x90, 0xFF});
                        snprintf(buf, sizeof(buf), "%.2f", MAT_ELE(ti, chosen_input, i));
                        DrawText(buf, neuron_center.x - circ_r*0.45, neuron_center.y - circ_r*0.25, 20, RED);
                    }
                }
            }
            
            snprintf(buf, sizeof(buf), "Epoch = %zu/%d", epoch, MAX_EPOCHS);
            DrawText(buf, win_width*0.35, win_lenth*0.07, 50, RAYWHITE);
            snprintf(buf, sizeof(buf), "Cost = %.3f", nn_cost(nn, ti, to));
            DrawText(buf, win_width*0.1, win_lenth*0.3, 50, RAYWHITE);
            
            for (size_t i = 0 ; i < ti.rows; ++i) {
                matrix_copy(NN_INPUT(nn), matrix_row(ti, i));
                nn_forward(nn);
                snprintf(buf, sizeof(buf), "%.0f    %.0f  ->  %.3f", MAT_ELE(ti, i, 0), MAT_ELE(ti, i, 1), MAT_ELE(NN_OUTPUT(nn), 0, 0));
                DrawText(buf, win_width*0.12, win_lenth*0.5+0.1*win_lenth*i, 30 , RAYWHITE);
                
            }
            
            //DrawLineEx((Vector2) {box_xpadl, box_ypadt}, (Vector2){box_xpadl+box_w, box_ypadt}, line_thick,(Color){0x00, 0xFF, 0x00, 0xFF});
            //DrawLineEx((Vector2) {box_xpadl, box_ypadt+box_l}, (Vector2){box_xpadl+box_w, box_ypadt+box_l}, line_thick,(Color){0x00, 0xFF, 0x00, 0xFF});
            //DrawLineEx((Vector2) {box_xpadl, box_ypadt}, (Vector2){box_xpadl, box_ypadt+box_l}, line_thick,(Color){0x00, 0xFF, 0x00, 0xFF});
            //DrawLineEx((Vector2) {box_xpadl+box_w, box_ypadt}, (Vector2){box_xpadl+box_w, box_ypadt+box_l}, line_thick,(Color){0x00, 0xFF, 0x00, 0xFF});
            //DrawCircle(win_width/2 , win_lenth/2 , 4, RED); 
        EndDrawing();
    }

    CloseWindow();

	return 0;
}
