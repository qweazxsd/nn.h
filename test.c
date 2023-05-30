#include "raylib.h"

int main (int argc, char *argv[])
{
    const int Width = 800;    
    const int Length = 600;    

    InitWindow(Width, Length, "Test");

    SetTargetFPS(60);

    while (!WindowShouldClose()) {
        BeginDrawing();

            ClearBackground(RAYWHITE);
            DrawCircle(Width/2 , Length/2 , 100, RED);
        EndDrawing();

    }

    CloseWindow();
    return 0;
}
