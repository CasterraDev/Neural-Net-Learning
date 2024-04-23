#include <SDL2/SDL_blendmode.h>
#include <SDL2/SDL_pixels.h>
#include <SDL2/SDL_rect.h>
#include <SDL2/SDL_timer.h>
#include <SDL2/SDL_video.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#define NN_IMPLEMENTATION
#include "include/nn.h"
#include "include/plot.h"
#include "include/nnFile.h"

#include <SDL2/SDL_events.h>
#include <SDL2/SDL_render.h>
#include <stdio.h>

#include <SDL2/SDL.h>

#define WINDOW_FACTOR 80
#define WINDOW_HEIGHT 9 * WINDOW_FACTOR
#define WINDOW_WIDTH 16 * WINDOW_FACTOR

char *argsShift(int *argc, char*** argv){
    assert(*argc > 0);
    char *res = **argv;
    *argc -= 1;
    *argv += 1;
    return res;
}

int main(int argc, char** argv) {
    //srand(time(0));
    srand(40);

    char* imgPath = argv[1];
    int w,h,c;
    Mat imgMat = imgToMat(imgPath, &w, &h, &c);

    char running = 1;
    if (SDL_Init(SDL_INIT_VIDEO) != 0) {
        printf("Error: %s", SDL_GetError());
    }
    SDL_Window* window = SDL_CreateWindow("Test Window", 0, 0, 16 * 80, 9 * 80,
                                          SDL_WINDOW_SHOWN);
    SDL_Renderer* renderer = SDL_CreateRenderer(
        window, -1, SDL_RENDERER_ACCELERATED | SDL_RENDERER_PRESENTVSYNC);
    SDL_Event event;
    SDL_SetRenderDrawBlendMode(renderer, SDL_BLENDMODE_BLEND);

    // REMEMBER: Batchsize is a tweaky variable. Should try to make it so
    // trainingData.sampleSize / batchSize equals a whole number and make sure
    // it's not too big.
    size_t batchSize = 30;

    // REMEMBER: The nn architecture is a tweaky variable. The most important
    // one.
    size_t arch[] = {2, 4, 1};

    // Allocate the neural net and the gradient net.
    NN nn = nnAlloc(arch, ARRAY_LEN(arch));
    NN g = nnAlloc(arch, ARRAY_LEN(arch));
    // Randomize the weights/bias of the neural net.
    nnRand(nn, 0, 1);

    float rate = 1;
    // REMEMBER: runsAmt is a thing to tweak if nn isn't working completely
    size_t runsAmt = 2000;
    Plot plot = {0};


    SDL_Texture* texture = SDL_CreateTexture(renderer, SDL_PIXELFORMAT_RGBA8888, SDL_TEXTUREACCESS_STREAMING, w, h);
    uint8_t* pixels;
    int pitch;

    SDL_LockTexture(texture, NULL, (void**)&pixels, &pitch);

    printf("rows: %zu pitch: %d\n", imgMat.rows, pitch);
    for (size_t c = 0; c < imgMat.rows; c++){
        int pixelColor = MAT_AT(imgMat, c, 2) * 255;
        int x = MAT_AT(imgMat, c, 0) * (w-1);
        int y = MAT_AT(imgMat, c, 1) * (h-1);
        //printf("%d , %d = %d\n", x, y, pixelColor);
        pixels[(x*4)+(y*pitch)] = pixelColor;
        pixels[(x*4)+(y*pitch) + 1] = pixelColor;
        pixels[(x*4)+(y*pitch) + 2] = pixelColor;
        pixels[(x*4)+(y*pitch) + 3] = pixelColor;
    }
    SDL_Rect rect2;
    rect2.x = 100;
    rect2.y = 100;
    rect2.w = w;
    rect2.h = h;
    SDL_RenderClear(renderer);
    SDL_RenderCopyEx(renderer, texture, NULL, &rect2, 0, NULL, SDL_FLIP_NONE);
    SDL_UnlockTexture(texture);

    size_t scale = 200;
    while (running) {
        while (SDL_PollEvent(&event)) {
            if (event.type == SDL_QUIT) {
                running = 0;
            }
        }

        int rw, rh, rx, ry;
        int padding = 25;
        rw = (WINDOW_WIDTH / 2) - padding;
        rh = WINDOW_HEIGHT * 2 / 3;
        rx = padding;
        ry = (WINDOW_HEIGHT / 2 - rh / 2);

        SDL_SetRenderDrawColor(renderer, 18, 255, 18, 255);
        SDL_RenderClear(renderer);

        SDL_Rect rect;
        rect.x = WINDOW_WIDTH - (w+scale) - 50;
        rect.y = WINDOW_HEIGHT - (h+scale) - 50;
        rect.w = w + scale;
        rect.h = h + scale;

        SDL_RenderCopyEx(renderer, texture, NULL, &rect, 0, NULL, SDL_FLIP_NONE);

        //plotCost(renderer, plot, rx, ry, rw, rh);
        SDL_RenderPresent(renderer);
    }

    NN_PRINT(nn);


    SDL_DestroyTexture(texture);
    SDL_DestroyRenderer(renderer);
    SDL_DestroyWindow(window);
    SDL_Quit();

    return 0;
}
