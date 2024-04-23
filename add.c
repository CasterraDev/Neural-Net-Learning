#include <SDL2/SDL_rect.h>
#include <SDL2/SDL_timer.h>
#include <SDL2/SDL_video.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#define NN_IMPLEMENTATION
#include "include/nn.h"
#include "include/plot.h"
#include "include/trainer.h"

#define BITS 4

#include <SDL2/SDL_events.h>
#include <SDL2/SDL_render.h>
#include <stdio.h>

#include <SDL2/SDL.h>

#define WINDOW_FACTOR 80
#define WINDOW_HEIGHT 9 * WINDOW_FACTOR
#define WINDOW_WIDTH 16 * WINDOW_FACTOR

int main(void) {
    srand(time(0));
    // srand(40);

    char running = 1;
    if (SDL_Init(SDL_INIT_VIDEO) != 0) {
        printf("Error: %s", SDL_GetError());
    }
    SDL_Window* window = SDL_CreateWindow("Test Window", 0, 0, 16 * 80, 9 * 80,
                                          SDL_WINDOW_SHOWN);
    SDL_Renderer* renderer = SDL_CreateRenderer(
        window, -1, SDL_RENDERER_ACCELERATED | SDL_RENDERER_PRESENTVSYNC);
    SDL_Event event;

    // Generate the training data
    size_t n = (1 << BITS);
    size_t rows = n * n;
    Mat td = matAlloc(rows, (2 * BITS) + 1 + BITS);
    for (size_t i = 0; i < rows; ++i) {
        size_t x = i / n;
        size_t y = i % n;
        size_t z = x + y;
        for (size_t j = 0; j < BITS; ++j) {
            MAT_AT(td, i, j) = (x >> j) & 1;
            MAT_AT(td, i, j + BITS) = (y >> j) & 1;
            MAT_AT(td, i, j + (BITS * 2)) = (z >> j) & 1;
        }
        MAT_AT(td, i, BITS + (BITS * 2)) = (z >= n);
    }
    // Shuffle the training data for batch training
    matShuffleRows(td);

    // REMEMBER: Batchsize is a tweaky variable. Should try to make it so
    // trainingData.sampleSize / batchSize equals a whole number and make sure
    // it's not too big.
    size_t batchSize = 30;

    // REMEMBER: The nn architecture is a tweaky variable. The most important
    // one.
    size_t arch[] = {2 * BITS, 4 * BITS, BITS + 1};

    // Allocate the neural net and the gradient net.
    NN nn = nnAlloc(arch, ARRAY_LEN(arch));
    NN g = nnAlloc(arch, ARRAY_LEN(arch));
    // Randomize the weights/bias of the neural net.
    nnRand(nn, 0, 1);

    float rate = 1;
    // REMEMBER: runsAmt is a thing to tweak if nn isn't working completely
    size_t runsAmt = 2000;
    Plot plot = {0};

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

        SDL_SetRenderDrawColor(renderer, 18, 18, 18, 255);
        SDL_RenderClear(renderer);

        SDL_SetRenderDrawColor(renderer, 255, 0, 0, 255);

        for (size_t i = 0; i < runsAmt; i++){
            tnrBatchTrain(&nn, &g, &td, &plot, batchSize, BITS * 2, BITS + 1, rate);
        }

        plotCost(renderer, plot, rx, ry, rw, rh);
        SDL_RenderPresent(renderer);
    }

    NN_PRINT(nn);

    size_t fails = 0;
    size_t overflows = 0;

    for (size_t x = 0; x < n; ++x) {
        for (size_t y = 0; y < n; ++y) {
            size_t z = x + y;
            for (size_t j = 0; j < BITS; ++j) {
                MAT_AT(NN_INPUT(nn), 0, j) = (x >> j) & 1;
                MAT_AT(NN_INPUT(nn), 0, j + BITS) = (y >> j) & 1;
            }
            nnForward(nn);

            if (MAT_AT(NN_OUTPUT(nn), 0, BITS) > 0.5f) {
                if (z < n) {
                    printf("Overflowed: %zu + %zu = %zu\n", x, y, z);
                    overflows++;
                }
            } else {
                size_t ans = 0;
                for (size_t j = 0; j < BITS; ++j) {
                    size_t bit = MAT_AT(NN_OUTPUT(nn), 0, j) > 0.5f;
                    ans |= bit << j;
                }

                if (z != ans) {
                    printf("%zu + %zu = %zu(%zu)\n", x, y, ans, z);
                    fails++;
                }
            }
        }
    }

    printf("--------------------------\n");
    printf("Fails: %zu\n", fails);
    printf("Overflows: %zu\n", overflows);

    SDL_DestroyWindow(window);
    SDL_Quit();

    return 0;
}
