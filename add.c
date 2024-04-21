#include <SDL2/SDL_rect.h>
#include <SDL2/SDL_timer.h>
#include <SDL2/SDL_video.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#define NN_IMPLEMENTATION
#include "nn.h"

#define BITS 4

#include <SDL2/SDL_events.h>
#include <SDL2/SDL_render.h>
#include <stdio.h>

#include <SDL2/SDL.h>

#define WINDOW_FACTOR 80
#define WINDOW_HEIGHT 9*WINDOW_FACTOR
#define WINDOW_WIDTH 16*WINDOW_FACTOR

typedef struct {
    float* items;
    size_t count;
    size_t capacity;
} Plot;

#define plotAppend(p, item)                                                    \
    do {                                                                       \
        if ((p)->count >= (p)->capacity) {                                     \
            (p)->capacity = (p)->capacity == 0 ? 256 : (p)->capacity * 2;      \
            (p)->items =                                                       \
                realloc((p)->items, (p)->capacity * sizeof(*(p)->items));      \
        }                                                                      \
        (p)->items[(p)->count++] = (item);                                     \
    } while (0)

void plotCost(SDL_Renderer* renderer, Plot plot, int rx, int ry, int rw,
              int rh) {
    SDL_Rect rect;
    rect.h = rh;
    rect.w = rw;
    rect.x = rx;
    rect.y = ry;
    SDL_RenderDrawRect(renderer, &rect);
    float min, max;
    min = FLT_MAX;
    max = FLT_MIN;
    for (size_t i = 0; i < plot.count; ++i) {
        if (max < plot.items[i])
            max = plot.items[i];
        if (min > plot.items[i])
            min = plot.items[i];
    }
    if (min > 0) min = 0;
    size_t n = plot.count;
    if (n < 1000)
        n = 1000;
    SDL_FPoint points[plot.count];
    for (size_t i = 0; i + 1 < plot.count; ++i) {
        float x1 = rx + (float)rw / n * i;
        float y1 = ry + (1 - (plot.items[i] - min) / (max - min)) * rh;
        float x2 = rx + (float)rw / n * (i + 1);
        float y2 = ry + (1 - (plot.items[i + 1] - min) / (max - min)) * rh;
        SDL_FPoint t;
        t.x = x1;
        t.y = y1;
        SDL_FPoint t2;
        t2.x = x2;
        t2.y = y2;
        points[i] = t;
        points[i + 1] = t2;
    }
    SDL_RenderDrawLinesF(renderer, points, plot.count);
}

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
    // Round up the count of batches
    size_t batchCnt = (td.rows + batchSize - 1) / batchSize;

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
    size_t runs = 0;
    float cost = 0;
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
        rh = WINDOW_HEIGHT*2 / 3;
        rx = padding;
        ry = (WINDOW_HEIGHT / 2 - rh / 2);

        SDL_SetRenderDrawColor(renderer, 18, 18, 18, 255);
        SDL_RenderClear(renderer);

        SDL_SetRenderDrawColor(renderer, 255, 0, 0, 255);
        for (size_t i = 0; i < batchCnt && runs < runsAmt; ++i) {
            size_t size = batchSize;
            size_t batchStart = i * batchSize;
            // If on the last batch. The size most likely won't be a full
            // batch, so we fix the size and say it's the last batch of the
            // trainingData so we can calculate the avg cost.
            if (td.rows - batchStart < batchSize) {
                size = td.rows - batchStart;
            }

            // Split the training input/output
            Mat ti = {.rows = size,
                      .cols = BITS * 2,
                      .stride = td.stride,
                      .es = &MAT_AT(td, batchStart, 0)};
            Mat to = {.rows = size,
                      .cols = 1 + BITS,
                      .stride = td.stride,
                      .es = &MAT_AT(td, batchStart, ti.cols)};

            // Train this batch
            nnBackprop(nn, g, ti, to);
            nnLearn(nn, g, rate);
            cost += nnCost(nn, ti, to);

            if (i == batchCnt - 1) {
                printf("%zu: Average Cost = %f\n", i, cost / batchSize);
                plotAppend(&plot, cost / batchSize);
                cost = 0;
            }
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
