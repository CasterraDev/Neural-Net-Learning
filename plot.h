#pragma once
#include "nn.h"
#include <SDL2/SDL_render.h>

typedef struct {
    float* items;
    size_t count;
    size_t capacity;
} Plot;

void plotAppend(Plot* p, float item);
void plotCost(SDL_Renderer* renderer, Plot plot, int rx, int ry, int rw,
              int rh);
