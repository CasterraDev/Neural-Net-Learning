#include "include/plot.h"
#define NN_IMPLEMENTATION
#include "include/nn.h"
#include <SDL2/SDL_rect.h>
#include <SDL2/SDL_timer.h>
#include <SDL2/SDL_video.h>
#include <SDL2/SDL_events.h>
#include <SDL2/SDL_render.h>
#include <SDL2/SDL.h>

void plotAppend(Plot* p, float item){
        if (p->count >= p->capacity) {
            p->capacity = p->capacity == 0 ? 256 : p->capacity * 2;
            p->items =
                realloc(p->items, p->capacity * sizeof(*p->items));
        }
        p->items[p->count++] = item;
}

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
