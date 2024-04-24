#include "include/trainer.h"
#include <SDL2/SDL_blendmode.h>
#include <SDL2/SDL_pixels.h>
#include <SDL2/SDL_rect.h>
#include <SDL2/SDL_timer.h>
#include <SDL2/SDL_ttf.h>
#include <SDL2/SDL_video.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#define NN_IMPLEMENTATION
#include "include/nn.h"
#include "include/nnFile.h"
#include "include/plot.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "include/stb_image_write.h"

#include <SDL2/SDL_events.h>
#include <SDL2/SDL_render.h>
#include <stdio.h>

#include <SDL2/SDL.h>

#define WINDOW_FACTOR 80
#define WINDOW_HEIGHT 9 * WINDOW_FACTOR
#define WINDOW_WIDTH 16 * WINDOW_FACTOR

char* argsShift(int* argc, char*** argv) {
    assert(*argc > 0);
    char* res = **argv;
    *argc -= 1;
    *argv += 1;
    return res;
}

int main(int argc, char* argv[]) {
    const char* programName = argsShift(&argc, &argv);

    if (argc <= 0) {
        printf("USAGE: ./%s [pathToImage]", programName);
        printf("ERROR: No image given\n");
    }

    // srand(time(0));
    srand(40);

    char* imgPath = argsShift(&argc, &argv);
    int w, h, c;
    Mat imgMat = imgToMat(imgPath, &w, &h, &c);

    char running = 1;
    if (SDL_Init(SDL_INIT_VIDEO) != 0) {
        fprintf(stderr, "Error: %s", SDL_GetError());
    }
    TTF_Init();
    SDL_Window* window = SDL_CreateWindow("Image Trainer", 0, 0, 16 * 80,
                                          9 * 80, SDL_WINDOW_SHOWN);
    SDL_Renderer* renderer = SDL_CreateRenderer(
        window, -1, SDL_RENDERER_ACCELERATED | SDL_RENDERER_PRESENTVSYNC);
    SDL_Event event;
    SDL_SetRenderDrawBlendMode(renderer, SDL_BLENDMODE_BLEND);

    // REMEMBER: The nn architecture is a tweaky variable. The most important
    // one.
    size_t arch[] = {2, 28, 14, 1};

    // Allocate the neural net and the gradient net.
    NN nn = nnAlloc(arch, ARRAY_LEN(arch));
    NN g = nnAlloc(arch, ARRAY_LEN(arch));
    // Randomize the weights/bias of the neural net.
    nnRand(nn, -1, 1);

    // REMEMBER: runsAmt is a thing to tweak if nn isn't working completely
    size_t runsAmt = 500;
    size_t runs = 0;
    Plot plot = {0};
    size_t scale = 200;

    SDL_Rect imgExampleRect;
    imgExampleRect.x = WINDOW_WIDTH - (w + scale) - 50;
    imgExampleRect.y = WINDOW_HEIGHT - (h + scale) - 50;
    imgExampleRect.w = w + scale;
    imgExampleRect.h = h + scale;

    SDL_Rect liveImgRect;
    liveImgRect.x = WINDOW_WIDTH - (w + scale) - 50;
    liveImgRect.y = (h + scale) - 50;
    liveImgRect.w = w + scale;
    liveImgRect.h = h + scale;

    // Create example image
    SDL_Texture* texture = SDL_CreateTexture(renderer, SDL_PIXELFORMAT_RGBA8888,
                                             SDL_TEXTUREACCESS_STREAMING, w, h);
    uint8_t* pixels;
    int pitch;

    SDL_LockTexture(texture, NULL, (void**)&pixels, &pitch);

    for (size_t c = 0; c < imgMat.rows; c++) {
        int pixelColor = MAT_AT(imgMat, c, 2) * 255;
        int x = MAT_AT(imgMat, c, 0) * (w - 1);
        int y = MAT_AT(imgMat, c, 1) * (h - 1);
        pixels[(x * 4) + (y * pitch)] = pixelColor;
        pixels[(x * 4) + (y * pitch) + 1] = pixelColor;
        pixels[(x * 4) + (y * pitch) + 2] = pixelColor;
        pixels[(x * 4) + (y * pitch) + 3] = pixelColor;
    }

    SDL_RenderClear(renderer);
    SDL_RenderCopy(renderer, texture, NULL, &imgExampleRect);
    SDL_UnlockTexture(texture);

    Mat aiImg = matAlloc(w * h, 3);
    matFill(aiImg, 0);

    // Shuffle the training data now. For batch training
    matShuffleRows(imgMat);

    int plotRw, plotRh, plotRx, plotRy;
    int padding = 25;
    plotRw = (WINDOW_WIDTH / 2) - padding;
    plotRh = WINDOW_HEIGHT * 2 / 3;
    plotRx = padding;
    plotRy = (WINDOW_HEIGHT / 2 - plotRh / 2);

    SDL_Texture* costTexture;
    TTF_Font* font = TTF_OpenFont("fonts/HackNerdFont-Regular.ttf", 72);
    SDL_Rect costDestRect;

    SDL_Texture* liveTexture = SDL_CreateTexture(
        renderer, SDL_PIXELFORMAT_RGBA8888, SDL_TEXTUREACCESS_STREAMING, w, h);
    uint8_t* livePixels;
    int livePitch;

    while (running) {
        runs++;
        while (SDL_PollEvent(&event)) {
            if (event.type == SDL_QUIT) {
                running = 0;
            }
        }

        if (runs < runsAmt) {
            SDL_SetRenderDrawColor(renderer, 18, 18, 18, 255);
            SDL_RenderClear(renderer);

            SDL_RenderCopy(renderer, texture, NULL, &imgExampleRect);

            for (size_t i = 0; i < 100; i++) {
                tnrBatchTrain(&nn, &g, &imgMat, &plot, 28, 2, 1, 1);
            }
            matShuffleRows(imgMat);

            SDL_SetRenderDrawColor(renderer, 255, 0, 0, 255);

            plotCost(renderer, plot, plotRx, plotRy, plotRw, plotRh);

            // TODO: Look into TTF_Glyplhs and replace this with them
            char costBuf[10];
            if (!snprintf(costBuf, sizeof(costBuf), "%f",
                          plot.items[plot.count - 1])) {
                fprintf(stderr, "Error: Putting costBuf into string.");
            }
            SDL_Surface* costTextSurf = TTF_RenderText_Solid(
                font, costBuf, (SDL_Color){255, 255, 255, 255});
            costTexture = SDL_CreateTextureFromSurface(renderer, costTextSurf);

            costDestRect.x = 320 - (costTextSurf->w / 2.0f);
            costDestRect.y = 240;
            costDestRect.w = costTextSurf->w / 2;
            costDestRect.h = costTextSurf->h / 2;
            SDL_RenderCopy(renderer, costTexture, NULL, &costDestRect);

            SDL_DestroyTexture(costTexture);
            SDL_FreeSurface(costTextSurf);
            for (int x = 0; x < w; x++) {
                for (int y = 0; y < h; y++) {
                    MAT_AT(NN_INPUT(nn), 0, 0) = (float)x / (w - 1);
                    MAT_AT(NN_INPUT(nn), 0, 1) = (float)y / (h - 1);
                    nnForward(nn);
                    uint8_t pixel = MAT_AT(NN_OUTPUT(nn), 0, 0) * 255.0f;

                    SDL_LockTexture(liveTexture, NULL, (void**)&livePixels,
                                    &livePitch);

                    livePixels[(x * 4) + (y * livePitch)] = pixel;
                    livePixels[(x * 4) + (y * livePitch) + 1] = pixel;
                    livePixels[(x * 4) + (y * livePitch) + 2] = pixel;
                    livePixels[(x * 4) + (y * livePitch) + 3] = pixel;

                    SDL_RenderCopy(renderer, liveTexture, NULL, &liveImgRect);
                    SDL_UnlockTexture(liveTexture);
                }
            }
        }

        SDL_RenderPresent(renderer);
    }

    // Actually upscale the image
    int outWidth = 256, outHeight = 256;
    uint8_t* outPixels = malloc(sizeof(uint8_t*) * outWidth * outHeight);
    for (int x = 0; x < outWidth; x++) {
        for (int y = 0; y < outHeight; y++) {
            MAT_AT(NN_INPUT(nn), 0, 0) = (float)x / (outWidth - 1);
            MAT_AT(NN_INPUT(nn), 0, 1) = (float)y / (outHeight - 1);
            nnForward(nn);
            uint8_t pixel = MAT_AT(NN_OUTPUT(nn), 0, 0) * 255.0f;

            outPixels[y * outWidth + x] = pixel;
        }
    }

    // Write the image to the FS with stb
    const char* outFile = "img.png";
    if (!stbi_write_png(outFile, outWidth, outHeight, 1, outPixels,
                        sizeof(*outPixels)*outWidth)) {
        fprintf(stderr, "ERROR: Couldn't make image\n");
        return 1;
    }

    NN_PRINT(nn);

    // Destroy EVERYTHING
    TTF_CloseFont(font);
    SDL_DestroyTexture(texture);
    SDL_DestroyRenderer(renderer);
    SDL_DestroyWindow(window);
    TTF_Quit();
    SDL_Quit();

    return 0;
}
