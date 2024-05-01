#include "include/trainer.h"
#include <SDL2/SDL_blendmode.h>
#include <SDL2/SDL_pixels.h>
#include <SDL2/SDL_rect.h>
#include <SDL2/SDL_timer.h>
#include <SDL2/SDL_ttf.h>
#include <SDL2/SDL_video.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#define NN_IMPLEMENTATION
#include "include/nn.h"
#include "include/plot.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "include/stb_image_write.h"
#define STB_IMAGE_IMPLEMENTATION
#include "include/stb_image.h"

#include <SDL2/SDL_events.h>
#include <SDL2/SDL_render.h>
#include <dirent.h>
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

typedef enum {
    TRAINING,
    COMPUTING,
    SCANDIRS,
} Mode;

typedef struct {
    char** items;
    size_t count;
    size_t capacity;
} arrList;

typedef struct ImageFileHeader {
    uint32_t magic;
    uint32_t numOfImages;
    uint32_t numOfRows;
    uint32_t numOfCols;
} ImageFileHeader;

typedef struct LabelFileHeader {
    uint32_t magic;
    uint32_t numOfLabels;
} LabelFileHeader;

static uint32_t flipBytes(uint32_t n) {
    uint32_t b0, b1, b2, b3;

    b0 = (n & 0x000000ff) << 24u;
    b1 = (n & 0x0000ff00) << 8u;
    b2 = (n & 0x00ff0000) >> 8u;
    b3 = (n & 0xff000000) >> 24u;

    return (b0 | b1 | b2 | b3);
}

void listAppend(arrList* p, char* item) {
    if (p->count >= p->capacity) {
        p->capacity = p->capacity == 0 ? 256 : p->capacity * 2;
        p->items = realloc(p->items, p->capacity * sizeof(*p->items));
    }
    strcpy(p->items[p->count++], item);
}

void getImage(FILE* file, uint8_t* pixels) {
    size_t result = fread(pixels, sizeof(*pixels), 1, file);
    if (result != 1) {
        fprintf(stderr, "ERROR: Couldn't load image\n");
        return;
    }
}

uint8_t getLabel(FILE* file) {
    uint8_t label;
    size_t result = fread(&label, sizeof(uint8_t), 1, file);
    if (result != 1) {
        fprintf(stderr, "ERROR: Couldn't load label\n");
        exit(1);
    }

    return label;
}

FILE* openImageFile(char* filename, ImageFileHeader* header) {
    FILE* file = fopen(filename, "rb");
    if (file == NULL) {
        fprintf(stderr, "ERROR: Couldn't load image\n");
        return 0;
    }

    header->magic = 0;
    header->numOfImages = 0;
    header->numOfRows = 0;
    header->numOfCols = 0;

    fread(&header->magic, 4, 1, file);
    header->magic = flipBytes(header->magic);
    fread(&header->numOfImages, 4, 1, file);
    header->numOfImages = flipBytes(header->numOfImages);
    fread(&header->numOfRows, 4, 1, file);
    header->numOfRows = flipBytes(header->numOfRows);
    fread(&header->numOfCols, 4, 1, file);
    header->numOfCols = flipBytes(header->numOfCols);
    return file;
}

FILE* openLabelFile(char* filename, LabelFileHeader* header) {
    FILE* file = fopen(filename, "rb");
    if (file == NULL) {
        fprintf(stderr, "ERROR: Couldn't load image\n");
        return 0;
    }

    header->magic = 0;
    header->numOfLabels = 0;

    fread(&header->magic, 4, 1, file);
    header->magic = flipBytes(header->magic);
    fread(&header->numOfLabels, 4, 1, file);
    header->numOfLabels = flipBytes(header->numOfLabels);
    return file;
}

void getData(Mat* mat) {
    ImageFileHeader header;
    FILE* imageFile = openImageFile("./mnist/train-images-idx3-ubyte", &header);
    LabelFileHeader lheader;
    FILE* labelFile =
        openLabelFile("./mnist/train-labels-idx1-ubyte", &lheader);
    uint32_t imgSize = header.numOfRows * header.numOfCols;

    *mat = matAlloc(header.numOfImages, imgSize + 10);
    for (uint32_t i = 0; i < header.numOfImages; i++) {
        uint8_t pixels[header.numOfCols * header.numOfRows];
        getImage(imageFile, pixels);
        uint8_t label = getLabel(labelFile);
        int y = 0;
        for (uint32_t j = 0; j < imgSize; j++) {
            y = j;
            MAT_AT(*mat, i, y) = pixels[y];
        }
        y++;
        for (int k = 0; k < 10; k++) {
            if (k == label) {
                MAT_AT(*mat, i, y+k) = 1;
            } else {
                MAT_AT(*mat, i, y+k) = 0;
            }
        }
    }
}

int main(int argc, char* argv[]) {
    const char* programName = argsShift(&argc, &argv);

    if (argc <= 0) {
        printf("USAGE: ./%s [pathToImage]", programName);
        printf("ERROR: No image given\n");
    }

    // srand(time(0));
    srand(40);


    Mat allImgMat = {0};
    getData(&allImgMat);
    // REMEMBER: The nn architecture is a tweaky variable. The most important
    // one.
    size_t arch[] = {allImgMat.cols - 10, 64, 10};

    // Allocate the neural net and the gradient net.
    NN nn = nnAlloc(arch, ARRAY_LEN(arch));
    NN g = nnAlloc(arch, ARRAY_LEN(arch));
    // Randomize the weights/bias of the neural net.
    nnRand(nn, 0, 1);
    NN_PRINT(nn);

    // Shuffle the training data now. For batch training
    matShuffleRows(allImgMat);

    Plot plot = {0};

    for (size_t i = 0; i < allImgMat.rows; i++){
        Mat trainMat = {
            .rows = 10,
            .cols = allImgMat.cols,
            .stride = allImgMat.stride,
            .es = &MAT_AT(allImgMat, i, 0),
        };
        for (size_t i = 0; i < 10; i++) {
            tnrBatchTrain(&nn, &g, &trainMat, &plot, 5000, allImgMat.cols - 10, 10, 1);
            printf("Cost: %f\n", plot.items[plot.count-1]);
        }
    }

    printf("Finished\n");

    return 0;
}
