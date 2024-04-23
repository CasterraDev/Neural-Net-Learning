#include <stdint.h>
#define NN_IMPLEMENTATION
#include "include/nn.h"
#define STB_IMAGE_IMPLEMENTATION
#include "include/stb_image.h"
#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>

int strEqualIArr(const char* str0, const char* str1) {
    return strstr(str0, str1) != NULL;
}

char* strSub(const char* str, const char* sub) {
    return strstr(str, sub);
}

typedef enum NN_FILE_SECTION {
    ARCH,
    BATCH_SIZE,
    DATA,
    TRAINING_INPUT_COLSIZE,
    TRAINING_OUTPUT_COLSIZE,
    TRAINING_ROW_SIZE
} NN_FILE_SECTION;

#define DynoAppend(p, item)                                                    \
    do {                                                                       \
        if ((p)->count >= (p)->capacity) {                                     \
            (p)->capacity = (p)->capacity == 0 ? 256 : (p)->capacity * 2;      \
            (p)->items =                                                       \
                realloc((p)->items, (p)->capacity * sizeof(*(p)->items));      \
        }                                                                      \
        (p)->items[(p)->count++] = (item);                                     \
    } while (0)

#define readNumbers(da, str)                                                   \
    char* p = (str);                                                           \
    while (*p) {                                                               \
        if (isdigit(*p) || ((*p == '-' || *p == '+') && isdigit(*(p + 1)))) {  \
            long val = strtol(p, &p, 10);                                      \
            DynoAppend((da), val);                                             \
        } else {                                                               \
            p++;                                                               \
        }                                                                      \
    }

float readFirstNumber(char* str) {
    char* p = str;
    while (*p) {
        if (isdigit(*p) || ((*p == '-' || *p == '+') && isdigit(*(p + 1)))) {
            long val = strtol(p, &p, 10);
            return val;
        } else {
            p++;
        }
    }
    printf("ERROR: Getting first number");
    return 0;
}

void loadNNFromFile(char* fileName, ArchDA* arch, size_t* batchSize,
                    size_t* tiColSize, size_t* toColSize, size_t* tRowSize,
                    Mat* data) {
    FILE* file;
    printf("%s", fileName);
    file = fopen(fileName, "r");
    if (file == NULL) {
        printf("ERROR: Couldn't open file\n");
        return;
    }

    printf("Contents of %s are:\n", fileName);
    NN_FILE_SECTION section;
    int startOfData = -1;

    char buff[255];
    size_t lineNum = 0;
    while (fgets(buff, 255, file)) {
        lineNum++;
        if (strEqualIArr(buff, "--Arch--")) {
            section = ARCH;
            continue;
        } else if (strEqualIArr(buff, "--BatchSize--")) {
            section = BATCH_SIZE;
            continue;
        } else if (strEqualIArr(buff, "--Data--")) {
            section = DATA;
            continue;
        } else if (strEqualIArr(buff, "--TrainingInputColSize--")) {
            section = TRAINING_INPUT_COLSIZE;
            continue;
        } else if (strEqualIArr(buff, "--TrainingOutputColSize--")) {
            section = TRAINING_OUTPUT_COLSIZE;
            continue;
        } else if (strEqualIArr(buff, "--TrainingRowSize")) { section = TRAINING_ROW_SIZE;
            continue;
        }

        char* str = buff;
        switch (section) {
            case (ARCH): {
                readNumbers(arch, str);
                break;
            }
            case (BATCH_SIZE): {
                char* str = buff;
                *batchSize = readFirstNumber(str);
                break;
            }
            case (DATA): {
                if (tiColSize == 0 || toColSize == 0 || tRowSize == 0) {
                    printf("TrainingInputColSize, TrainingOutputColSize, and "
                           "TrainingRowSize must be before Data in the config "
                           "file.");
                    return;
                }
                if (startOfData == -1) {
                    startOfData = lineNum;
                    *data = matAlloc(*tRowSize, *tiColSize + *toColSize);
                }
                MatDA row = {0};
                readNumbers(&row, buff);
                for (size_t i = 0; i < row.count; i++) {
                    MAT_AT(*data, lineNum - startOfData, i) = row.items[i];
                }
                break;
            }
            case (TRAINING_INPUT_COLSIZE): {
                char* str = buff;
                *tiColSize = readFirstNumber(str);
                break;
            }
            case (TRAINING_OUTPUT_COLSIZE): {
                char* str = buff;
                *toColSize = readFirstNumber(str);
                break;
            }
            case (TRAINING_ROW_SIZE): {
                char* str = buff;
                *tRowSize = readFirstNumber(str);
                break;
            }
        }
    }
    fclose(file);
}

Mat imgToMat(char* imgPath, int* w, int* h, int* c){
    uint8_t* data = (uint8_t*)stbi_load(imgPath, w, h, c, 0);
    if (data == NULL){
        printf("ERROR: Couldn't load image %s\n", imgPath);
    }

    //Cols: x y grayVal(0-255)
    Mat t = matAlloc(*w**h, 3);
    for (int x = 0; x < *w; x++){
        for (int y = 0; y < *h; y++){
            size_t i = y**w + x;
            MAT_AT(t, i, 0) = (float)x/(*w-1);
            MAT_AT(t, i, 1) = (float)y/(*h-1);
            MAT_AT(t, i, 2) = data[i]/255.f;
        }
    }
    return t;
}
