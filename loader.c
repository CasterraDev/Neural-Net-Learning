#define NN_IMPLEMENTATION
#include "nn.h"
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
    TRAINING_OUTPUT_COLSIZE
} NN_FILE_SECTION;

typedef struct {
    float* items;
    size_t count;
    size_t capacity;
} DA;

#define DynoAppend(p, item)                                                    \
    do {                                                                       \
        if ((p)->count >= (p)->capacity) {                                     \
            (p)->capacity = (p)->capacity == 0 ? 256 : (p)->capacity * 2;      \
            (p)->items =                                                       \
                realloc((p)->items, (p)->capacity * sizeof(*(p)->items));      \
        }                                                                      \
        (p)->items[(p)->count++] = (item);                                     \
    } while (0)

void readNumbers(DA* da, char* str) {
    char* p = str;
    while (*p) {
        if (isdigit(*p) || ((*p == '-' || *p == '+') && isdigit(*(p + 1)))) {
            long val = strtol(p, &p, 10);
            DynoAppend(da, val);
        } else {
            p++;
        }
    }
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

int main(int argc, char* argv[]) {
    char* fileName = argv[1];
    FILE* file;
    file = fopen(fileName, "r");
    if (file == NULL) {
        printf("ERROR: Couldn't open file\n");
        return 1;
    }

    printf("Contents of %s are:\n", fileName);
    NN_FILE_SECTION section;
    DA arch = {0};
    size_t batchSize = 0;
    size_t tiColSize = 0;
    size_t toColSize = 0;
    //TODO: Don't hard code this
    Mat data = matAlloc(3, 4);
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
        }

        char* str = buff;
        switch (section) {
            case (ARCH): {
                readNumbers(&arch, str);
                break;
            }
            case (BATCH_SIZE): {
                char* str = buff;
                batchSize = readFirstNumber(str);
                break;
            }
            case (DATA): {
                if (startOfData == -1) {
                    startOfData = lineNum;
                }
                DA row = {0};
                readNumbers(&row, buff);
                for (size_t i = 0; i < row.count; i++) {
                    MAT_AT(data, lineNum - startOfData, i) = row.items[i];
                }
                break;
            }
            case (TRAINING_INPUT_COLSIZE): {
                char* str = buff;
                tiColSize = readFirstNumber(str);
                break;
            }
            case (TRAINING_OUTPUT_COLSIZE): {
                char* str = buff;
                toColSize = readFirstNumber(str);
                break;
            }
        }
    }

    printf("Arch:\n");
    for (size_t i = 0; i < arch.count; i++) {
        printf("%f ", arch.items[i]);
    }
    printf("Batchsize: %zu\n", batchSize);
    printf("TIColSize: %zu\n", tiColSize);
    printf("TOColSize: %zu\n", toColSize);
    MAT_PRINT(data);

    fclose(file);
    return 0;
}
