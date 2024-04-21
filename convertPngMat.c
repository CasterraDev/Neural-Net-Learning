#define NN_IMPLEMENTATION
#include "include/nn.h"
#include <assert.h>
#include <stdio.h>

#define STB_IMAGE_IMPLEMENTATION
#include "include/stb_image.h"

char *argsShift(int *argc, char*** argv){
    assert(*argc > 0);
    char *res = **argv;
    *argc -= 1;
    *argv += 1;
    return res;
}

int main(int argc, char* argv[]) {
    const char *programName = argsShift(&argc, &argv);

    if (argc <= 0){
        printf("USAGE: ./%s [pathToImage]", programName);
        printf("ERROR: No image given\n");
    }

    char* imgPath = argsShift(&argc, &argv);

    int w,h,c;
    uint8_t* data = (uint8_t*)stbi_load(imgPath, &w, &h, &c, 0);
    if (data == NULL){
        printf("ERROR: Couldn't load image %s\n", imgPath);
    }

    //Cols: x y grayVal(0-255)
    Mat t = matAlloc(w*h, 3);
    for (int x = 0; x < w; x++){
        for (int y = 0; y < h; y++){
            size_t i = y*w + x;
            MAT_AT(t, i, 0) = (float)x/(w-1);
            MAT_AT(t, i, 1) = (float)y/(h-1);
            MAT_AT(t, i, 2) = data[i]/255.f;
        }
    }
    char* path = strcat(imgPath,".mat");

    FILE* file = fopen(path, "wb");
    if (file == NULL) {
        printf("ERROR: Couldn't open file\n");
        return 1;
    }

    matSave(file, t);

    fclose(file);

    return 0;
}
