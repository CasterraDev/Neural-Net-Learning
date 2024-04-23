#pragma once
#include "nn.h"

void loadNNFromFile(char* fileName, ArchDA* arch, size_t* batchSize,
                    size_t* tiColSize, size_t* toColSize, size_t* tRowSize,
                    Mat* data);

Mat imgToMat(char* imgPath, int* w, int* h, int* c);
