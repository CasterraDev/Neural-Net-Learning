#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#define NN_IMPLEMENTATION
#include "nn.h"

#define BITS 4

int main(void) {
    srand(time(0));
    // srand(40);

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
    size_t runsAmt = 100 * 2000;
    char last = 0;
    float cost = 0;
    for (size_t i = 0; i < runsAmt; ++i) {
        size_t curBatch = i % batchCnt;
        size_t size = batchSize;
        size_t batchStart = curBatch * batchSize;
        // If on the last batch. The size most likely won't be a full batch, so
        // we fix the size and say it's the last batch of the trainingData so we
        // can calculate the avg cost.
        if (td.rows - batchStart < batchSize) {
            size = td.rows - batchStart;
            last = 1;
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

        if (last) {
            printf("Average Cost = %f\n", cost / batchSize);
            cost = 0;
            last = 0;
        }
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

    return 0;
}
