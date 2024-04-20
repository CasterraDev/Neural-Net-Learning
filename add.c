#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#define NN_IMPLEMENTATION
#include "nn.h"

#define BITS 4

int main(void) {
    // srand(time(0));
    srand(40);
    size_t n = (1 << BITS);
    size_t rows = n * n;
    Mat ti = matAlloc(rows, 2 * BITS);
    Mat to = matAlloc(rows, 1 + BITS);
    for (size_t i = 0; i < ti.rows; ++i) {
        size_t x = i / n;
        size_t y = i % n;
        size_t z = x + y;
        for (size_t j = 0; j < BITS; ++j) {
            MAT_AT(ti, i, j) = (x >> j) & 1;
            MAT_AT(ti, i, j + BITS) = (y >> j) & 1;
            MAT_AT(to, i, j) = (z >> j) & 1;
        }
        MAT_AT(to, i, BITS) = (z >= n);
    }
    MAT_PRINT(ti);
    MAT_PRINT(to);

    size_t arch[] = {2 * BITS, 4 * BITS, BITS + 1};
    // Allocate the neural net and the gradient net.
    NN nn = nnAlloc(arch, ARRAY_LEN(arch));
    NN g = nnAlloc(arch, ARRAY_LEN(arch));
    // Randomize the weights/bias of the neural net.
    nnRand(nn, 0, 1);
    NN_PRINT(nn);

    float rate = 1;
    for (size_t i = 0; i < 10 * 1000; ++i) {
        nnBackprop(nn, g, ti, to);
        nnLearn(nn, g, rate);
        printf("cost = %f\n", nnCost(nn, ti, to));
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
