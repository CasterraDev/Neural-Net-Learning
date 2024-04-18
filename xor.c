#include <time.h>
#define NN_IMPLEMENTATION
#include "nn.h"

// clang-format off
// syntax: value1, value2, answer
float trainingData[] = {
    0, 0, 0,
    0, 1, 1,
    1, 0, 1,
    1, 1, 0,
};
// clang-format on

int main(void) {
    //srand(time(0));
    srand(40);

    // The architecture is very important and something that you can tweak to make the nn better or worse.
    size_t arch[] = {2,3,1};
    // Allocate the neural net and the gradient net.
    NN nn = nnAlloc(arch, ARRAY_LEN(arch));
    NN g = nnAlloc(arch, ARRAY_LEN(arch));
    // Randomize the weights/bias of the neural net.
    nnRand(nn, 0, 1);

    size_t stride = 3;
    size_t n = sizeof(trainingData) / sizeof(trainingData[0]) / stride;

    // Split the training data into input and output
    Mat ti = {.rows = n, .cols = 2, .stride = stride, .es = trainingData};
    Mat to = {.rows = n, .cols = 1, .stride = stride, .es = trainingData + 2};

    printf("cost = %f\n", nnCost(nn, ti, to));

    float rate = 1;
    // Train the nn using finiteDiff
    for (size_t i = 0; i < 5000; ++i){
        nnBackprop(nn, g, ti, to);
        nnLearn(nn, g, rate);
        printf("cost = %f\n", nnCost(nn, ti, to));
    }

    printf("--------------------------\n");

    NN_PRINT(nn);

    for (size_t i = 0; i < 2; ++i) {
        for (size_t j = 0; j < 2; ++j) {
            MAT_AT(NN_INPUT(nn), 0, 0) = i;
            MAT_AT(NN_INPUT(nn), 0, 1) = j;
            nnForward(nn);
            printf("%zu ^ %zu = %f\n", i, j, MAT_AT(NN_OUTPUT(nn), 0, 0));
        }
    }

    return 0;
}
