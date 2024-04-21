#include "trainer.h"
#define NN_IMPLEMENTATION
#include "nn.h"
#include <stdio.h>
#include <strings.h>

void tnrBatchTrain(NN* nn, NN* g, Mat* td, Plot* plot, size_t runsAmt, size_t batchSize, size_t tiColSize, size_t toColSize, size_t rate){
    size_t batchCnt = (td->rows + batchSize - 1) / batchSize;
    size_t runs = 0;
    float cost = 0;
    matShuffleRows(*td);
    for (size_t i = 0; i < batchCnt && runs < runsAmt; ++i) {
        size_t size = batchSize;
        size_t batchStart = i * batchSize;
        // If on the last batch. The size most likely won't be a full
        // batch, so we fix the size
        if (td->rows - batchStart < batchSize) {
            size = td->rows - batchStart;
        }

        // Split the training input/output
        Mat ti = {.rows = size,
                  .cols = tiColSize,
                  .stride = td->stride,
                  .es = &MAT_AT(*td, batchStart, 0)};
        Mat to = {.rows = size,
                  .cols = toColSize,
                  .stride = td->stride,
                  .es = &MAT_AT(*td, batchStart, ti.cols)};

        // Train this batch
        nnBackprop(*nn, *g, ti, to);
        nnLearn(*nn, *g, rate);
        cost += nnCost(*nn, ti, to);

        if (i == batchCnt - 1) {
            printf("%zu: Average Cost = %f\n", i, cost / batchSize);
            plotAppend(plot, cost / batchSize);
            cost = 0;
        }
    }
}
