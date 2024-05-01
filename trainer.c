#include "include/trainer.h"
#include "include/nn.h"
#include <strings.h>

void tnrBatchTrain(NN* nn, NN* g, Mat* td, Plot* plot, size_t batchSize, size_t tiColSize, size_t toColSize, float rate){
    size_t batchCnt = (td->rows + batchSize - 1) / batchSize;
    float cost = 0;
    for (size_t i = 0; i < batchCnt; ++i) {
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
    }

    plotAppend(plot, cost / batchSize);
}
