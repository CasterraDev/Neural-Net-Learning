#pragma once
#include "nn.h"
#include "plot.h"

void tnrBatchTrain(NN* nn, NN* g, Mat* td, Plot* plot, size_t runsAmt, size_t batchSize, size_t tiColSize, size_t toColSize, size_t rate);