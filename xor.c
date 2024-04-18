#include <time.h>
#define NN_IMPLEMENTATION
#include "nn.h"

typedef struct {
    Mat a0;
    Mat w1, b1, a1;
    Mat w2, b2, a2;
} Xor;

float forwardXor(Xor x);

float cost(Xor x, Mat ti, Mat to) {
    float rt = 0;
    assert(ti.rows == to.rows);
    assert(to.cols == x.a2.cols);
    size_t n = ti.rows;
    for (size_t i = 0; i < n; i++) {
        Mat c = matRow(ti, i);
        Mat r = matRow(to, i);
        matCopy(x.a0, c);
        forwardXor(x);
        size_t ocn = to.cols;

        for (size_t j = 0; j < ocn; j++) {
            float d = MAT_AT(x.a2, 0, j) - MAT_AT(r, 0, j);
            rt += d * d;
        }
    }
    return rt / n;
}

float forwardXor(Xor x) {
    matDot(x.a1, x.a0, x.w1);
    matSum(x.a1, x.b1);
    matSig(x.a1);

    matDot(x.a2, x.a1, x.w2);
    matSum(x.a2, x.b2);
    matSig(x.a2);
    return *x.a2.es;
}

void finiteDiff(Xor x, Xor g, float eps, Mat ti, Mat to) {

    float saved;
    float c = cost(x, ti, to);

    for (size_t i = 0; i < x.w1.rows; i++) {
        for (size_t j = 0; j < x.w1.cols; j++) {
            saved = MAT_AT(x.w1, i, j);
            MAT_AT(x.w1, i, j) += eps;
            MAT_AT(g.w1, i, j) = (cost(x, ti, to) - c) / eps;
            MAT_AT(x.w1, i, j) = saved;
        }
    }

    for (size_t i = 0; i < x.b1.rows; i++) {
        for (size_t j = 0; j < x.b1.cols; j++) {
            saved = MAT_AT(x.b1, i, j);
            MAT_AT(x.b1, i, j) += eps;
            MAT_AT(g.b1, i, j) = (cost(x, ti, to) - c) / eps;
            MAT_AT(x.b1, i, j) = saved;
        }
    }

    for (size_t i = 0; i < x.w2.rows; i++) {
        for (size_t j = 0; j < x.w2.cols; j++) {
            saved = MAT_AT(x.w2, i, j);
            MAT_AT(x.w2, i, j) += eps;
            MAT_AT(g.w2, i, j) = (cost(x, ti, to) - c) / eps;
            MAT_AT(x.w2, i, j) = saved;
        }
    }

    for (size_t i = 0; i < x.b2.rows; i++) {
        for (size_t j = 0; j < x.b2.cols; j++) {
            saved = MAT_AT(x.b2, i, j);
            MAT_AT(x.b2, i, j) += eps;
            MAT_AT(g.b2, i, j) = (cost(x, ti, to) - c) / eps;
            MAT_AT(x.b2, i, j) = saved;
        }
    }
}

void xorlearn(Xor x, Xor g, float rate) {
    for (size_t i = 0; i < x.w1.rows; i++) {
        for (size_t j = 0; j < x.w1.cols; j++) {
            MAT_AT(x.w1, i, j) -= rate * MAT_AT(g.w1, i, j);
        }
    }

    for (size_t i = 0; i < x.b1.rows; i++) {
        for (size_t j = 0; j < x.b1.cols; j++) {
            MAT_AT(x.b1, i, j) -= rate * MAT_AT(g.b1, i, j);
        }
    }

    for (size_t i = 0; i < x.w2.rows; i++) {
        for (size_t j = 0; j < x.w2.cols; j++) {
            MAT_AT(x.w2, i, j) -= rate * MAT_AT(g.w2, i, j);
        }
    }

    for (size_t i = 0; i < x.b2.rows; i++) {
        for (size_t j = 0; j < x.b2.cols; j++) {
            MAT_AT(x.b2, i, j) -= rate * MAT_AT(g.b2, i, j);
        }
    }
}


Xor xorAlloc() {
    Xor x;
    x.a0 = matAlloc(1, 2);
    x.w1 = matAlloc(2, 2);
    x.b1 = matAlloc(1, 2);
    x.a1 = matAlloc(1, 2);

    x.w2 = matAlloc(2, 1);
    x.b2 = matAlloc(1, 1);
    x.a2 = matAlloc(1, 1);

    return x;
}

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
    srand(time(0));

    // The architecture is very important and something that you can tweak to make the nn better or worse.
    size_t arch[] = {2,3,1};
    // Allocate the neural net and the gradient net.
    NN nn = nnAlloc(arch, ARRAY_LEN(arch));
    NN g = nnAlloc(arch, ARRAY_LEN(arch));
    // Randomize the weights/bias of the neural net.
    nnRand(nn, 0, 1);

    float eps = 1e-1;
    float rate = 1e-1;
    size_t stride = 3;
    size_t n = sizeof(trainingData) / sizeof(trainingData[0]) / stride;

    // Split the training data into input and output
    Mat ti = {.rows = n, .cols = 2, .stride = stride, .es = trainingData};
    Mat to = {.rows = n, .cols = 1, .stride = stride, .es = trainingData + 2};

    printf("cost = %f\n", nnCost(nn, ti, to));

    // Train the nn using finiteDiff
    for (size_t i = 0; i < 100*1000; ++i){
        nnFiniteDiff(nn, g, eps, ti, to);
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
