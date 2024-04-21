#pragma once
#ifndef NN_H_
#define NN_H_

#include <math.h>
#include <stddef.h>
#include <stdio.h>

#ifndef NN_MALLOC
#include <stdlib.h>
#define NN_MALLOC malloc
#endif // NN_MALLOC

#ifndef NN_ASSERT
#include <assert.h>
#define NN_ASSERT assert
#endif // NN_ASSERT

typedef struct {
    size_t rows;
    size_t cols;
    size_t stride;
    float* es;
} Mat;

typedef struct {
    float* items;
    size_t count;
    size_t capacity;
} MatDA;

typedef struct {
    size_t* items;
    size_t count;
    size_t capacity;
} ArchDA;

#define MAT_AT(m, i, j) (m).es[(i) * (m).stride + j]
#define ARRAY_LEN(xs) sizeof((xs)) / sizeof((xs)[0])
float randFloat(void);
float sigmoidf(float x);

Mat matAlloc(size_t rows, size_t cols);
void matRand(Mat m, float low, float high);
void matFill(Mat m, float n);
void matDot(Mat dst, Mat a, Mat b);
Mat matRow(Mat m, size_t row);
void matCopy(Mat dst, Mat src);
void matSum(Mat dst, Mat a);
void matSig(Mat m);
void matPrint(Mat m, const char* name, size_t p);
void matPrintRow(Mat m, size_t row);
void matShuffleRows(Mat m);
#define MAT_PRINT(m) matPrint(m, #m, 0);

typedef struct {
    size_t count;
    Mat* ws;
    Mat* bs;
    Mat* as; // The amount of activations is count+1
} NN;

#define NN_INPUT(nn) (nn).as[0]
#define NN_OUTPUT(nn) (nn).as[(nn).count]

NN nnAlloc(size_t* arch, size_t archCount);
void nnFill(NN nn, size_t val);
void nnPrint(NN nn, const char* name);
#define NN_PRINT(nn) nnPrint(nn, #nn);
void nnRand(NN nn, float low, float high);
void nnForward(NN nn);
float nnCost(NN nn, Mat ti, Mat to);
void nnFiniteDiff(NN nn, NN g, float eps, Mat ti, Mat to);
void nnBackprop(NN nn, NN g, Mat ti, Mat to);
void nnLearn(NN nn, NN g, float rate);

#endif // NN_H_

#ifdef NN_IMPLEMENTATION

float randFloat(void) {
    return (float)rand() / (float)RAND_MAX;
}

float sigmoidf(float x) {
    return 1.f / (1.f + expf(-x));
}

Mat matAlloc(size_t rows, size_t cols) {
    Mat m;
    m.rows = rows;
    m.cols = cols;
    m.stride = cols;
    m.es = NN_MALLOC(sizeof(*m.es) * rows * cols);
    NN_ASSERT(m.es != NULL);
    return m;
}

void matRand(Mat m, float low, float high) {
    for (size_t i = 0; i < m.rows; i++) {
        for (size_t j = 0; j < m.cols; j++) {
            MAT_AT(m, i, j) = randFloat() * (high - low) + low;
        }
    }
}

void matFill(Mat m, float n) {
    for (size_t i = 0; i < m.rows; i++) {
        for (size_t j = 0; j < m.cols; j++) {
            MAT_AT(m, i, j) = n;
        }
    }
}

void matDot(Mat dst, Mat a, Mat b) {
    NN_ASSERT(a.cols == b.rows);
    size_t n = a.cols;
    NN_ASSERT(dst.rows == a.rows);
    NN_ASSERT(dst.cols == b.cols);
    for (size_t i = 0; i < dst.rows; i++) {
        for (size_t j = 0; j < dst.cols; j++) {
            for (size_t k = 0; k < n; k++) {
                MAT_AT(dst, i, j) += MAT_AT(a, i, k) * MAT_AT(b, k, j);
            }
        }
    }
}

Mat matRow(Mat m, size_t row) {
    return (Mat){.rows = 1,
                 .cols = m.cols,
                 .stride = m.stride,
                 .es = &MAT_AT(m, row, 0)};
}

void matCopy(Mat dst, Mat src) {
    NN_ASSERT(dst.rows == src.rows);
    NN_ASSERT(dst.cols == src.cols);

    for (size_t i = 0; i < dst.rows; i++) {
        for (size_t j = 0; j < dst.cols; j++) {
            MAT_AT(dst, i, j) = MAT_AT(src, i, j);
        }
    }
}

void matSum(Mat dst, Mat a) {
    NN_ASSERT(dst.rows == a.rows);
    NN_ASSERT(dst.cols == a.cols);
    for (size_t i = 0; i < dst.rows; i++) {
        for (size_t j = 0; j < dst.cols; j++) {
            MAT_AT(dst, i, j) += MAT_AT(a, i, j);
        }
    }
}

void matSig(Mat m) {
    for (size_t i = 0; i < m.rows; i++) {
        for (size_t j = 0; j < m.cols; j++) {
            MAT_AT(m, i, j) = sigmoidf(MAT_AT(m, i, j));
        }
    }
}

void matPrint(Mat m, const char* name, size_t padding) {
    printf("%*s%s = [\n", (int)padding, "", name);
    for (size_t i = 0; i < m.rows; i++) {
        printf("%*s    ", (int)padding, "");
        for (size_t j = 0; j < m.cols; j++) {
            printf("%f ", MAT_AT(m, i, j));
        }
        printf("\n");
    }
    printf("%*s]\n", (int)padding, "");
}

void matPrintRow(Mat m, size_t row) {
    printf(" ");
    for (size_t j = 0; j < m.cols; j++) {
        printf("%f ", MAT_AT(m, row, j));
    }
    printf("\n");
}

void matShuffleRows(Mat m) {
    for (size_t i = 0; i < m.rows; i++) {
        size_t randRow = i + rand() % (m.rows - i);
        for (size_t k = 0; k < m.cols; k++) {
            float save = MAT_AT(m, i, k);
            MAT_AT(m, i, k) = MAT_AT(m, randRow, k);
            MAT_AT(m, randRow, k) = save;
        }
    }
}

NN nnAlloc(size_t* arch, size_t arch_count) {
    NN_ASSERT(arch_count > 0);

    NN nn;
    nn.count = arch_count - 1;

    nn.ws = NN_MALLOC(sizeof(*nn.ws) * nn.count);
    NN_ASSERT(nn.ws != NULL);
    nn.bs = NN_MALLOC(sizeof(*nn.bs) * nn.count);
    NN_ASSERT(nn.bs != NULL);
    nn.as = NN_MALLOC(sizeof(*nn.as) * (nn.count + 1));
    NN_ASSERT(nn.as != NULL);

    nn.as[0] = matAlloc(1, arch[0]);
    for (size_t i = 1; i < arch_count; ++i) {
        nn.ws[i - 1] = matAlloc(nn.as[i - 1].cols, arch[i]);
        nn.bs[i - 1] = matAlloc(1, arch[i]);
        nn.as[i] = matAlloc(1, arch[i]);
    }

    return nn;
}

void nnFill(NN nn, size_t val) {
    for (size_t i = 0; i < nn.count; ++i) {
        matFill(nn.ws[i], val);
        matFill(nn.bs[i], val);
        matFill(nn.as[i], val);
    }
    matFill(nn.as[nn.count], 0);
}

void nnPrint(NN nn, const char* name) {
    char buf[256];
    printf("%s = [\n", name);
    for (size_t i = 0; i < nn.count; ++i) {
        snprintf(buf, sizeof(buf), "ws%zu", i);
        matPrint(nn.ws[i], buf, 4);
        snprintf(buf, sizeof(buf), "bs%zu", i);
        matPrint(nn.bs[i], buf, 4);
    }
    printf("]\n");
}

void nnRand(NN nn, float low, float high) {
    for (size_t i = 0; i < nn.count; ++i) {
        matRand(nn.ws[i], low, high);
        matRand(nn.bs[i], low, high);
    }
}

void nnForward(NN nn) {
    for (size_t i = 0; i < nn.count; ++i) {
        // Multipling as[i] by ws[i] and putting into as[i + 1]
        matDot(nn.as[i + 1], nn.as[i], nn.ws[i]);
        // Adding the bias
        matSum(nn.as[i + 1], nn.bs[i]);
        // Activating the activation layer
        matSig(nn.as[i + 1]);
    }
}

float nnCost(NN nn, Mat ti, Mat to) {
    NN_ASSERT(ti.rows == to.rows);
    NN_ASSERT(to.cols == NN_OUTPUT(nn).cols);
    size_t n = ti.rows;

    float c = 0;
    for (size_t i = 0; i < n; ++i) {
        Mat x = matRow(ti, i);
        Mat y = matRow(to, i);

        matCopy(NN_INPUT(nn), x);
        nnForward(nn);
        size_t q = to.cols;
        for (size_t j = 0; j < q; ++j) {
            float d = MAT_AT(NN_OUTPUT(nn), 0, j) - MAT_AT(y, 0, j);
            c += d * d;
        }
    }

    return c / n;
}

void nnFiniteDiff(NN nn, NN g, float eps, Mat ti, Mat to) {
    float saved;
    float c = nnCost(nn, ti, to);

    for (size_t i = 0; i < nn.count; ++i) {
        for (size_t j = 0; j < nn.ws[i].rows; ++j) {
            for (size_t k = 0; k < nn.ws[i].cols; ++k) {
                saved = MAT_AT(nn.ws[i], j, k);
                MAT_AT(nn.ws[i], j, k) += eps;
                MAT_AT(g.ws[i], j, k) = (nnCost(nn, ti, to) - c) / eps;
                MAT_AT(nn.ws[i], j, k) = saved;
            }
        }

        for (size_t j = 0; j < nn.bs[i].rows; ++j) {
            for (size_t k = 0; k < nn.bs[i].cols; ++k) {
                saved = MAT_AT(nn.bs[i], j, k);
                MAT_AT(nn.bs[i], j, k) += eps;
                MAT_AT(g.bs[i], j, k) = (nnCost(nn, ti, to) - c) / eps;
                MAT_AT(nn.bs[i], j, k) = saved;
            }
        }
    }
}

void nnBackprop(NN nn, NN g, Mat ti, Mat to) {
    NN_ASSERT(ti.rows == to.rows);
    NN_ASSERT(NN_OUTPUT(nn).cols == to.cols);

    nnFill(g, 0);

    // Current sample
    for (size_t i = 0; i < ti.rows; ++i) {
        // Put the training input into the nn
        matCopy(NN_INPUT(nn), matRow(ti, i));
        // Run the nn and get the 'answers'
        nnForward(nn);

        // Zero out the gradient activation matrix so garbage data won't mess
        // with testing
        for (size_t j = 0; j <= nn.count; ++j) {
            matFill(g.as[j], 0);
        }

        // Put the (nn calculated value - desired value) difference into the
        // gradient matrix
        for (size_t j = 0; j < to.cols; ++j) {
            MAT_AT(NN_OUTPUT(g), 0, j) =
                MAT_AT(NN_OUTPUT(nn), 0, j) - MAT_AT(to, i, j);
        }

        // Reverse my way through the layers
        for (size_t l = nn.count; l > 0; --l) {
            for (size_t col = 0; col < nn.as[l].cols; ++col) {
                float act = MAT_AT(nn.as[l], 0, col);
                float ga = MAT_AT(g.as[l], 0, col);
                float equation = 2 * ga * act * (1 - act);
                MAT_AT(g.bs[l - 1], 0, col) += equation;
                // Prev activations
                for (size_t row = 0; row < nn.as[l - 1].cols; ++row) {
                    float prevAct = MAT_AT(nn.as[l - 1], 0, row);
                    float prevW = MAT_AT(nn.ws[l - 1], row, col);
                    MAT_AT(g.ws[l - 1], row, col) += equation * prevAct;
                    MAT_AT(g.as[l - 1], 0, row) += equation * prevW;
                }
            }
        }
    }

    for (size_t i = 0; i < g.count; ++i) {
        for (size_t row = 0; row < g.ws[i].rows; ++row) {
            for (size_t col = 0; col < g.ws[i].cols; ++col) {
                MAT_AT(g.ws[i], row, col) /= ti.rows;
            }
        }
        for (size_t row = 0; row < g.bs[i].rows; ++row) {
            for (size_t col = 0; col < g.bs[i].cols; ++col) {
                MAT_AT(g.bs[i], row, col) /= ti.rows;
            }
        }
    }
}

void nnLearn(NN nn, NN g, float rate) {
    for (size_t i = 0; i < nn.count; ++i) {
        for (size_t j = 0; j < nn.ws[i].rows; ++j) {
            for (size_t k = 0; k < nn.ws[i].cols; ++k) {
                MAT_AT(nn.ws[i], j, k) -= rate * MAT_AT(g.ws[i], j, k);
            }
        }

        for (size_t j = 0; j < nn.bs[i].rows; ++j) {
            for (size_t k = 0; k < nn.bs[i].cols; ++k) {
                MAT_AT(nn.bs[i], j, k) -= rate * MAT_AT(g.bs[i], j, k);
            }
        }
    }
}
#endif // NN_IMPLEMENTATION
