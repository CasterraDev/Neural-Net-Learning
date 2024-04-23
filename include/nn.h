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

#define MAT_AT(m, r, c) (m).es[(r) * (m).stride + c]
#define ARRAY_LEN(xs) sizeof((xs)) / sizeof((xs)[0])
float randFloat(void);
float sigmoidf(float x);

Mat matAlloc(size_t rows, size_t cols);
void matSave(FILE* out, Mat m);
Mat matLoad(FILE* in);
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
