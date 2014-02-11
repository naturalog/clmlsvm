#ifndef CLASSIFIER_H
#define CLASSIFIER_H

#include "defs.h"
#include "QString"

class classifier
{
    matrix random(uint r, uint c);
    matrix ensgn(const matrix& y);
    matrix gram(const matrix& x, kernel_t kernel, bool vex);
    void peg(scalar T, scalar lambda);
    void create_trainset(uint xdim, uint ydim, uint N);
    void initcl();

public:
    classifier();

    trainset_t testset;
    uint xdim, ydim, N;
    trainset_t trainset;
    matrix train_hyperplane, K, alpha;

    void test(const QString& kernel);
    void run_test(
            const uint indim,
            const uint outdim,
            const float lambda,
            const QString& kernel,
            const uint ntest,
            const uint ntrain,
            const uint nbatch,
            const uint niters
            );
};

#endif // CLASSIFIER_H
