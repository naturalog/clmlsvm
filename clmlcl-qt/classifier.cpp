#define __CL_ENABLE_EXCEPTIONS
#include "classifier.h"
#include <vector>
#include <iostream>
#include <fstream>
#include <string>
#include <CL/cl.hpp>
#include <functional>
#include <time.h>
#include <iostream>
#include <fstream>
#include <exception>
#include <cstdlib>
#include <vector>
#include <ctime>
#include <stdexcept>

//cl::Context context;
//std::vector<cl::CommandQueue> queues;
//cl::Program program;

classifier::classifier()
{
    initcl();
}

void classifier::initcl()
{
//    cl_int status = 0;
//    context = cl::Context(CL_DEVICE_TYPE_GPU, 0, 0, 0, &status);
//    if (status != CL_SUCCESS)
//    {
//        std::cout<<"GPU not found, falling back to CPU!"<<std::endl;
//        context = cl::Context(CL_DEVICE_TYPE_CPU, 0, 0, 0, &status);
//        if (status != CL_SUCCESS)
//            throw std::runtime_error("Error: Creating context!");
//    }

//    std::ifstream file(CLSOURCE);
//    std::string sourceCode(std::istreambuf_iterator<char>(file), (std::istreambuf_iterator<char>()));
//    cl::Program::Sources source(1, std::make_pair(sourceCode.c_str(), sourceCode.length() + 1));
//    program = cl::Program(context, source);

//    // Get a list of devices on this platform
//    std::vector<cl::Device> devices = context.getInfo<CL_CONTEXT_DEVICES>();
//    //if(devices.size() != 1) devices.erase(devices.begin() + 1, devices.end());
//    for (auto it = devices.begin(); it != devices.end(); it++)
//    {
//        std::cout<<it->getInfo<CL_DEVICE_NAME>()<<std::endl;
//        for (uint n = 0; n < devices.size(); n++)
//            queues.push_back(cl::CommandQueue(context, *it));

//        try
//        {
//            program.build(std::vector<cl::Device>(1,*it), "");
//        } catch(cl::Error e) {
//            std::cout << "Build Status: " << program.getBuildInfo<CL_PROGRAM_BUILD_STATUS>((*it)()) << std::endl
//                      << "Build Log:\t " << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>((*it)()) << std::endl
//                      << e.what() << std::endl<< e.err() << std::endl;
//        }
//    }
}

#include <vexcl/vexcl.hpp>

vex::Context ctx(vex::Filter::Type(CL_DEVICE_TYPE_GPU) && vex::Filter::Count(1));

// http://stackoverflow.com/questions/21682012/gram-matrix-using-vexcl
void clGram(//uint d, // device id
            const cl_matrix<float> data,
            cl_matrix<float>* _gram,
            const char* type,
            float sigma)
{
    uint m = data.rows, n = data.cols;

    vex::vector<float> chunk1(ctx, m * n);
    vex::vector<float> chunk2(ctx, m * n);

    vex::copy(std::vector<float>(data.data, &data.data[data.cols * data.rows]), chunk1);
    vex::copy(std::vector<float>(data.data, &data.data[data.cols * data.rows]), chunk2);

    vex::vector<float> gram(ctx, m * m);
    using vex::extents;

    auto x = vex::reshape(chunk1, extents[m][m][n], extents[0][2]);
    auto y = vex::reshape(chunk2, extents[m][m][n], extents[1][2]);

//    std::cout<<"data:\n"<<(matrix)data<<std::endl;

    gram = vex::reduce<vex::SUM>(
                extents[m][m][n],
                x * y,
                2
                );
    //gram = vex::exp(gram * gram * (-.5*sigma));
//    std::cout<<"gram:\n"<<gram<<std::endl;
    vex::copy<float>(gram, _gram->data);
}

scalar linear_kernel(const vector& x, const vector& y) { return x.dot(y); }
scalar gaussian_kernel(const vector& x, const vector& y) { return exp(-.5*(x-y).squaredNorm()); }

void classifier::run_test(
        const uint indim,
        const uint outdim,
        const float lambda,
        const QString& kernel,
        const uint ntest,
        const uint ntrain,
        const uint nbatch,
        const uint niters
        )
{
    xdim = indim;
    ydim = outdim;
    N = ntrain;
    using namespace std;
    create_trainset(indim, outdim, ntrain);
    clock_t begin = clock();
    K = gram(trainset.first,
             kernel == "Linear" ? linear_kernel : gaussian_kernel,
             false);
    clock_t end = clock();
    std::cout<<"cpp gram: " << double(end - begin) / CLOCKS_PER_SEC << " sec"<<std::endl;
    cl_matrix<float> K2(trainset.first.rows(), trainset.first.rows());
    begin = clock();
    clGram(trainset.first, &K2,
           kernel == "Linear" ? "linear" : "gauss", 1);
    end = clock();
    std::cout<<"cl gram: " << double(end - begin) / CLOCKS_PER_SEC << " sec"<<std::endl;
        matrix kk2(K2);
        //std::cout<<"clgram:\n"<<kk2<<std::endl;
        //std::cout<<"gram:\n"<<K<<std::endl;
    //    std::cout<<"\nx:\n"<<trainset.first<<std::endl;
    //std::cout<<"G-C: "<<(K-matrix(K2))<<std::endl;
    std::cout<<"||G/C||: "<<(K-matrix(K2)).norm()<<std::endl;
    peg(niters, lambda);
}

vector vclassify(const vector& x, const matrix& X, const matrix& alpha, kernel_t kernel)
{
    vector k(X.rows());
    for (uint r = 0; r < X.rows(); r++)
        k(r) = kernel(x, X.row(r));
    return alpha * k.transpose();
}

matrix classify(const matrix& x, const matrix& X, const matrix& alpha, kernel_t kernel)
{
    matrix y(x.rows(), alpha.rows());
    for (uint r = 0; r < x.rows(); r++)
        y.row(r) = vclassify(x.row(r), X, alpha, kernel);
    return y;
}

void classifier::test(const QString& kernel)
{
    testset.first = random(N, xdim);
    testset.second = ensgn(classify(testset.first, trainset.first, alpha,
                                    kernel == "Linear" ? linear_kernel : gaussian_kernel));
    matrix real = ensgn(testset.first * train_hyperplane.transpose());
    scalar err = 0;
    for (uint r = 0; r < real.rows(); r++)
        for (uint c = 0; c < real.cols(); c++)
            if (real(r, c) != testset.second(r, c))
                err++;

    //    showmatrix(real, "real");
    //    showmatrix(testset.second, "hat");
    std::cout<<"err: "<<err/scalar(real.rows() * real.cols())<<std::endl;
}

matrix classifier::random(uint r, uint c)
{
    matrix x(r,c);
    for (uint n = 0; n < r; n++)
        for (uint k = 0; k < c; k++)
            x(n,k) = drand48() * 2. - 1.;
    return x;
}

matrix classifier::ensgn(const matrix& y)
{
    matrix x(y.rows(), y.cols());
    for (uint n = 0; n < y.rows(); n++)
        for (uint k = 0; k < y.cols(); k++)
            x(n,k) = (y(n,k) > 0) ? 1 : -1;
    return x;
}

matrix classifier::gram(const matrix& x, kernel_t kernel, bool vex)
{
    matrix K(x.rows(), x.rows());
    //if (!vex)
    for (uint n = 0; n < x.rows(); n++)
        for (uint k = 0; k < x.rows(); k++)
            K(n,k) = kernel(x.row(n), x.row(k));
    //else
    {

    }


    return K;
}

void classifier::peg(scalar T, scalar lambda)
{
    uint N = trainset.second.rows(), ydim = trainset.second.cols();
    alpha = matrix::Zero(ydim, N);
    for (scalar t = 1; t <= T; t++)
    {
        uint sample = rand() % N;
        for (uint n = 0; n < ydim; n++)
            if (trainset.second(sample, n) * K.row(sample).dot(alpha.row(n)) < lambda * t)
                alpha(n, sample)++;
    }
}

void classifier::create_trainset(uint xdim, uint ydim, uint N) // returns the hyperplane w
{
    train_hyperplane = random(ydim, xdim);
    trainset.first = random(N, xdim);
    trainset.second = ensgn(trainset.first * train_hyperplane .transpose());
    //showmatrix(trainset.second);
}

