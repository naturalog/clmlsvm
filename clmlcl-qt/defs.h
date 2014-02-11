#ifndef DEFS_H
#define DEFS_H

#include <eigen3/Eigen/Eigen>
#include <utility>
#include <cstdlib>
#include <cmath>

typedef float scalar;
typedef Eigen::Matrix<scalar, Eigen::Dynamic, Eigen::Dynamic> matrix;
typedef Eigen::Matrix<scalar, 1, Eigen::Dynamic> rowvec;
typedef rowvec vector;
typedef Eigen::Matrix<scalar, Eigen::Dynamic, 1> colvec;
typedef std::pair<matrix, matrix> trainset_t;
typedef scalar (kernel_t)(const vector& x, const vector& y);

#define CLSOURCE "cls.cl"

template<typename T>
struct cl_matrix
{
    cl_matrix(uint _rows, uint _cols) :
        rows(_rows), cols(_cols), data(new T[_rows * _cols]) {}
    cl_matrix(const matrix& x) :
        rows(x.rows()), cols(x.cols()), data(new T[x.rows() * x.cols()]) {
        for (uint n = 0; n < rows; n++)
            for (uint k = 0; k < cols; k++)
                data[n * cols + k] = x(n, k);
    }

    operator matrix() const
    {
        matrix x(rows, cols);
        for (uint n = 0; n < rows; n++)
            for (uint k = 0; k < cols; k++)
                x(n, k) = data[n * cols + k];
        return x;
    }

    T* data;
    uint rows, cols;
    T& operator()(uint r, uint c) { return data[r * cols + c]; }
    const T& operator()(uint r, uint c) const { return data[r * cols + c]; }
    ~cl_matrix() { delete[] data; data = 0; }
    uint size() const { return rows * cols * sizeof(T); }
    static uint size(uint rows, uint cols) { return rows * cols * sizeof(T); }
};

#endif // DEFS_H
