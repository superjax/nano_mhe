#pragma once

#include <Eigen/Core>
#include <unsupported/Eigen/AutoDiff>
#include <initializer_list>
#include <numeric>
#include <iostream>

#ifndef NDEBUG
#define NANO_AD_ASSERT(condition, ...) \
    do { \
        if (! (condition)) { \
            std::cerr << "Assertion `" #condition "` failed in " << __FILE__ \
                      << " line " << __LINE__ << ": " << printf(__VA_ARGS__) << std::endl; \
            assert(condition); \
        } \
    } while (false)
#else
#   define ASSERT(condition, message) do { } while (false)
#endif

using namespace Eigen;

constexpr int accumulator(const int* arr, int size, int current_length = 0)
{
    return size == 0 ? current_length : accumulator(arr+1, size-1, current_length+(*arr));
}

constexpr int arrEq(const int* arr1, const int* arr2, const int size)
{
    return size == 0 ? true : ((*arr1) == (*arr2)) && (arrEq(arr1+1, arr2+1, size-1));
}

constexpr int arrEq(const int* arr1, const int arr2, const int size)
{
    return size == 0 ? true : ((*arr1) == arr2) && (arrEq(arr1+1, arr2, size-1));
}

class EmptyCostFunctor
{
public:
    template <typename OutType, typename... InTypes>
    bool operator()(OutType& res, const InTypes&... x) const
    {
        return true;
    }
};


template <typename Functor, typename OutType, typename... InTypes>
class CostFunctorAutoDiff : public Functor
{
    typedef typename OutType::Scalar Scalar;
    constexpr static int NO = OutType::RowsAtCompileTime;

public:
    template <typename... JacTypes>
    bool Evaluate(OutType& res, const InTypes&... x, JacTypes&... j) const
    {
        // Do a bunch of checks to ensure all the matrices are the right size
        static_assert(sizeof...(j) == 0 || sizeof...(x) == sizeof...(j),
                      "supply either no jacobians, or the same number of "
                      "jacobians as inputs.");
        constexpr int NIarr[sizeof...(x)] = {InTypes::RowsAtCompileTime...};
        constexpr int NI = accumulator(NIarr, sizeof...(x));

        constexpr int NJrowarr[sizeof...(j)] = {JacTypes::RowsAtCompileTime...};
        constexpr int NJcolarr[sizeof...(j)] = {JacTypes::ColsAtCompileTime...};
        static_assert(sizeof...(j) > 0 ? arrEq(NJrowarr, NO, sizeof...(x)) : true, "jacobian rows must match rows of output");
        static_assert(sizeof...(j) > 0 ? arrEq(NJcolarr, NIarr, sizeof...(x)): true, "jacobian cols must match rows of inputs");

        typedef Matrix<Scalar, NI, 1> xVec; // vector sized for total number of inputs
        typedef Matrix<Scalar, NO, 1> yVec; // vector sized for total number of outputs

        typedef AutoDiffScalar<xVec> ADScalar; // AD scalar type used for this function (one output and dual numbers for all inputs)
        typedef Matrix<ADScalar, NO, 1> yAD; // output vector (one AD scalar type for all outputs) (will have the jacobian in the .derivatives field)

        if (sizeof...(j) == 0)
        {
            return (*this)(res, x...); // simply forward arguments
        }

        yAD r_ad; // object to hold residual and jacobian (vs all inputs)
        bool success = (*this)(r_ad, x.template cast<ADScalar>()...); // convert all the inputs to ad types and push through functor

        int counter = 0;
        const int* arrptr = NIarr;

        for (int i = 0; i < NO; i++)
            res(i) = r_ad(i).value();
        int dummy[sizeof...(j)] = {extractJac(counter, arrptr, r_ad, j)...};
    }

    template <typename yAD, typename JacType>
    int extractJac(int& counter, const int*& cols, const yAD& r, JacType& j) const
    {
        for (int i = 0; i < NO; i++)
        {
            int i1 = counter;
            int i2 = counter + *(cols);
            MatrixXd test = r(i).derivatives().block(counter, 0, *(cols), 1);

            j.row(i) = test.transpose();
        }
        counter+=*(cols);
        cols++;
        return 0;
    }

};
