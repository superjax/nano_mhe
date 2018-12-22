#pragma once

#include <Eigen/Core>
#include <unsupported/Eigen/AutoDiff>
#include <initializer_list>
#include <numeric>
#include <iostream>
#include <utility>

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


template <typename Scalar, typename Functor, int NO, int... NIs>
class CostFunctorAutoDiff : public Functor
{
    static constexpr int NIarr[sizeof...(NIs)] = {NIs...};
    static constexpr int NI = accumulator(NIarr, sizeof...(NIs));

    typedef Matrix<Scalar, NI, 1> xVec; // vector sized for total number of inputs
    typedef Matrix<Scalar, NO, 1> yVec; // vector sized for total number of outputs

    typedef AutoDiffScalar<xVec> ADScalar; // AD scalar type used for this function (one output and dual numbers for all inputs)
    typedef Matrix<ADScalar, NO, 1> yAD; // output vector (one AD scalar type for all outputs) (will have the jacobian in the .derivatives field)


public:
    template <typename... InPtrs, typename... JacPtrs>
    bool Evaluate(Scalar* res, const InPtrs... x, JacPtrs... j) const
    {
        // Do a bunch of checks to ensure all the matrices are the right size
        static_assert(sizeof...(j) == 0 || sizeof...(x) == sizeof...(j),
                      "supply either no jacobians, or the same number of "
                      "jacobians as inputs.");

        if (sizeof...(j) == 0)
        {
            return (*this)(res, x...); // simply forward arguments
        }

        yAD r_ad; // object to hold residual and jacobian (vs all inputs)
        int counter = NI;

        bool success = callJac(r_ad, x..., std::index_sequence_for<InPtrs...>());

        for (int i = 0; i < NO; i++)
            res(i) = r_ad(i).value();

        counter = 0;
        const int* arrptr = NIarr;
        int dummy[sizeof...(j)] = {extractJac(counter, arrptr, r_ad, j)...};
        (void)dummy;

        return success;
    }


    template<typename yAD, typename...InPtrs, std::size_t...Is>
    bool callJac(yAD& r, const InPtrs&... x, const std::index_sequence<Is...>&) const
    {
        return (*this)(r, startAD<ADScalar>(accumulator(NIarr, Is), Map<Matrix<Scalar, NIs, 1>>(x)).data()...);
    }

    template <typename ADType, int Rows>
    Matrix<ADType, Rows, 1> startAD(const int id, const Map<Matrix<Scalar, Rows, 1>>& x) const
    {
        Matrix<ADType, Rows, 1> x_ad = x.template cast<ADType>();
        for (int i = 0; i < Rows; i++)
        {
            x_ad[i].derivatives()(id + i) = 1.0;
        }
        return x_ad;
    }

    template <typename yAD, typename JacType>
    int extractJac(int& counter, const int*& cols, const yAD& r, JacType& j) const
    {
        for (int i = 0; i < JacType::RowsAtCompileTime; i++)
        {
            j.row(i) = r(i).derivatives().block(counter, 0, *(cols), 1).transpose();
        }
        counter+=*(cols);
        cols++;
        return 0;
    }
};
