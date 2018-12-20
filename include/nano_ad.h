#pragma once

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
    return size == 0 ? current_length :
                       accumulator(arr+1, size-1, current_length+(*arr));
}

constexpr int arrEq(const int* arr1, const int* arr2, const int size)
{
    return size == 0 ? ((*arr1) == (*arr2)) :
                ((*arr1) == (*arr2)) && (arrEq(arr1+1, arr2+1, size-1));
}

constexpr int arrEq(const int* arr1, const int arr2, const int size)
{
    return size == 0 ? true :
                       ((*arr1) == arr2) && (arrEq(arr1+1, arr2, size-1));
}

template <typename OutType, typename... InType>
class CostFunctor
{
public:
    template <typename... JacType>
    bool Evaluate(OutType& res, const InType&... x, JacType&... j) const
    {
        // Do a bunch of checks to ensure all the matrices are the right size
        static_assert(sizeof...(j) == 0 || sizeof...(x) == sizeof...(j),
                      "supply either no jacobians, or the same number of "
                      "jacobians as inputs.");
        constexpr int NO = OutType::RowsAtCompileTime;
        constexpr int NIarr[sizeof...(x)] = {InType::RowsAtCompileTime...};
        constexpr int NI = accumulator(NIarr, sizeof...(x));

        constexpr int NJrowarr[sizeof...(j)] = {JacType::RowsAtCompileTime...};
        constexpr int NJcolarr[sizeof...(j)] = {JacType::ColsAtCompileTime...};
        NANO_AD_ASSERT(sizeof...(j) != 0 ? arrEq(NJrowarr, NO, sizeof...(x)) : true, "jacobian rows must match rows of output");
        NANO_AD_ASSERT(sizeof...(j) != 0 ? arrEq(NJcolarr, NIarr, sizeof...(x)): true, "jacobian cols must match rows of input");







    }
};
