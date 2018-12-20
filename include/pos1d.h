#pragma once

#include <Eigen/Core>


template <typename T>
class Pos1D
{
public:
    Pos1D()
    {
        y_ = 0.0;
    }

    void init(T* pvar)
    {
        var_ = pvar;
    }

    void set_meas(const T& y)
    {
        y_ = y;
    }

    void operator()(const T* xhat, T* residual)
    {
        (*residual) = ((*xhat) - y_)/(*var_);
    }

    const T* var_;
    T y_;
};
