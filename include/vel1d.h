#pragma once

#include <Eigen/Core>


template <typename T>
class Vel1D
{
public:
    Vel1D()
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
        active_ = true;
    }

    template <typename Scalar>
    void operator()(const Scalar* xhat, Scalar* residual) const
    {
        (*residual) = ((*xhat) - y_)/sqrt(*var_);
    }

    bool active_ = false;
    const T* var_;
    T y_;
};
