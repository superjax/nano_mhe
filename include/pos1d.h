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

    void set_meas(const T& y, const T& pvar)
    {
        y_ = y;
        stdev_ = sqrt(pvar);
        active_ = true;
    }

    template <typename Scalar>
    void operator()(const Scalar* xhat, Scalar* residual) const
    {
        (*residual) = ((*xhat) - y_)/stdev_;
    }

    bool active_ = false;
    T stdev_;
    T y_;
};
