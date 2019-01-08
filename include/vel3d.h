#pragma once


#include <Eigen/Core>
using namespace Eigen;

template <typename T>
class Vel3D
{
public:
    typedef Matrix<T, 3, 1> Vec3;
    typedef Matrix<T, 3, 3> Mat3;
    enum
    {
        NumInputs = 3,
        NumResiduals = 3
    };

    Vel3D()
    {
        y_ << 0.0, 0.0, 0.0;
        active_ = false;
    }

    void set_meas(const Vec3& y, const Mat3& cov)
    {
        y_ = y;
        info_sqrt_ = cov.inverse().llt().matrixL().transpose();
        active_ = true;
    }

    template <typename Scalar>
    void operator()(const Scalar* xhat, Scalar* residual) const
    {
        Map<Vec3> r(residual);
        Map<Vec3> x(xhat);
        r = info_sqrt_ * (x - y_);
    }

    bool active_;
    Mat3 info_sqrt_;
    Vec3 y_;
};
