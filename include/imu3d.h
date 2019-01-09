#pragma once

#include <Eigen/Core>
#include "geometry/xform.h"

using namespace Eigen;
using namespace xform;

#ifndef NDEBUG
#define NANO_IMU_ASSERT(condition, ...) \
    do { \
        if (! (condition)) { \
            std::cerr << "Assertion `" #condition "` failed in " << __FILE__ \
                      << " line " << __LINE__ << ": " << printf(__VA_ARGS__) << std::endl; \
            assert(condition); \
        } \
    } while (false)
#else
#   define NANO_IMU_ASSERT(...)
#endif

template <typename Scalar>
class Imu3D
{
public:
    enum
    {
        NRES = 9
    };
    typedef Quat<Scalar> QuatT;
    typedef Xform<Scalar> XformT;
    typedef Matrix<Scalar, 3, 1> Vec3;
    typedef Matrix<Scalar, 6, 1> Vec6;
    typedef Matrix<Scalar, 9, 1> Vec9;
    typedef Matrix<Scalar, 10, 1> Vec10;

    typedef Matrix<Scalar, 6, 6> Mat6;
    typedef Matrix<Scalar, 9, 9> Mat9;
    typedef Matrix<Scalar, 9, 6> Mat96;

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    Imu3D()
    {
        t0_ = INFINITY;
        delta_t_ = 0.0;
        b_.setZero();
        y_.setZero();
        P_.setZero();
        J_.setZero();
    }


    void reset(const Scalar& _t0, const Vec6& b0)
    {
        delta_t_ = 0.0;
        t0_ = _t0;
        b_ = b0;

        n_updates_ = 0;
        y_.setZero();
        y_(Q) = 1.0;
        P_.setZero();
        J_.setZero();
    }

    void errorStateDynamics(const Vec10& y, const Vec9& dy, const Vec6& u, const Vec6& eta, Vec9& dydot)
    {
        auto dalpha = dy.template segment<3>(ALPHA);
        auto dbeta = dy.template segment<3>(BETA);
        QuatT gamma(y.data()+GAMMA);
        auto a = u.template segment<3>(ACC);
        auto w = u.template segment<3>(OMEGA);
        auto ba = b_.template segment<3>(ACC);
        auto bw = b_.template segment<3>(OMEGA);
        auto dgamma = dy.template segment<3>(GAMMA);

        auto eta_a = eta.template segment<3>(ACC);
        auto eta_w = eta.template segment<3>(OMEGA);

        dydot.template segment<3>(ALPHA) = dbeta;
        dydot.template segment<3>(BETA) = -gamma.rota(skew(a - ba)*dgamma) - gamma.rota(eta_a);
        dydot.template segment<3>(GAMMA) = -skew(w - bw)*dgamma - eta_w;
    }


    // ydot = f(y, u) <-- nonlinear dynamics (reference state)
    // A = d(dydot)/d(dy) <-- error state
    // B = d(dydot)/d(eta) <-- error state
    // Because of the error state, ydot != Ay+Bu
    void dynamics(const Vec10& y, const Vec6& u, Vec9& ydot, Mat9& A, Mat96&B)
    {
        auto alpha = y.template segment<3>(ALPHA);
        auto beta = y.template segment<3>(BETA);
        QuatT gamma(y.data()+GAMMA);
        auto a = u.template segment<3>(ACC);
        auto w = u.template segment<3>(OMEGA);
        auto ba = b_.template segment<3>(ACC);
        auto bw = b_.template segment<3>(OMEGA);

        ydot.template segment<3>(ALPHA) = beta;
        ydot.template segment<3>(BETA) = gamma.rota(a - ba);
        ydot.template segment<3>(GAMMA) = w - bw;

        A.setZero();
        A.template block<3,3>(ALPHA, BETA) = I_3x3;
        A.template block<3,3>(BETA, GAMMA) = -gamma.R().transpose() * skew(a - ba);
        A.template block<3,3>(GAMMA, GAMMA) = skew(bw-w);

        B.setZero();
        B.template block<3,3>(BETA, ACC) = -gamma.R().transpose();
        B.template block<3,3>(GAMMA, OMEGA) = -I_3x3;
    }


    static void boxplus(const Vec10& y, const Vec9& dy, Vec10& yp)
    {
        yp.template segment<3>(P) = y.template segment<3>(P) + dy.template segment<3>(P);
        yp.template segment<3>(V) = y.template segment<3>(V) + dy.template segment<3>(V);
        yp.template segment<4>(Q) = (QuatT(y.template segment<4>(Q)) + dy.template segment<3>(Q)).elements();
    }


    static void boxminus(const Vec10& y1, const Vec10& y2, Vec9& d)
    {
        d.template segment<3>(P) = y1.template segment<3>(P) - y2.template segment<3>(P);
        d.template segment<3>(V) = y1.template segment<3>(V) - y2.template segment<3>(V);
        d.template segment<3>(Q) = QuatT(y1.template segment<4>(Q)) - QuatT(y2.template segment<4>(Q));
    }


    void integrate(const Scalar& _t, const Vec6& u, const Mat6& cov)
    {
        n_updates_++;
        Scalar dt = _t - (t0_ + delta_t_);
        delta_t_ = _t - t0_;
        Vec9 ydot;
        Mat9 A;
        Mat96 B;
        Vec10 yp;
        dynamics(y_, u, ydot, A, B);
        boxplus(y_, ydot * dt, yp);
        y_ = yp;

        A = Mat9::Identity() + A*dt + 1/2.0 * A*A*dt*dt;
        B = B*dt;

        P_ = A*P_*A.transpose() + B*cov*B.transpose();
        J_ = A*J_ + B;

        NANO_IMU_ASSERT((P_.array() == P_.array()).all(), "NaN detected in covariance on propagation");
    }


    void estimateXj(const Scalar* _xi, const Scalar* _vi, Scalar* _xj, Scalar* _vj) const
    {
        auto alpha = y_.template segment<3>(ALPHA);
        auto beta = y_.template segment<3>(BETA);
        QuatT gamma(y_.data()+GAMMA);
        XformT xi(_xi);
        XformT xj(_xj);
        Map<const Vec3> vi(_vi);
        Map<Vec3> vj(_vj);

        xj.t_ = xi.t_ + xi.q_.rota(vi*delta_t_) + 1/2.0 * gravity_*delta_t_*delta_t_ + xi.q_.rotp(alpha);
        vj = gamma.rotp(vi + xi.q_.rotp(gravity_)*delta_t_ + beta);
        xj.q_ = xi.q_ * gamma;
    }


    void finished()
    {
      if (n_updates_ < 2)
      {
        P_ = P_ + Mat9::Identity() * 1e-10;
      }
      Xi_ = P_.inverse().llt().matrixL().transpose();
      NANO_IMU_ASSERT((Xi_.array() == Xi_.array()).all(), "NaN detected in IMU information matrix");
    }

    template<typename T>
    bool operator()(const T* _xi, const T* _xj, const T* _vi, const T* _vj, const T* _b, T* residuals) const
    {
        typedef Matrix<T,3,1> VecT3;
        typedef Matrix<T,6,1> VecT6;
        typedef Matrix<T,9,1> VecT9;
        typedef Matrix<T,10,1> VecT10;

        Xform<T> xi(_xi);
        Xform<T> xj(_xj);
        Map<const VecT3> vi(_vi);
        Map<const VecT3> vj(_vj);
        Map<const VecT6> b(_b);

        VecT9 dy = J_ * (b - b_);
        VecT10 y;
        y.template segment<6>(0) = y_.template segment<6>(0) + dy.template segment<6>(0);
        y.template segment<4>(6) = (QuatT(y_.template segment<4>(6)).template otimes<T,T>(Quat<T>::exp(dy.template segment<3>(6)))).elements();

        Map<VecT3> alpha(y.data()+ALPHA);
        Map<VecT3> beta(y.data()+BETA);
        Quat<T> gamma(y.data()+GAMMA);
        Map<VecT9> r(residuals);

        r.template block<3,1>(ALPHA, 0) = xi.q_.rotp(xj.t_ - xi.t_ - 1/2.0*gravity_*delta_t_*delta_t_) - vi*delta_t_ - alpha;
        r.template block<3,1>(BETA, 0) = gamma.rota(vj) - vi - xi.q_.rotp(gravity_)*delta_t_ - beta;
        r.template block<3,1>(GAMMA, 0) = (xi.q_.inverse() * xj.q_) - gamma;

        r = Xi_ * r;

        return true;
    }

    enum : int
    {
        ALPHA = 0,
        BETA = 3,
        GAMMA = 6,
    };

    enum :int
    {
        ACC = 0,
        OMEGA = 3
    };

    enum : int
    {
        P = 0,
        V = 3,
        Q = 6,
    };

    Scalar t0_;
    Scalar delta_t_;
    Vec6 b_;
    int n_updates_ = 0;

    Mat9 P_;
    Mat9 Xi_;
    Vec10 y_;

    Mat96 J_;
    Vec3 gravity_ = (Vec3() << 0, 0, 9.80665).finished();
};
