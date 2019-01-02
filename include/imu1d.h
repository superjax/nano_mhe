#pragma once

#include <Eigen/Core>
#include <Eigen/LU>
#include <Eigen/Cholesky>


using namespace Eigen;

template <typename T>
class Imu1D
{
    typedef Matrix<T, 2, 2> Mat2;
    typedef Matrix<T, 2, 1> Vec2;
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    Imu1D()
    {
        t0_ = INFINITY;
        delta_t_ = 0;
        y_.setZero();
        P_.setZero();
        J_.setZero();
    }

    void init(const T& _b0, T* avar)
    {
        b_ = _b0;
        avar_ = avar;
    }

    const T& dp() const
    {
        return y_(0);
    }

    const T& dv() const
    {
        return y_(1);
    }

    void reset(T _t0)
    {
        t0_ = _t0;
        delta_t_ = 0;
        y_.setZero();
        P_.setZero();
        J_.setZero();
    }

    void integrate(T _t, T y)
    {
      assert(_t > t0_ + delta_t_);
      T dt = _t - (t0_ + delta_t_);
      delta_t_ = _t - t0_;

      // propagate covariance
      Mat2 A = (Mat2() << 1.0, dt, 0.0, 1.0).finished();
      Vec2 B {0.5*dt*dt, dt};
      Vec2 C {-0.5*dt*dt, dt};

      // Propagate state
      y_ = A*y_ + B*y + C*b_;

      P_ = A*P_*A.transpose() + B*(*avar_)*B.transpose();

      // propagate Jacobian dy/db
      J_ = A*J_ + C;
    }

    Vec2 estimate_xj(const Vec2& xi) const
    {
      assert(delta_t_ > 0);
      // Integrate starting at origin pose to get a measurement of the final pose
      Vec2 xj;
      xj(P) = xi(P) + xi(V)*delta_t_ + y_(ALPHA);
      xj(V) = xi(V) + y_(BETA);
      return xj;
    }

    void finished()
    {
      if (delta_t_ > 0)
        Omega_ = P_.inverse().llt().matrixL().transpose();
    }

    template<typename Scalar>
    bool operator()(const Scalar* _xi, const Scalar* _xj, const Scalar* _b, Scalar *residuals) const
    {
      typedef Matrix<Scalar, 2, 1> Vec2;
      Map<const Vec2> xi(_xi);
      Map<const Vec2> xj(_xj);
      Map<Vec2> r(residuals);

      // Use the jacobian to re-calculate y_ with change in bias
      Scalar db = *_b - b_;

      Vec2 y_db = y_ + J_ * db;


      r(P) = (xj(P) - xi(P) - xi(V)*delta_t_) - y_db(ALPHA);
      r(V) = (xj(V) - xi(V)) - y_db(BETA);
      Vec2 xid = xi;
      Vec2 xjd = xj;
      Vec2 rd = r;
      r = Omega_ * r;

      return true;
    }
private:

    enum {
      ALPHA = 0,
      BETA = 1,
    };
    enum {
      P = 0,
      V = 1
    };

    T t0_;
    T b_;
    T* avar_;

    T delta_t_;
    Mat2 P_;
    Mat2 Omega_;
    Vec2 y_;
    Vec2 J_;
};
