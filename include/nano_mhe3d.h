#pragma once

#include <Eigen/Core>

#include "imu3d.h"
#include "pos3d.h"
#include "vel3d.h"

#include "nano_lm.h"

using namespace Eigen;

#ifdef NDEBUG
#define NANO_MHE_ASSERT(...)
#else
#define NANO_MHE_ASSERT(cond, message) \
    assert(cond)
//    if(!(cond)) \
//    { \
//        throw std::runtime_error(message);\
//        assert(cond);\
//    }
#endif


// N is the number of nodes
template<typename Scalar, int K>
class MHE_3D_Base
{

public:
    enum
    {
        //     IMU     POS  VEL
        NY = 9*(K-1) + 3*K + 3, // number of residuals in estimator
        //   POS/ATT/VEL  BIAS
        NX =    K*10    +  6, // number of states in estimator
        NUM_NODES = K,
        NSZ = 10
    };
    typedef Matrix<Scalar, NX, 1> XVec;
    typedef Matrix<Scalar, NY, 1> ZVec;
    typedef Matrix<Scalar, NY, NX> JMat;
    typedef Matrix<Scalar, 3, 3> Mat3;

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    typedef Matrix<Scalar, NSZ, 1> nVec; // node
    typedef Matrix<Scalar, 6, 1> Vec6; // bias
    MHE_3D_Base()
    {
        vel_init_cov_ = Mat3::Identity() * 1e-6;
        vel_cov_ = Mat3::Identity() * 1e-3;
        reset();
    }

    void reset()
    {
        k_ = -1;
        n_ = -1;
        i_ = -1;
        vk_ = 0;
        X_.setZero();

        for (int i = 0; i < (K-1); i++)
        {
            imu_ids_[i].first = imu_ids_[i].second = -1;
        }
    }

    void setBias(const Vec6& bias)
    {
         X_.template segment<6>(NSZ*K) = bias;
    }

    inline const VectorBlock<const XVec, 6> getBias() const
    {
        return X_.template segment<6>(NSZ*K);
    }

    const Map<const nVec> getState(int node) const
    {
        NANO_MHE_ASSERT(node <= n_, "Tried access a node from the future");
        NANO_MHE_ASSERT(node > n_-K, "Tried access a node beyond buffer");
        NANO_MHE_ASSERT(k_ >= 0 && n_ >= 0, "tried to get state before setting state");

        return Map<const nVec>(X_.data() + NSZ*n2k(node));
    }

    int addNode(const Scalar& t)
    {
        return addNode(t, imu_[i_].estimateXj(X_.template segment<NSZ>(NSZ*k_)));
    }

    int addNode(const Scalar& t, const nVec& xhat)
    {
        k_ = (k_ + 1) % K;

        if (i_ >= 0)
        {
            imu_ids_[i_].second = k_;
            imu_[i_].finished();
        }
        i_ = (i_ + 1) % (K-1);
        n_++;

        // pin the velocity of the origin if we haven't filled the set of nodes yet
        if (n_ < K)
            vk_ = 0; // set it to the origin
        else
            vk_ = (k_ + 1) % K; // set it to the last node in the buffer (or the next one because the buffer is circular)


        X_.template segment<NSZ>(NSZ*k_) = xhat;
        imu_[i_].reset(t, getBias());
        imu_ids_[i_].first = k_;
        imu_ids_[i_].second = -1;
        pos_[k_].active_ = false;
        vel_.set_meas(X_.template segment<3>(NSZ*vk_+3), vk_ > 0 ? vel_cov_ : vel_init_cov_);
        return n_;
    }

    int addPosMeas(int node, const Scalar& y_pos, const Scalar& var)
    {
        NANO_MHE_ASSERT(node <= n_, "Tried acces a node from the future");
        NANO_MHE_ASSERT(node > n_-K, "Tried acces a node beyond buffer");
        NANO_MHE_ASSERT(std::isfinite(y_pos), "measurements must be finite");
        NANO_MHE_ASSERT(std::isfinite(var), "measurement variance must be finite");

        pos_[n2k(node)].set_meas(y_pos, var);
    }

    int addImuMeas(const Scalar& t, const Scalar& y_imu, const Scalar& var)
    {
        NANO_MHE_ASSERT(std::isfinite(y_imu), "measurements must be finite");
        NANO_MHE_ASSERT(std::isfinite(var), "measurement variance must be finite");

        imu_[i_].integrate(t, y_imu, var);
    }

    bool evalResiduals()
    {
        return f(Z_, X_);
    }

    template <typename OutType, typename InType>
    bool f(OutType& z, const InType& x) const
    {
        typedef typename OutType::Scalar T;
        for (int i = 0; i < K-1; i++)
        {
            if (imu_ids_[i].second >= 0)
            {
                const T* xi = x.data() + NSZ*imu_ids_[i].first;
                const T* xj = x.data() + NSZ*imu_ids_[i].second;
                const T* b = x.data() + NSZ*K;
                T* res = z.data() + 9*i;
                imu_[i](xi, xj, b, res);
            }
            else
            {
                z.template segment<9>(9*i).setZero();
            }
        }

        for (int k = 0; k < K; k++)
        {
            if (pos_[k].active_)
            {
                const T* xk = x.data() + NSZ*k;
                T* res = z.data() + 9*(K-1)+k;
                pos_[k](xk, res);
            }
            else
            {
                z(2*(K-1) + k) = 0.0;
            }
        }

        const T* vpin = x.data() + NSZ*vk_ + 1;
        T* vres = z.data() + 9*(K-1)+K;
        vel_(vpin, vres);
        return true;
    }

    inline int n2k(int node) const
    {
        NANO_MHE_ASSERT(node <= n_, "Tried acces a node from the future");
        NANO_MHE_ASSERT(node >= 0, "Tried access an invalid node");
        NANO_MHE_ASSERT(node > n_-K, "Tried access a node beyond buffer");
        return ((K + k_) - (n_-node)) % K;
    }

    inline const VectorBlock<const XVec, NSZ> x(int node) const
    {
        return X_.template segment<NSZ>(NSZ*n2k(node));
    }

    XVec X_;
    ZVec Z_;
    JMat J_;
    Imu3D<Scalar> imu_[K-1]; // IMU residuals (2 residuals each)
    std::pair<int, int> imu_ids_[K-1]; // from -> to pairs for imu residuals;
    Pos3D<Scalar> pos_[K]; // Position residuals (1 residual each)
    Vel3D<Scalar> vel_; // Velocity (pinning) constraint
    int vk_; // id of the velocity constraint
    int k_; // current internal state index [0-K]
    int n_; // current node id [0-inf]
    int i_; // current imu id [0-(K-1)]
    Mat3 vel_init_cov_; // covariance of velocity measurement at start
    Mat3 vel_cov_; // covariance of velocity measurement after start

    nano::levenbergMarquardtParameters<Scalar> params_;
};

template<typename Scalar, int K>
using MHE_3D_AD = CostFunctorAutoDiff<Scalar, MHE_3D_Base<Scalar, K>, MHE_3D_Base<Scalar, K>::NY, MHE_3D_Base<Scalar, K>::NX>;

template<typename Scalar, int K>
class MHE_3D_Optimizer : public MHE_3D_AD<Scalar, K>
{
public:
    MHE_3D_Optimizer() :
        lm_(this, &params_)
    {}

    void optimize()
    {
        lm_.minimize(this->X_);
    }
    nano::levenbergMarquardtParameters<Scalar> params_;
    nano::levenbergMarquardt<Scalar, MHE_3D_AD<Scalar, K>> lm_;
};

template<typename Scalar, int K>
using MHE_3D = MHE_3D_Optimizer<Scalar, K>;
