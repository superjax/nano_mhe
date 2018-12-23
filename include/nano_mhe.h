#pragma once

#include <Eigen/Core>

#include "imu1d.h"
#include "pos1d.h"

using namespace Eigen;

#ifndef NDEBUG
#define NANO_MHE_ASSERT(condition, ...) \
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



// N is the number of nodes
template<typename T, int K>
class MHE_1D
{
static constexpr int NY = 2*(K-1)+K; // number of residuals in estimator
static constexpr int NX = (K*2)+1; // number of states in estimator
typedef Matrix<T, NX, 1> XVec;
typedef Matrix<T, NY, 1> ZVec;
typedef Matrix<T, NY, NX> JMat;

public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    typedef Matrix<T, 2, 1> nVec; // node
    MHE_1D()
    {
        k_ = -1;
        n_ = -1;
        i_ = -1;
        X_.setZero();
        acc_var_ = 0.3;
        pos_var_ = 0.1;

        // wire up the measurements to constants
        for (int k = 0; k < K; k++)
            pos_[k].init(&pos_var_);
        for (int i = 0; i < (K-1); i++)
        {
            imu_[i].init(X_.data()+2*K, &acc_var_);
            imu_ids_[i].first = imu_ids_[i].second = -1;
        }
    }

    void setBias(const T& bias)
    {
        X_(2*K) = bias;
    }

    const T& getBias() const
    {
        return X_(2*K);
    }

    const Map<const nVec> getState(int node) const
    {
        NANO_MHE_ASSERT(node <= n_, "Tried to access a node from the future");
        NANO_MHE_ASSERT(node > n_-K, "Tried to access a node beyond buffer");
        NANO_MHE_ASSERT(k_ >= 0 && n_ >= 0, "tried to get state before setting state");

        return Map<const nVec>(X_.data() + 2*n2k(node));
    }

    int addNode(const T& t)
    {
        addNode(t, imu_[i_].estimate_xj(X_.template segment<2>(2*k_)));
    }

    int addNode(const T& t, const nVec& xhat)
    {
        k_ = (k_ + 1) % K;

        if (i_ >= 0)
        {
            imu_ids_[i_].second = k_;
            imu_[i_].finished();
        }

        i_ = (i_ + 1) % (K-1);
        n_++;

        X_.template segment<2>(2*k_) = xhat;
        imu_[i_].reset(t);
        imu_ids_[i_].first = k_;
        imu_ids_[i_].second = -1;
        return n_;
    }

    int addPosMeas(int node, const T& y_pos)
    {
        NANO_MHE_ASSERT(node <= n_, "Tried acces a node from the future");
        NANO_MHE_ASSERT(node > n_-K, "Tried acces a node beyond buffer");
        NANO_MHE_ASSERT(std::isfinite(y_pos), "measurements must be finite");

        pos_[n2k(node)].set_meas(y_pos);
    }

    int addImuMeas(const T& t, const T& y_imu)
    {
        NANO_MHE_ASSERT(std::isfinite(y_imu), "measurements must be finite");

        imu_[i_].integrate(t, y_imu);
    }

    void optimize()
    {

    }

    void evalResiduals()
    {
        for (int i = 0; i < K-1; i++)
        {
            if (imu_ids_[i].second >= 0)
            {
                T* xi = X_.data() + 2*imu_ids_[i].first;
                T* xj = X_.data() + 2*imu_ids_[i].second;
                T* b = X_.data() + 2*K;
                T* res = Z_.data() + 2*i;
                imu_[i](xi, xj, b, res);
            }
            else
            {
                Z_.template segment<2>(2*i).setZero();
            }
        }

        for (int k = 0; k < K; k++)
        {
            T* xk = X_.data() + 2*k;
            T* res = Z_.data() + 2*(K-1)+k;
            pos_[k](xk, res);
        }
    }

    inline int n2k(int node) const
    {
        NANO_MHE_ASSERT(node <= n_, "Tried acces a node from the future");
        NANO_MHE_ASSERT(node >= 0, "Tried access an invalid node");
        NANO_MHE_ASSERT(node > n_-K, "Tried acces a node beyond buffer");
        return ((K + k_) - (n_-node)) % K;
    }

    XVec X_;
    ZVec Z_;
    JMat J_;
    Imu1D<T> imu_[K-1]; // IMU residuals (2 residuals each)
    std::pair<int, int> imu_ids_[K-1]; // from -> to pairs for imu residuals;
    Pos1D<T> pos_[K]; // Position residuals (1 residual each)
    int k_; // current internal state index [0-K]
    int n_; // current node id [0-inf]
    int i_; // current imu id [0-(K-1)]

    double acc_var_; // variance of acceleration
    double pos_var_; // variance of position


};
