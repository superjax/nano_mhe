#include <gtest/gtest.h>

#include <Eigen/Core>
#include <unsupported/Eigen/LevenbergMarquardt>

#include "test_common.h"
#include "nano_ad.h"
#include "nano_lm.h"

using namespace Eigen;

struct lmAnalytical : DenseFunctor<double>
{
    static constexpr int NX = DenseFunctor::InputsAtCompileTime;
    lmAnalytical(void): DenseFunctor<double>(3,15) {}
    int operator()(const VectorXd &x, VectorXd &fvec) const
    {
        double tmp1, tmp2, tmp3;
        static const double y[15] = {1.4e-1, 1.8e-1, 2.2e-1, 2.5e-1, 2.9e-1, 3.2e-1, 3.5e-1,
            3.9e-1, 3.7e-1, 5.8e-1, 7.3e-1, 9.6e-1, 1.34, 2.1, 4.39};

        for (int i = 0; i < values(); i++)
        {
            tmp1 = i+1;
            tmp2 = 16 - i - 1;
            tmp3 = (i>=8)? tmp2 : tmp1;
            fvec[i] = y[i] - (x[0] + tmp1/(x[1]*tmp2 + x[2]*tmp3));
        }
        return 0;
    }

    int df(const VectorXd &x, MatrixXd &fjac) const
    {
        double tmp1, tmp2, tmp3, tmp4;
        for (int i = 0; i < values(); i++)
        {
            tmp1 = i+1;
            tmp2 = 16 - i - 1;
            tmp3 = (i>=8)? tmp2 : tmp1;
            tmp4 = (x[1]*tmp2 + x[2]*tmp3); tmp4 = tmp4*tmp4;
            fjac(i,0) = -1;
            fjac(i,1) = tmp1*tmp2/tmp4;
            fjac(i,2) = tmp1*tmp3/tmp4;
        }
        return 0;
    }
};

TEST (EigenLM, AnalyticalDerivatives)
{
    const int m=15, n=3;
    int info;
    double fnorm, covfac;
    VectorXd x;

    /* the following starting values provide a rough fit. */
    x.setConstant(n, 1.);

    // do the computation
    lmAnalytical functor;
    LevenbergMarquardt<lmAnalytical> lm(functor);
    info = lm.minimize(x);

    // check return values
    ASSERT_EQ(info, 1);
    ASSERT_EQ(lm.nfev(), 6);
    ASSERT_EQ(lm.njev(), 5);

    // check norm
    fnorm = lm.fvec().blueNorm();
    ASSERT_NEAR(fnorm, 0.09063596, 1e-3);

    // check x
    VectorXd x_ref(n);
    x_ref << 0.08241058, 1.133037, 2.343695;
    ASSERT_MAT_NEAR(x, x_ref, 1e-3);

    MatrixXd cov_ref(n,n);
    cov_ref <<
        0.0001531202,   0.002869941,  -0.002656662,
        0.002869941,    0.09480935,   -0.09098995,
        -0.002656662,   -0.09098995,    0.08778727;

    MatrixXd cov;
    cov =  covfac*lm.matrixR().topLeftCorner<n,n>();
}

struct lmFunc
{
public:
    template<typename D1, typename D2>
    bool f(D1 &fvec, const D2 &x) const
    {
        double tmp1, tmp2, tmp3;
        static const double y[15] = {1.4e-1, 1.8e-1, 2.2e-1, 2.5e-1, 2.9e-1, 3.2e-1, 3.5e-1,
            3.9e-1, 3.7e-1, 5.8e-1, 7.3e-1, 9.6e-1, 1.34, 2.1, 4.39};

        for (int i = 0; i < 15; i++)
        {
            tmp1 = i+1;
            tmp2 = 16 - i - 1;
            tmp3 = (i>=8)? tmp2 : tmp1;
            fvec[i] = y[i] - (x[0] + tmp1/(x[1]*tmp2 + x[2]*tmp3));
        }
        return true;
    }
};
typedef CostFunctorAutoDiff<double, lmFunc, 15, 3> lmFuncAD;


struct lmAutoDiff : public DenseFunctor<double>
{
public:
    typedef Matrix<double, 3, 1> xVec;
    typedef Matrix<double, 15, 1> yVec;
    typedef Matrix<double, 15, 3> jMat;
    lmAutoDiff(void): DenseFunctor<double>(3,15) {}

    int operator()(const VectorXd &_x, VectorXd &_y) const
    {
        xVec x = _x;
        yVec y;
        if (f_.Evaluate(y, x))
        {
            _y = y;
            return 0;
        }
        else
        {
            return 1;
        }
    }

    int df(const VectorXd &_x, MatrixXd &_j) const
    {
        yVec y;
        xVec x = _x;
        jMat j;
        if (f_.Evaluate(y, x, j))
        {
            _j = j;
            return 0;
        }
        else
        {
            return 1;
        }
    }
    CostFunctorAutoDiff<double, lmFunc, 15, 3> f_;
};

TEST (EigenLM, AutoDiffEquivalent)
{
    const int m=15, n=3;
    int info;
    double fnorm, covfac;
    VectorXd x;
    /* the following starting values provide a rough fit. */
    x.setConstant(n, 1.);

    // Create the Functors
    lmAutoDiff ad_functor;
    lmAnalytical an_functor;

    // Check that the analytical and auto diff functors are equivalent
    VectorXd y_A, y_AD;
    y_A.resize(m);
    y_AD.resize(m);
    ad_functor(x, y_AD);
    an_functor(x, y_A);

    ASSERT_MAT_EQ(y_A, y_AD);

    MatrixXd J_A;
    MatrixXd J_AD;
    J_A.resize(m, n);
    J_AD.resize(m, n);
    ad_functor.df(x, J_AD);
    an_functor.df(x, J_A);

    VectorXd dx;
    dx.setRandom(n);
    dx*=0.001;

    VectorXd yp;
    ad_functor(x+dx, yp);
    VectorXd yp_AD_approx = y_AD + J_AD*dx;
    VectorXd yp_AN_approx = y_A + J_A*dx;
    ASSERT_MAT_NEAR(yp_AN_approx, yp, 1e-1);
    ASSERT_MAT_NEAR(yp_AD_approx, yp, 1e-1);

    ASSERT_MAT_EQ(J_A, J_AD);
}

TEST (EigenLM, AutoDiff)
{
    const int m=15, n=3;
    int info;
    double fnorm;
    VectorXd x;

    /* the following starting values provide a rough fit. */

    // Create the Functors
    lmAutoDiff ad_functor;
    LevenbergMarquardt<lmAutoDiff> lm(ad_functor);
    for (int i = 0; i < 1000000; i++)
    {
        x.setConstant(n, 1.);
        info = lm.minimize(x);
    }
    // check return values
    ASSERT_EQ(info, 1);
    ASSERT_EQ(lm.nfev(), 6);
    ASSERT_EQ(lm.njev(), 5);

    // check norm
    fnorm = lm.fvec().blueNorm();
    ASSERT_NEAR(fnorm, 0.09063596, 1e-3);

    // check x
    VectorXd x_ref(n);
    x_ref << 0.08241058, 1.133037, 2.343695;
    ASSERT_MAT_NEAR(x, x_ref, 1e-6);
}

TEST (nanoLM, Minimize)
{
    lmFuncAD functor;
    nano::levenbergMarquardtParameters<double> params;
    nano::levenbergMarquardt<double, lmFuncAD> lm(&functor, &params);

    Matrix<double, 3, 1> x;
    Matrix<double, 15, 1> y;

    int info;
    for (int i = 0; i < 1000000; i++)
    {
        x.setOnes();
        info = lm.minimize(x);
    }

    ASSERT_EQ(info, 1);
    ASSERT_EQ(lm.nfev_, 6);
    ASSERT_EQ(lm.njev_, 5);

    Matrix<double, 3, 1> x_ref;
    x_ref << 0.08241058, 1.133037, 2.343695;
    ASSERT_MAT_NEAR(x, x_ref, 1e-6);
}
