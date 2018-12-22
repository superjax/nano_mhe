#include <gtest/gtest.h>
#include <iostream>
#include <Eigen/Dense>

#include "nano_ad.h"
#include "geometry/quat.h"

#include "test_common.h"

using namespace quat;

TEST (CostFunctorAutoDiff, Compile)
{
    typedef Matrix<double, 2, 1> OT;
    typedef Matrix<double, 3, 1> IT1;
    typedef Matrix<double, 4, 1> IT2;
    typedef Matrix<double, 2, 3> JT1;
    typedef Matrix<double, 2, 4> JT2;
    CostFunctorAutoDiff<double, EmptyCostFunctor, 2, 3, 4> f;

    OT res;
    IT1 x;
    IT2 x2;
    JT1 j;
    JT2 j2;
    f.Evaluate(res.data(), x.data(), x2.data());
//    f.Evaluate(res, x, x2, j, j2);
}


TEST (Autodiff, SimpleTest)
{
    Eigen::AutoDiffScalar<Eigen::Vector2d> x, y, z;
    x = 8.0;
    x.derivatives() << 1,0;
    z = 2.0;
    z.derivatives() << 0,1;

    y = x*x + z;

    ASSERT_FLOAT_EQ(y.value(), 66.0);
    ASSERT_FLOAT_EQ(y.derivatives()(0), 2*8.0); // dydx = 2x
    ASSERT_FLOAT_EQ(y.derivatives()(1), 1.0); // dydz = 1
}

TEST (Autodiff, VectorTest)
{
    enum {
        NI = 2,
        NO = 2
    };
    typedef Eigen::Matrix<double, NI, 1> xVec;
    typedef Eigen::Matrix<double, NO, 1> yVec;
    typedef Eigen::AutoDiffScalar<xVec> ADScalar;
    typedef Eigen::Matrix<ADScalar, NI, 1> xAD;
    typedef Eigen::Matrix<ADScalar, NO, 1> yAD;

    yAD y;
    xAD x;
    x << 8.0, 2.0;
    x(0).derivatives() << 1,0;
    x(1).derivatives() << 0,1;

    y = x.squaredNorm() * Vector2d{1, 1};

    Matrix2d dydx_desired;
    dydx_desired << 16, 4, 16, 4;

    ASSERT_FLOAT_EQ(y(0).value(), 68);
    ASSERT_FLOAT_EQ(y(1).value(), 68);

    ASSERT_MAT_EQ(y(0).derivatives().transpose(), dydx_desired.row(0));
    ASSERT_MAT_EQ(y(1).derivatives().transpose(), dydx_desired.row(1));
}

TEST (Autodiff, MultiParameterVectorTest)
{
    enum {
        NI = 4,
        NO = 2,
        NX1 = 2,
        NX2 = 2
    };
    typedef Eigen::Matrix<double, NI, 1> inputVec;
    typedef Eigen::Matrix<double, NO, 1> yVec;
    typedef Eigen::AutoDiffScalar<inputVec> ADScalar;
    typedef Eigen::Matrix<ADScalar, NX1, 1> x1AD;
    typedef Eigen::Matrix<ADScalar, NX2, 1> x2AD;
    typedef Eigen::Matrix<ADScalar, NO, 1> yAD;

    yAD y;
    x1AD x1;
    x2AD x2;
    x1 << 8.0, 2.0;
    x2 << 3.0, 4.0;
    x1(0).derivatives() << 1, 0, 0, 0;
    x1(1).derivatives() << 0, 1, 0, 0;
    x2(0).derivatives() << 0, 0, 1, 0;
    x2(1).derivatives() << 0, 0, 0, 1;


    y = x1.cwiseProduct(x2);

    ASSERT_FLOAT_EQ(y(0).value(), 24.0);
    ASSERT_FLOAT_EQ(y(1).value(), 8.0);

    Matrix2d dydx1, dydx2;
    dydx1 << 3.0, 0, 0, 4.0;
    dydx2 << 8.0, 0, 0, 2.0;

    ASSERT_MAT_EQ(y(0).derivatives().topRows<2>().transpose(), dydx1.row(0));
    ASSERT_MAT_EQ(y(1).derivatives().topRows<2>().transpose(), dydx1.row(1));
    ASSERT_MAT_EQ(y(0).derivatives().bottomRows<2>().transpose(), dydx2.row(0));
    ASSERT_MAT_EQ(y(1).derivatives().bottomRows<2>().transpose(), dydx2.row(1));
}

struct SimpleFunctor
{
    template<typename Derived1, typename Derived2, typename Derived3>
    bool operator()(Derived1& y, const Derived2& x1, const Derived3& x2) const
    {
        y = x1.cwiseProduct(x2);
        return true;
    }
};

//TEST (Autodiff, EvaluateFunctor)
//{
//    CostFunctorAutoDiff<SimpleFunctor, Vector2d, Vector2d, Vector2d> f;

//    Vector2d x1{8, 2};
//    Vector2d x2{3, 4};
//    Vector2d y;

//    f.Evaluate(y, x1, x2);

//    Vector2d y_des;
//    y_des << 24, 8;
//    ASSERT_MAT_EQ(y_des, y);
//}

//TEST (Autodiff, AutoDiffFunctor)
//{
//    CostFunctorAutoDiff<SimpleFunctor, Vector2d, Vector2d, Vector2d> f;

//    Vector2d x1{8, 2};
//    Vector2d x2{3, 4};
//    Matrix2d dfdx1, dfdx2;
//    Vector2d y;

//    f.Evaluate(y, x1, x2, dfdx1, dfdx2);

//    Vector2d y_des;
//    y_des << 24, 8;
//    ASSERT_MAT_EQ(y_des, y);

//    Matrix2d dydx1_des, dydx2_des;
//    dydx1_des << 3.0, 0, 0, 4.0;
//    dydx2_des << 8.0, 0, 0, 2.0;

//    ASSERT_MAT_EQ(dfdx1, dydx1_des);
//    ASSERT_MAT_EQ(dfdx2, dydx2_des);
//}

struct QuatPlus
{
    template<typename Derived1, typename Derived2, typename Derived3>
    bool operator()(Derived1& qp, const Derived2 q, const Derived3 delta) const
    {
        qp = q + delta;
        return true;
    }
};
//typedef CostFunctorAutoDiff<QuatPlus, Vector4d, Vector4d, Vector3d> QuatParam;

//TEST (Autodiff, Quaternion)
//{
//    Quatd q1 = Quatd::from_euler(10.0*M_PI/180.0, -45.0*M_PI/180.0, 20.0*M_PI/180.0);
//    Vector3d delta = (Vector3d() << 0.1, 0.2, 0.3).finished();
//    Quatd qp;
//    Matrix<double, 4, 4> dqp_dq1;
//    Matrix<double, 4, 3> dqp_ddelta;

//    QuatParam qf;
//    qf.Evaluate(qp.arr_, q1.arr_, delta, dqp_dq1, dqp_ddelta);


//}

