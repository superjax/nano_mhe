#include <gtest/gtest.h>

#include "test_common.h"
#include "utils/jac.h"
#include "utils/logger.h"
#include "multirotor_sim/simulator.h"
#include "multirotor_sim/dynamics.h"
#include "imu3d.h"
#include "geometry/support.h"

TEST(Imu3D, compile)
{
    Imu3D<double> imu;
}


TEST(Imu3D, reset)
{
    Imu3D<double> imu;
    Vector6d b0;
    b0 << 0, 1, 2, 3, 4, 5;
    imu.reset(0, b0);
}


Vector10d boxplus(const Vector10d& y, const Vector9d& dy)
{
    Vector10d yp;
    yp.block<3,1>(0,0) = y.block<3,1>(0,0) + dy.block<3,1>(0,0);
    yp.block<3,1>(3,0) = y.block<3,1>(3,0) + dy.block<3,1>(3,0);
    yp.block<4,1>(6,0) = (Quatd(y.block<4,1>(6,0)) + dy.block<3,1>(6,0)).elements();
    return yp;
}

Vector9d boxminus(const Vector10d& y1, const Vector10d& y2)
{
    Vector9d out;
    out.block<3,1>(0,0) = y1.block<3,1>(0,0) - y2.block<3,1>(0,0);
    out.block<3,1>(3,0) = y1.block<3,1>(3,0) - y2.block<3,1>(3,0);
    out.block<3,1>(6,0) = Quatd(y1.block<4,1>(6,0)) - Quatd(y2.block<4,1>(6,0));
    return out;
}

TEST(Imu3D, CheckDynamics)
{
    typedef Imu3D<double> IMU;
    IMU y;
    Vector9d dy;
    double t = 0;

    Vector6d bias;
    bias.setZero();
    y.reset(t, bias);
    yhat.reset(t, bias);
    IMU::boxplus(y.y_, Vector9d::Constant(0.001), yhat.y_);
    IMU::boxminus(y.y_, yhat.y_, dy);

    Vector10d y_check;
    IMU::boxplus(yhat.y_, dy, y_check);
    ASSERT_MAT_NEAR(y.y_, y_check, 1e-8);

    Vector6d u;
    u.setZero();

    std::default_random_engine gen;
    std::normal_distribution<double> normal;

    Logger<double> log("../logs/Imu3d.CheckDynamics.log");

    static const double dt = 0.001;
    Matrix6d cov = Matrix6d::Identity() * 1e-3;
    Vector9d dydot;
    Matrix<double, 9, 9> A;
    Matrix<double, 9, 6> B, C;
    log.log(t);
    log.logVectors(y.y_, yhat.y_, dy, y_check, u);
    for (int i = 0; i < 1.0/dt; i++)
    {
        u += dt * normalRandomVector<Vector6d>(normal, gen);
        u.segment<3>(IMU::OMEGA).setZero();
        t += dt;
        y.dynamics(y.y_, u, dydot, A, B, C);
        dy += dydot * dt;

        y.integrate(t, u, cov);
        yhat.integrate(t, u, cov);
        IMU::boxplus(yhat.y_, dy, y_check);
        log.log(t);
        log.logVectors(y.y_, yhat.y_, dy, y_check, u);
    }
}


TEST(Imu3D, CheckDynamicsJacobians)
{
    Matrix6d cov = Matrix6d::Identity()*1e-3;

    Vector6d b0;
    Vector10d y0;
    Vector6d u0;
    Vector9d ydot;

    Matrix9d A;
    Eigen::Matrix<double, 9, 6> B;
    Eigen::Matrix<double, 9, 6> C;

    for (int i = 0; i < 100; i++)
    {
        b0.setRandom();
        y0.setRandom();
        y0.segment<4>(6) = Quatd::Random().elements();
        u0.setRandom();
        Imu3D<double> f;
        f.reset(0, b0);
        f.dynamics(y0, u0, ydot, A, B, C);

        auto yfun = [&cov, &b0, &u0](const Vector10d& y)
        {
            Imu3D<double> f;
            f.reset(0, b0);
            Vector9d ydot;
            Matrix9d A;
            Eigen::Matrix<double, 9, 6> B;
            Eigen::Matrix<double, 9, 6> C;
            f.dynamics(y, u0, ydot, A, B, C);
            return ydot;
        };
        auto bfun = [&cov, &y0, &u0](const Vector6d& b)
        {
            Imu3D<double> f;
            f.reset(0, b);
            Vector9d ydot;
            Matrix9d A;
            Eigen::Matrix<double, 9, 6> B;
            Eigen::Matrix<double, 9, 6> C;
            f.dynamics(y0, u0, ydot, A, B, C);
            return ydot;
        };
        auto ufun = [&cov, &b0, &y0](const Vector6d& u)
        {
            Imu3D<double> f;
            f.reset(0, b0);
            Vector9d ydot;
            Matrix9d A;
            Eigen::Matrix<double, 9, 6> B;
            Eigen::Matrix<double, 9, 6> C;
            f.dynamics(y0, u, ydot, A, B, C);
            return ydot;
        };

        Matrix9d AFD = calc_jac(yfun, y0, boxminus, boxplus);
        Eigen::Matrix<double, 9, 6> BFD = calc_jac(ufun, u0);
        Eigen::Matrix<double, 9, 6> CFD = calc_jac(bfun, b0);

        ASSERT_MAT_NEAR(AFD, A, 1e-7);
        ASSERT_MAT_NEAR(BFD, B, 1e-7);
        ASSERT_MAT_NEAR(CFD, C, 1e-7);
    }
}

TEST(Imu3D, CheckBiasJacobians)
{
    Simulator multirotor(false);
    multirotor.load("../lib/multirotor_sim/params/sim_params.yaml");
    std::vector<Vector6d,Eigen::aligned_allocator<Vector6d>> meas;
    std::vector<double> t;

    while (multirotor.t_ < 1.0)
    {
        multirotor.run();
        meas.push_back(multirotor.get_imu_prev());
        t.push_back(multirotor.t_);
    }

    Matrix6d cov = Matrix6d::Identity()*1e-3;
    Vector6d b0;
    Eigen::Matrix<double, 9, 6> J, JFD;

    b0.setZero();
    Imu3D<double> f;
    f.reset(0, b0);
    Vector10d y0 = f.y_;
    for (int i = 0; i < meas.size(); i++)
    {
        f.integrate(t[i], meas[i], cov);
    }
    J = f.J_;

    auto fun = [&cov, &meas, &t, &y0](const Vector6d& b0)
    {
        Imu3D<double> f;
        f.reset(0, b0);
        for (int i = 0; i < meas.size(); i++)
        {
            f.integrate(t[i], meas[i], cov);
        }
        return f.y_;
    };
    JFD = calc_jac(fun, b0, nullptr, nullptr, boxminus, nullptr, 1e-5);
//    std::cout << "FD:\n" << JFD << std::endl;
//    std::cout << "A:\n" << J << std::endl;
    ASSERT_MAT_NEAR(J, JFD, 1e-2);
}


