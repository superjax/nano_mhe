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

TEST(Imu3D, CheckErrorStateDynamics)
{
    typedef Imu3D<double> IMU;
    IMU y;
    IMU yhat;
    Vector9d dy;
    double t = 0;
    const double Tmax = 10.0;
    static const double dt = 0.001;

    Vector6d bias;
    bias.setZero();
    y.reset(t, bias);
    yhat.reset(t, bias);
    IMU::boxplus(y.y_, Vector9d::Constant(0.01), yhat.y_);
    IMU::boxminus(y.y_, yhat.y_, dy);

    Vector10d y_check;
    IMU::boxplus(yhat.y_, dy, y_check);
    ASSERT_MAT_NEAR(y.y_, y_check, 1e-8);

    Vector6d u, eta;
    u.setZero();
    eta.setZero();

    std::default_random_engine gen;
    std::normal_distribution<double> normal;

    Logger<double> log("../logs/Imu3d.CheckDynamics.log");


    Matrix6d cov = Matrix6d::Identity() * 1e-3;
    Vector9d dydot;
    log.log(t);
    log.logVectors(y.y_, yhat.y_, dy, y_check, u);
    for (int i = 0; i < Tmax/dt; i++)
    {
        u += dt * normalRandomVector<Vector6d>(normal, gen);
        t += dt;
        y.errorStateDynamics(y.y_, dy, u, eta, dydot);
        dy += dydot * dt;

        y.integrate(t, u, cov);
        yhat.integrate(t, u, cov);
        IMU::boxplus(yhat.y_, dy, y_check);
        log.log(t);
        log.logVectors(y.y_, yhat.y_, dy, y_check, u);
        ASSERT_MAT_NEAR(y.y_, y_check, t > 0.3 ? 5e-6*t*t : 2e-7);
    }
}


TEST(Imu3D, CheckDynamicsJacobians)
{
    Matrix6d cov = Matrix6d::Identity()*1e-3;

    Vector6d b0;
    Vector10d y0;
    Vector6d u0;
    Vector6d eta0;
    Vector9d ydot;
    Vector9d dy0;

    Matrix9d A;
    Eigen::Matrix<double, 9, 6> B;

    for (int i = 0; i < 100; i++)
    {
        b0.setRandom();
        y0.setRandom();
        y0.segment<4>(6) = Quatd::Random().elements();
        u0.setRandom();

        eta0.setZero();
        dy0.setZero();

        Imu3D<double> f;
        f.reset(0, b0);
        f.dynamics(y0, u0, ydot, A, B);
        Vector9d dy0;

        auto yfun = [&y0, &cov, &b0, &u0, &eta0](const Vector9d& dy)
        {
            Imu3D<double> f;
            f.reset(0, b0);
            Vector9d dydot;
            f.errorStateDynamics(y0, dy, u0, eta0, dydot);
            return dydot;
        };
        auto etafun = [&y0, &cov, &b0, &dy0, &u0](const Vector6d& eta)
        {
            Imu3D<double> f;
            f.reset(0, b0);
            Vector9d dydot;
            f.errorStateDynamics(y0, dy0, u0, eta, dydot);
            return dydot;
        };

        Matrix9d AFD = calc_jac(yfun, dy0);
        Eigen::Matrix<double, 9, 6> BFD = calc_jac(etafun, u0);

        cout << "A\n" << A << "\nAFD\n" << AFD << "\n\n";
        cout << "B\n" << B << "\nBFD\n" << BFD << "\n\n";

        ASSERT_MAT_NEAR(AFD, A, 1e-7);
        ASSERT_MAT_NEAR(BFD, B, 1e-7);
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
    JFD = calc_jac(fun, b0, nullptr, nullptr, boxminus, 1e-5);
//    std::cout << "FD:\n" << JFD << std::endl;
//    std::cout << "A:\n" << J << std::endl;
    ASSERT_MAT_NEAR(J, JFD, 1e-2);
}


