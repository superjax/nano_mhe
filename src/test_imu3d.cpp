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

TEST(Imu3D, CheckPropagation)
{
    Simulator multirotor(false);
    multirotor.load("../lib/multirotor_sim/params/sim_params.yaml");


    typedef Imu3D<double> IMU;
    IMU imu;
    Vector6d b0;
    b0.setZero();
    Matrix6d cov = Matrix6d::Identity() * 1e-3;

    multirotor.run();
    imu.reset(multirotor.t_, b0);
    Xformd x0 = multirotor.get_pose();
    Vector3d v0 = multirotor.get_vel();

    Logger<double> log("../logs/Imu3D.CheckPropagation.log");

    Xformd xhat = multirotor.get_pose();
    Vector3d vhat = multirotor.get_vel();
    log.log(multirotor.t_);
    log.logVectors(xhat.elements(), vhat, multirotor.get_pose().elements(), multirotor.get_vel(), multirotor.get_true_imu());

    double next_reset = 1.0;
    multirotor.tmax_ = 10.0;
    while (multirotor.run())
    {
        imu.integrate(multirotor.t_, multirotor.get_true_imu(), cov);

        if (std::abs(multirotor.t_ - next_reset) <= multirotor.dt_ /2.0)
        {
            imu.reset(multirotor.t_, b0);
            x0 = multirotor.get_pose();
            v0 = multirotor.get_vel();
            next_reset += 1.0;
        }

        imu.estimateXj(x0.data(), v0.data(), xhat.data(), vhat.data());
        log.log(multirotor.t_);
        log.logVectors(xhat.elements(), vhat, multirotor.get_pose().elements(), multirotor.get_vel(), multirotor.get_true_imu());
        ASSERT_MAT_NEAR(xhat.t(), multirotor.get_pose().t(), 0.007);
        ASSERT_QUAT_NEAR(xhat.q(), multirotor.get_pose().q(), 0.0016);
        ASSERT_MAT_NEAR(vhat, multirotor.get_vel(), 0.011);
    }
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

//        cout << "A\n" << A << "\nAFD\n" << AFD << "\n\n";
//        cout << "B\n" << B << "\nBFD\n" << BFD << "\n\n";

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
    multirotor.dt_ = 0.001;

    while (multirotor.t_ < 0.1)
    {
        multirotor.run();
        meas.push_back(multirotor.get_true_imu());
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
    auto bm = [](const MatrixXd& x1, const MatrixXd& x2)
    {
        Matrix<double, 9, 1> dx;
        Imu3D<double>::boxminus(x1, x2, dx);
        return dx;
    };

    JFD = calc_jac(fun, b0, nullptr, nullptr, bm, 1e-5);
//    std::cout << "FD:\n" << JFD << "\nA:\n" << J <<std::endl;
    ASSERT_MAT_NEAR(J, JFD, 1e-4);
}


