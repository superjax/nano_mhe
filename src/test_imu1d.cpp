#include <gtest/gtest.h>

#include "nano_mhe.h"
#include "utils/robot1d.h"

typedef MHE_1D<double, 5> MHE_1D5;

#define ASSERT_MAT_EQ(v1, v2) \
{ \
    ASSERT_EQ((v1).rows(), (v2).rows()); \
    ASSERT_EQ((v1).cols(), (v2).cols()); \
    for (int row = 0; row < (v1).rows(); row++) {\
    for (int col = 0; col < (v2).cols(); col++) {\
    ASSERT_FLOAT_EQ((v1)(row, col), (v2)(row,col));\
    }\
    }\
    }

TEST(Imu1D, init)
{
    double b = 0;
    double avar = 0.3;
    Imu1D<double> imu;
    imu.init(&b, &avar);
}

TEST(Imu1D, integrate)
{
    double b = 0;
    double avar = 0.3;
    Imu1D<double> imu;
    imu.init(&b, &avar);

    imu.reset(0);
    for (int i = 1; i < 11; i++)
    {
        imu.integrate(0.1*i, 1.0);
    }
    imu.finished();

    ASSERT_NEAR(imu.dp(), 0.5, 1e-3);
    ASSERT_NEAR(imu.dv(), 1.0, 1e-3);
}

TEST (nano_mhe, init)
{
    MHE_1D<double, 5> mhe;
}

TEST (nano_mhe, add_node)
{
    MHE_1D<double, 5> mhe;
    MHE_1D5::nVec x0{0, 0};
    mhe.addNode(0, x0);
    ASSERT_MAT_EQ(x0, mhe.getState(0));
}

TEST (nano_mhe, overfill_nodes)
{
    MHE_1D<double, 5> mhe;
    for (int i = 0; i < 8; i ++)
    {
        MHE_1D5::nVec x{i, 1};
        mhe.addNode(i, x);
    }

    mhe.setBias(6.0);

    for (int i = 0; i < 8; i++)
    {
        MHE_1D5::nVec x{i, 1};
        if (i < 3)
        {
//            ASSERT_THROW(mhe.getState(i), std::runtime_error);
        }
        else
        {
            ASSERT_MAT_EQ(x, mhe.getState(i));
        }
    }
    ASSERT_FLOAT_EQ(6.0, mhe.getBias());
}

TEST (nano_mhe, single_window_integration)
{
    double ba = 0.0;
    double Q = 0.0;
    Robot1D Robot(ba, Q);
    Robot.waypoints_ = {3, 0, 3, 0};

    double dt_window = 1.0;
    double dt = 0.01;

    MHE_1D5 mhe;
    int node = 0;
    mhe.addNode(Robot.t_, MHE_1D5::nVec{Robot.x_, Robot.v_});
    while (Robot.t_ <= dt_window)
    {
        Robot.step(dt);
        mhe.addImuMeas(Robot.t_, Robot.ahat_);
    }
    node = mhe.addNode(Robot.t_);
    ASSERT_NEAR(Robot.x_, mhe.getState(node)(0), 1e-2);
    ASSERT_NEAR(Robot.v_, mhe.getState(node)(1), 1e-2);
    ASSERT_NEAR(Robot.xhat_, mhe.getState(node)(0), 1e-2);
    ASSERT_NEAR(Robot.vhat_, mhe.getState(node)(1), 1e-2);
}

TEST(nano_mhe, no_residual_init)
{
    MHE_1D5 mhe;
    mhe.evalResiduals();
    for (int i = 0; i < mhe.Z_.rows(); i++)
    {
        ASSERT_EQ(mhe.Z_(i), 0.0);
    }
}

TEST(nano_mhe, incomplete_residual_init)
{
    double ba = 0.3;
    double Q = 1e-4;
    Robot1D Robot(ba, Q);
    Robot.waypoints_ = {3, 0, 3, 0};

    double dt_window = 1.0;
    double dt = 0.01;

    MHE_1D5 mhe;
    int node = mhe.addNode(Robot.t_, MHE_1D5::nVec{Robot.x_, Robot.v_});
    mhe.addPosMeas(node, Robot.pos_meas(sqrt(0.1)));
    while (Robot.t_ <= dt_window)
    {
        Robot.step(dt);
        mhe.addImuMeas(Robot.t_, Robot.ahat_);
    }
    node = mhe.addNode(Robot.t_);
    mhe.addPosMeas(node, Robot.pos_meas(sqrt(0.1)));

    mhe.evalResiduals();
    for (int i = 0; i < mhe.Z_.rows(); i++)
    {
        if (i == 8 || i == 9)
            ASSERT_NE(mhe.Z_(i), 0.0);
        else
            ASSERT_EQ(mhe.Z_(i), 0.0);
    }
}

TEST (nano_mhe, full_graph)
{
    double ba = 10.0;
    double Q = 1e-6;
    Robot1D Robot(ba, Q);
    Robot.waypoints_ = {3, 0, 3, 0};

    double dt_window = 1.0;
    double dt = 0.01;
    double t_max = 5* dt_window;

    MHE_1D5 mhe;
    int node = mhe.addNode(Robot.t_, MHE_1D5::nVec{Robot.xhat_, Robot.vhat_});
    mhe.addPosMeas(node, Robot.pos_meas(sqrt(0.1)));
    while (Robot.t_ <= t_max)
    {
        Robot.step(dt);
        mhe.addImuMeas(Robot.t_, Robot.ahat_);
        if (fabs(Robot.t_ - (node+1)*dt_window) < dt/2.0)
        {
            node = mhe.addNode(Robot.t_);
            mhe.addPosMeas(node, Robot.pos_meas(sqrt(0.1)));
        }
    }

    mhe.evalResiduals();
    for (int i = 0; i < mhe.Z_.rows(); i++)
    {
        if (i > 7)
            ASSERT_NE(mhe.Z_(i), 0.0);
        else
            ASSERT_NEAR(mhe.Z_(i), 0.0, 1e-8);
    }
}
