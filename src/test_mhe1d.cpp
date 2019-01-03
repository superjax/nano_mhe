#include <fstream>
#include <gtest/gtest.h>

#include "test_common.h"
#include "nano_mhe1d.h"
#include "utils/robot1d.h"
#include "utils/logger.h"


TEST (nano_mhe_1d, init)
{
    MHE_1D<double, 5> mhe;
}

TEST (nano_mhe_1d, add_node)
{
    MHE_1D<double, 5> mhe;
    MHE_1D<double, 5>::nVec x0{0, 0};
    mhe.addNode(0, x0);
    ASSERT_MAT_EQ(x0, mhe.getState(0));
}

TEST (nano_mhe_1d, overfill_nodes)
{
    MHE_1D<double, 5> mhe;
    for (int i = 0; i < 8; i ++)
    {
        MHE_1D<double, 5>::nVec x{i, 1};
        mhe.addNode(i, x);
        if (i < 5)
        {
            ASSERT_EQ(mhe.vk_, 0);
        }
        else
        {
            ASSERT_EQ(mhe.vk_, (i-4)%5);
        }
    }

    mhe.setBias(6.0);

    for (int i = 7; i >= 0; i--)
    {
        MHE_1D<double, 5>::nVec x{i, 1};
        if (i >= 3)
        {
            ASSERT_MAT_EQ(x, mhe.getState(i));
        }
        else
        {
#ifndef NDEBUG
            ASSERT_DEATH(mhe.getState(i), "(Assertion `node > n_-K' failed)");
#endif
        }
    }
    ASSERT_FLOAT_EQ(6.0, mhe.getBias());
}

TEST (nano_mhe_1d, single_window_integration)
{
    double ba = 0.0;
    double Q = 0.0;
    Robot1D Robot(ba, Q);
    Robot.waypoints_ = {3, 0, 3, 0};

    double dt_window = 1.0;
    double dt = 0.01;

    MHE_1D<double, 5> mhe;
    int node = 0;
    mhe.addNode(Robot.t_, MHE_1D<double, 5>::nVec{Robot.x_, Robot.v_});
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

TEST(nano_mhe_1d, no_residual_init)
{
    MHE_1D<double, 5> mhe;
    mhe.evalResiduals();
    for (int i = 0; i < mhe.Z_.rows(); i++)
    {
        ASSERT_EQ(mhe.Z_(i), 0.0);
    }
}

TEST(nano_mhe_1d, incomplete_residual_init)
{
    double ba = 0.3;
    double Q = 1e-4;
    Robot1D Robot(ba, Q);
    Robot.waypoints_ = {3, 0, 3, 0};

    double dt_window = 1.0;
    double dt = 0.01;

    MHE_1D<double, 5> mhe;
    int node = mhe.addNode(Robot.t_, MHE_1D<double, 5>::nVec{Robot.x_, Robot.v_});
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

TEST(nano_mhe_1d, incomplete_residual_optimize)
{
    double ba = 0.1;
    double acc_var = 1e-3;
    double pos_var = 1e-3;
    double vel_var = 1e-5;
    Robot1D Robot(ba, acc_var);
    Robot.waypoints_ = {3, 0, 3, 0};

    double dt_window = 1.0;
    double dt = 0.01;

    MHE_1D<double, 5> mhe;
    mhe.init(acc_var, pos_var, vel_var);
    int node = mhe.addNode(Robot.t_, MHE_1D<double, 5>::nVec{Robot.x_, Robot.v_});
    mhe.addPosMeas(node, Robot.pos_meas(pos_var));
    while (Robot.t_ <= dt_window)
    {
        Robot.step(dt);
        mhe.addImuMeas(Robot.t_, Robot.ahat_);
    }
    node = mhe.addNode(Robot.t_);
    mhe.addPosMeas(node, Robot.pos_meas(pos_var));
    mhe.optimize();
    mhe.evalResiduals();

    for (int i = 0; i < mhe.Z_.rows(); i++)
    {
        ASSERT_NEAR(mhe.Z_(i), 0.0, 1e-8);
    }
}

TEST (nano_mhe_1d, full_graph)
{
    double ba = 10.0;
    double Q = 1e-6;
    Robot1D Robot(ba, Q);
    Robot.waypoints_ = {3, 0, 3, 0};

    double dt_window = 1.0;
    double dt = 0.01;
    double t_max = 5* dt_window;

    MHE_1D<double, 5> mhe;
    int node = mhe.addNode(Robot.t_, MHE_1D<double, 5>::nVec{Robot.xhat_, Robot.vhat_});
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
        if (i > 7 && i < 13)
            ASSERT_NE(mhe.Z_(i), 0.0);
        else
            ASSERT_NEAR(mhe.Z_(i), 0.0, 1e-8);
    }
}

TEST (nano_mhe_1d, Optimize)
{
    double ba = 0.1;
    double acc_var = 1e-8;
    double pos_var = 1e-3;
    double vel_var = 1e-5;
    Robot1D Robot(ba, acc_var);
    Robot.waypoints_ = {1.0, 0, 1.0, 0.3, 1.5, 1.2, 0};

    double dt_window = 1.0;
    double dt = 0.01;
    double t_max = 300*dt_window;

    MHE_1D<double, 25> mhe;
    Logger<double> hist("../logs/nano_mhe.Optimize.log");
    Logger<double> path_hist("../logs/nano_mhe.Path.log");
    int node = mhe.addNode(Robot.t_, Vector2d{Robot.xhat_, Robot.vhat_});
    double xbar = Robot.pos_meas(pos_var);
    mhe.addPosMeas(node, xbar, pos_var);
    hist.log(Robot.t_, Robot.x_, Robot.v_, Robot.xhat_, Robot.vhat_, 
    	     mhe.x(node)(0), mhe.x(node)(1), Robot.b_, mhe.getBias(), xbar);
    while (Robot.t_ <= t_max)
    {
        Robot.step(dt);
        mhe.addImuMeas(Robot.t_, Robot.ahat_, acc_var);
        if (fabs(Robot.t_ - (node+1)*dt_window) < dt/2.0)
        {
            node = mhe.addNode(Robot.t_);
            double xbar = Robot.pos_meas(pos_var);
            mhe.addPosMeas(node, xbar, pos_var);
            mhe.optimize();

            // Log everything to the history for plotting
            hist.log(Robot.t_, Robot.x_, Robot.v_, Robot.xhat_, Robot.vhat_, 
            	     mhe.x(node)(0), mhe.x(node)(1), Robot.b_, mhe.getBias(), xbar);
            for (int i = 0; i < mhe.NUM_NODES; i++)
            {
                int ni = node - i;
                if (ni < 0)
                {
                    path_hist.log(NAN, NAN, NAN);
                }
                else
                {
                    path_hist.log(Robot.t_ - i*dt_window, mhe.x(node - i)(0), mhe.x(node - i)(1));
                }
            }
            path_hist.log(NAN, NAN, NAN); // so python will draw a new line for each horizon
        }
    }
    // The python file python/nano_mhe_optimize.py will plot the results
}
