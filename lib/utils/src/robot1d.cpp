#include <vector>
#include <math.h>
#include <random>
#include <deque>

#include <Eigen/Core>

#include "utils/robot1d.h"

using namespace Eigen;

Robot1D::Robot1D(double _ba, double acc_var, double Td) :
    normal_(0.0, 1.0)
{
    x_ = 0;
    a_ = 0;
    v_ = 0;
    i_ = 0;
    t_ = 0;
    prev_x_ = NAN;
    kp_ = 0.3;
    kd_ = 0.03;
    max_F_ = 3.0;
    b_ = _ba;
    Td_ = Td;

    a_stdev_ = sqrt(acc_var);
    b_stdev_ = 0.0;

    xhat_ = x_;
    vhat_ = v_;
    ahat_ = a_;

    // Create a history
    history_t history {x_, v_, t_, xhat_, vhat_};
    hist_.push_back(history);
}

void Robot1D::add_waypoint(double wp)
{
    waypoints_.push_back(wp);
}

double Robot1D::pos_meas(const double& p_var)
{
    return x_ + normal_(gen_) * sqrt(p_var);
}

void Robot1D::step(double dt)
{
    if (std::abs(x_ - waypoints_[i_]) < 1e-2 && std::abs(v_) < 1e-1)
        i_ = (i_ + 1) % waypoints_.size();

    // propagate dynamics
    double e = waypoints_[i_] - x_;
    if (!std::isfinite(prev_x_))
        prev_x_ = x_;
    a_ = kp_*e - kd_ * (v_)/dt;
    a_ = a_ > max_F_ ? max_F_ : a_ < -max_F_ ? -max_F_ : a_;
    x_ += v_ * dt;
    v_ += a_ * dt;
    t_ += dt;
    b_ += normal_(gen_)*b_stdev_*dt;

    // propagate estimates
    ahat_ = a_ + normal_(gen_)*a_stdev_ - b_;
    xhat_ += vhat_ * dt;
    vhat_ += ahat_ * dt;

    // Save history
    history_t history {x_, v_, t_, xhat_, vhat_};
    hist_.push_back(history);
    while (t_ - hist_.front().t > Td_ && hist_.size() > 0)
    {
        hist_.pop_front();
    }
}

