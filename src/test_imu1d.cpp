#include <fstream>
#include <gtest/gtest.h>

#include "test_common.h"
#include "factors_1d/imu1d.h"
#include "utils/robot1d.h"

TEST(Imu1D, compile)
{
    Imu1D<double> imu;
}

TEST(Imu1D, integrate)
{
    double b = 0;
    double avar = 0.3;
    Imu1D<double> imu;

    imu.reset(0, b);
    for (int i = 1; i < 11; i++)
    {
        imu.integrate(0.1*i, 1.0, avar);
    }
    imu.finished();

    ASSERT_NEAR(imu.dp(), 0.5, 1e-3);
    ASSERT_NEAR(imu.dv(), 1.0, 1e-3);
}
