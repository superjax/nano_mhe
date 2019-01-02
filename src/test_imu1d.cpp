#include <fstream>
#include <gtest/gtest.h>

#include "test_common.h"
#include "nano_mhe1d.h"
#include "utils/robot1d.h"

TEST(Imu1D, init)
{
    double b = 0;
    double avar = 0.3;
    Imu1D<double> imu;
    imu.init(b, &avar);
}

TEST(Imu1D, integrate)
{
    double b = 0;
    double avar = 0.3;
    Imu1D<double> imu;
    imu.init(b, &avar);

    imu.reset(0);
    for (int i = 1; i < 11; i++)
    {
        imu.integrate(0.1*i, 1.0);
    }
    imu.finished();

    ASSERT_NEAR(imu.dp(), 0.5, 1e-3);
    ASSERT_NEAR(imu.dv(), 1.0, 1e-3);
}
