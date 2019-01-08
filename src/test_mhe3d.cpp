#include <gtest/gtest.h>
#include <Eigen/Core>

#include "test_common.h"
#include "nano_mhe3d.h"

TEST (nano_mhe_3d, compile)
{
    MHE_3D<double, 5> mhe;
}

TEST (nano_mhe_3d, set_bias)
{
    MHE_3D<double, 5> mhe;
    Matrix<double, 6, 1> b;
    b << 1, 2, 3, 4, 5, 6;
    mhe.setBias(b);
    ASSERT_MAT_EQ(mhe.X_.bottomRows(6), b);
    ASSERT_MAT_EQ(b, mhe.getBias());
}

TEST (nano_mhe_3d, add_node)
{
    MHE_3D<double, 5> mhe;
    MHE_3D<double, 5>::nVec x0;
    x0.setZero();
    mhe.addNode(0, x0);
    ASSERT_MAT_EQ(x0, mhe.getState(0));
}
