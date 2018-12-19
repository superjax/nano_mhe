#include <gtest/gtest.h>

#include "nano_mhe.h"

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

TEST (nano_mhe, init)
{
    MHE_1D<double, 5> mhe;
}

TEST (nano_mhe, add_node)
{
    MHE_1D<double, 5> mhe;
    MHE_1D5::nVec x0{0, 0};
    mhe.add_node(0, x0);
    ASSERT_MAT_EQ(x0, mhe.getState(0));
}

TEST (nano_mhe, overfill_nodes)
{
    MHE_1D<double, 5> mhe;
    for (int i = 0; i < 8; i ++)
    {
        MHE_1D5::nVec x{i, 1};
        mhe.add_node(i, x);
    }

    for (int i = 0; i < 8; i++)
    {
        MHE_1D5::nVec x{i, 1};
        if (i < 3)
        {
            ASSERT_THROW(mhe.getState(i), std::runtime_error);
        }
        else
        {
            ASSERT_MAT_EQ(x, mhe.getState(i));
        }
    }
}
