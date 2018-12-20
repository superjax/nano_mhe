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
