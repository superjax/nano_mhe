#include <gtest/gtest.h>
#include <iostream>
#include <Eigen/Dense>
#include "ADJacobian.h"

#include "nano_ad.h"

#include "test_common.h"
template<typename T>
T scale(T in, T s)
{
    return s * in;
}

template<typename... Ts>
void func(Ts... args)
{
    const int size = sizeof...(args) + 2;
    int res[size] = {1,args...,2};
    // since initializer lists guarantee sequencing, this can be used to
    // call a function on each element of a pack, in order:
    int scaled[sizeof...(Ts)] = { scale(args, 2)... };
}

TEST (varTemp, variadic_templates)
{
    func<int, int, int>(1, 2, 3);
    func<int, int, int, int, int, int>(1, 2, 3, 4, 5, 6);
}

TEST (CostFunctorAutoDiff, Compile)
{
    typedef Matrix<double, 2, 1> OT;
    typedef Matrix<double, 3, 1> IT1;
    typedef Matrix<double, 4, 1> IT2;
    typedef Matrix<double, 2, 3> JT1;
    typedef Matrix<double, 2, 4> JT2;
    CostFunctorAutoDiff<EmptyCostFunctor, OT, IT1, IT2> f;

    OT res;
    IT1 x;
    IT2 x2;
    JT1 j;
    JT2 j2;
    f.Evaluate(res, x, x2);
    f.Evaluate(res, x, x2, j, j2);
}


TEST (Autodiff, SimpleTest)
{
    Eigen::AutoDiffScalar<Eigen::Vector2d> x, y, z;
    x = 8.0;
    x.derivatives() << 1,0;
    z = 2.0;
    z.derivatives() << 0,1;

    y = x*x + z;

//    std::cout << "x = " << x << "\n"
//              << "z = " << z << "\n"
//              << "y = x^2 + z = " << y << "\n"
//              << "dy/dx = 2x = " << y.derivatives()[0] << "\n"
//              << "dy/dz = 1 = " << y.derivatives()[1] << std::endl;
}

TEST (Autodiff, VectorTest)
{
    enum {
        NI = 2,
        NO = 2
    };
    typedef Eigen::Matrix<double, NI, 1> xVec;
    typedef Eigen::Matrix<double, NO, 1> yVec;
    typedef Eigen::AutoDiffScalar<xVec> ADScalar;
    typedef Eigen::Matrix<ADScalar, NI, 1> xAD;
    typedef Eigen::Matrix<ADScalar, NO, 1> yAD;

    yAD y;
    xAD x;
    x << 8.0, 2.0;
    x(0).derivatives() << 1,0;
    x(1).derivatives() << 0,1;

    y = x.squaredNorm() * Vector2d{1, 1};

//    std::cout << "x = [" << x.transpose() << "]\n"
//              << "y = [x0^2 + x1^2, x0^2 + x1^2] = [" << y.transpose() << "]\n"
//              << "dy/dx = 2x0 2x1    = " << y(0).derivatives().transpose() << "\n"
//              << "        2x0 2x1      " << y(1).derivatives().transpose() << std::endl;
}

TEST (Autodiff, MultiParameterVectorTest)
{
    enum {
        NI = 4,
        NO = 2,
        NX1 = 2,
        NX2 = 2
    };
    typedef Eigen::Matrix<double, NI, 1> inputVec;
    typedef Eigen::Matrix<double, NO, 1> yVec;
    typedef Eigen::AutoDiffScalar<inputVec> ADScalar;
    typedef Eigen::Matrix<ADScalar, NX1, 1> x1AD;
    typedef Eigen::Matrix<ADScalar, NX2, 1> x2AD;
    typedef Eigen::Matrix<ADScalar, NO, 1> yAD;

    yAD y;
    x1AD x1;
    x2AD x2;
    x1 << 8.0, 2.0;
    x2 << 3.0, 4.0;
    x1(0).derivatives() << 1, 0, 0, 0;
    x1(1).derivatives() << 0, 1, 0, 0;
    x2(0).derivatives() << 0, 0, 1, 0;
    x2(1).derivatives() << 0, 0, 0, 1;


    y = x1.cwiseProduct(x2);

//    std::cout << "x1 = [" << x1.transpose() << "]\n"
//              << "x2 = [" << x2.transpose() << "]\n"
//              << "y = [x12*x21, x12*x22] = [" << y.transpose() << "]\n"
//              << "dy/dx1 = x21 0    = " << y(0).derivatives().segment<NX1>(0).transpose() << "\n"
//              << "         0   x22    " << y(1).derivatives().segment<NX1>(0).transpose() << "\n"
//              << "dy/dx2 = x11 0    = " << y(0).derivatives().segment<NX2>(NX1).transpose() << "\n"
//              << "         0   x12    " << y(1).derivatives().segment<NX2>(NX1).transpose() << std::endl;
}


//using namespace Eigen;

//class testFunctor : public CostFunctor<double, 2, 1, 2, 3>
//{

//};

///*
// * Testing differentiation that will produce a gradient.
// */
//template <typename Scalar, int iR>
//Scalar scalar_func(const Matrix<Scalar, iR, 1> &input, Matrix<Scalar, iR, 1> *grad)
//{
//  eigen_assert(grad != 0);

//  /* Some typedefs to not need and rewrite the long expressions. */
//  typedef AutoDiffScalar< Matrix<Scalar, iR, 1> > ADS;

//  /* Create and initialize the AutoDiff vector. */
//  Matrix<ADS, iR, 1> ad;

//  for (int i = 0; i < iR; i++)
//    ad(i) = ADS(input(i), iR, i);  // AutoDiff initialization

//  ADS s(0);

//  for (int i = 0; i < iR; i++)
//  {
//    s += exp(ad(i));
//  }

//  (*grad) = s.derivatives();

//  return s.value();
//}

///*
// * Testing differentiation that will produce a Jacobian, using functors and the
// * ADJacobian helper.
// *
// * Example: 2 state 0th order integrator
// */
//template <typename Scalar>
//struct integratorFunctor
//{
//  /*
//   * Definitions required by ADJacobian.
//   */
//  typedef Matrix<Scalar, 2, 1> InputType;
//  typedef Matrix<Scalar, 2, 1> ValueType;

//  /*
//   * Implementation starts here.
//   */
//  integratorFunctor(const Scalar gain) :
//      _gain(gain)
//  {}

//  const Scalar _gain;

//  /* The types of the arguments are inferred by ADJacobian.
//   * For ease of thinking, IT = InputType and V2 = ValueType. */
//  template <typename IT, typename V2>
//  void operator() (const IT &input, V2 *output, const Scalar dt) const
//  {
//    V2 &o = *output;

//    /* Integrator to test the AD. */
//    o[0] = input[0] + input[1] * dt * _gain;
//    o[1] = input[1] * _gain;
//  }
//};


//int main(int argc, char *argv[])
//{

//  typedef Matrix<float, 3, 1> myvec;

//  /* Value vector for the gradient test. */
//  myvec vec, vec_grad;
//  vec << 1,2,3;

//  /*
//   * Run the test using AutoDiffScalar.
//   */
//  auto grad_test = scalar_func(vec, &vec_grad);


//  /*
//   * Run the example using ADJacobian.
//   */

//  /* Input vector and sampling time. */
//  Matrix<float, 2, 1> in;
//  in << 1,2;
//  const float dt = 1e-2;
//  const float gain = 3;

//  /* Outputs. */
//  Matrix<float, 2, 1> out;
//  Matrix<float, 2, 2> jac;

//  /* Test the ADJacobian. */
//  ADJacobian< integratorFunctor<float> > adjac(3);
//  adjac(in, &out, &jac, dt);


//  /*
//   * Do some printing.
//   */
//  std::cout << "grad_test = " << std::endl;
//  std::cout << grad_test << std::endl << std::endl;
//  std::cout << "Gradient of grad_test = " << std::endl;
//  std::cout << vec_grad << std::endl << std::endl;

//  std::cout << std::endl << "ADJacobian test on 0th order integrator, dt = "
//            << dt << std::endl;
//  std::cout << "in = " << std::endl;
//  std::cout << in << std::endl << std::endl;
//  std::cout << "out = " << std::endl;
//  std::cout << out << std::endl << std::endl;
//  std::cout << "Jacobian = " << std::endl;
//  std::cout << jac << std::endl;

//  return 0;
//}

