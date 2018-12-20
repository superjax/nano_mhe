#include <gtest/gtest.h>
#include <iostream>
#include <Eigen/Dense>
#include "ADJacobian.h"

#include "nano_ad.h"

template<typename T>
T scale(T in, T s)
{
    return s * in;
}

template<typename... Ts> void func(Ts... args){
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

TEST (CostFunctor, Compile)
{
    typedef Matrix<double, 2, 1> OT;
    typedef Matrix<double, 3, 1> IT1;
    typedef Matrix<double, 4, 1> IT2;
    typedef Matrix<double, 2, 3> JT1;
    typedef Matrix<double, 2, 4> JT2;
    CostFunctor<OT, IT1, IT2> f;

    OT res;
    IT1  x;
    IT2  x2;
    JT1  j;
    JT2  j2;
    f.Evaluate(res, x, x2);
    f.Evaluate(res, x, x2, j, j2);
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

