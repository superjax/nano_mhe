#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Sparse>


#include "nano_ad.h"

#ifdef NDEBUG
#define NANO_LM_ASSERT(...)
#else
#define NANO_LM_ASSERT(cond, message) \
    do { \
    if(!(cond)) \
{ \
    std::cerr << message << std::endl;\
    assert(cond); \
    } \
    } while(0)
#endif

using namespace Eigen;

namespace nano
{

template <typename T>
struct levenbergMarquardtParameters
{
    T factor = 100.0;
    int maxfev = 400; // max function evalutations
    T ftol = std::sqrt(NumTraits<T>::epsilon()); // tolerance in the norm of the vector function
    T xtol = std::sqrt(NumTraits<T>::epsilon());
    T gtol = 0.0; // tolerance of the norm of the error gradient
    T epsfcn = 0.0;
};

template<typename Scalar, typename functor>
class levenbergMarquardt
{
public:
    enum {
        NI = functor::NumInputs,
        NO = functor::NumOutputs,
        MAX = NI > NO ? NI : NO
    };

    typedef Matrix<Scalar, NI, 1> xVec;
    typedef Matrix<Scalar, NO, 1> yVec;
    typedef Matrix<Scalar, NO, NI> jMat;

    enum {
        NotStarted = -2,
        Running = -1,
        ImproperInputParameters = 0,
        RelativeReductionTooSmall = 1,
        RelativeErrorTooSmall = 2,
        RelativeErrorAndReductionTooSmall = 3,
        CosinusTooSmall = 4,
        TooManyFunctionEvaluation = 5,
        FtolTooSmall = 6,
        XtolTooSmall = 7,
        GtolTooSmall = 8,
        UserAsked = 9
    };

    levenbergMarquardt(const functor* f, const levenbergMarquardtParameters<Scalar>* params) :
        f_(*f),
        params_(*params)
    {
        NANO_LM_ASSERT(params->ftol >= 0 && params->xtol >= 0 && params->gtol >= 0 &&
                       params->maxfev > 0 && params->factor > 0,
                       "invalid parameters to nano LM");
        initialized_ = false;
    }

    int minimizeInit(xVec& x)
    {

        if (f_.Evaluate(y_, x, jac_))
        {
            ynorm_ = y_.stableNorm();
        }
        else
        {
            return UserAsked;
        }

        njev_ = 0;
        nfev_ = 1;
        par_ = 0.0;
        iter_ = 1;
        return NotStarted;
    }

    int minimizeOneStep(xVec& x)
    {
        Scalar temp, temp1, temp2;
        Scalar ratio;
        Scalar pnorm, xnorm, fnorm1, actred, dirder, prered;

        temp = 0.0;
        xnorm = 0.0;

        for (int j = 0; j < NI; j++)
            wa2_(j) = jac_.col(j).blueNorm();
        QRSolver_.compute(jac_);

        jacR_ = QRSolver_.matrixR();
        jacP_ = QRSolver_.colsPermutation();

        // on the first iteration scale according to the norms of the
        // columns of the initial jacobian.
        if (iter_ == 1)
        {
            for (int j = 0; j < NI; j++)
            {
                diag_(j) = (wa2_(j) == 0.0) ? 1.0 : wa2_(j);
            }

            /* on the first iteration, calculate the norm of the scaled x */
            /* and initialize the step bound m_delta. */
            xnorm = diag_.cwiseProduct(x).stableNorm();
            delta_ = params_.factor * xnorm;
            if (delta_ == 0.)
            {
                delta_ = params_.factor;
            }
        }

        // form q.T*y_ and store the first n components in  qTy.
        wa4_= QRSolver_.matrixQ().adjoint() * y_;
        qTy_ = wa4_.template head<NI>(); /// TODO: Check if this is ever not NI

        // compute the norm of the scaled gradient.
        gnorm_ = 0.0;
        if (ynorm_ != 0.0)
        {
            for (int j = 0; j < NI; ++j)
            {
                if (wa2_[jacP_.indices()[j]] != 0.0)
                {
                    gnorm_ = std::max(gnorm_, abs(jacR_.col(j).head(j+1).dot(qTy_.head(j+1)/ynorm_) /
                                                  wa2_[jacP_.indices()[j]]));
                }
            }
        }

        // test for convergence of the gradient norm.
        if (gnorm_ <= params_.gtol) {
            info_ = Success;
            return CosinusTooSmall;
        }

        do
        {
            calcLMParameter();

            /* store the direction p and x + p. calculate the norm of p. */
            wa1_ = -wa1_;
            wa2_ = x + wa1_;
            pnorm = diag_.cwiseProduct(wa1_).stableNorm();

            /* on the first iteration, adjust the initial step bound. */
            if (iter_ == 1)
                delta_ = std::min(delta_, pnorm);

            /* evaluate the function at x + p and calculate its norm. */
            if ( f_.Evaluate(wa4_, wa2_, jac_) < 0)
                return UserAsked;
            ++nfev_;
            fnorm1 = wa4_.stableNorm();

            /* compute the scaled actual reduction. */
            actred = -1.0;
            if (Scalar(0.1) * fnorm1 < ynorm_)
            {
                actred = 1.0 - numext::abs2(fnorm1 / ynorm_);
            }

            /* compute the scaled predicted reduction and */
            /* the scaled directional derivative. */

            wa3_ = jacR_.template triangularView<Upper>() * (jacP_.inverse() * wa1_);
            temp1 = numext::abs2(wa3_.stableNorm() / ynorm_);
            temp2 = numext::abs2(sqrt(par_) * pnorm / ynorm_);
            prered = temp1 + temp2 / Scalar(.5);
            dirder = -(temp1 + temp2);


            /* compute the ratio of the actual to the predicted */
            /* reduction. */
            ratio = 0.0;
            if (prered != 0.0)
            {
                ratio = actred / prered;
            }

            /* update the step bound. */
            if (ratio <= Scalar(0.25))
            {
                if (actred >= 0.0)
                    temp = Scalar(.5);
                if (actred < 0.)
                    temp = Scalar(.5) * dirder / (dirder + Scalar(.5) * actred);
                if (Scalar(.1) * fnorm1 >= ynorm_ || temp < Scalar(.1))
                    temp = Scalar(.1);
                /* Computing MIN */
                delta_ = temp * (std::min)(delta_, pnorm / Scalar(.1));
                par_ /= temp;
            }
            else if (!(par_ != 0. && ratio < Scalar(.75)))
            {
                delta_ = pnorm / Scalar(.5);
                par_ = Scalar(.5) * par_;
            }

            /* test for successful iteration. */
            if (ratio >= Scalar(1e-4)) {
                /* successful iteration. update x, m_fvec, and their norms. */
                x = wa2_;
                wa2_ = diag_.cwiseProduct(x);
                y_ = wa4_;
                xnorm = wa2_.stableNorm();
                ynorm_ = fnorm1;
                ++iter_;
            }

            /* tests for convergence. */
            if (abs(actred) <= params_.ftol && prered <= params_.ftol && Scalar(.5) * ratio <= 1. && delta_ <= params_.xtol * xnorm)
            {
                info_ = Success;
                return RelativeErrorAndReductionTooSmall;
            }
            if (abs(actred) <= params_.ftol && prered <= params_.ftol && Scalar(.5) * ratio <= 1.)
            {
                info_ = Success;
                return RelativeReductionTooSmall;
            }
            if (delta_ <= params_.xtol * xnorm)
            {
                info_ = Success;
                return RelativeErrorTooSmall;
            }

            /* tests for termination and stringent tolerances. */
            if (nfev_ >= params_.maxfev)
            {
                info_ = NoConvergence;
                return TooManyFunctionEvaluation;
            }
            if (abs(actred) <= NumTraits<Scalar>::epsilon() && prered <= NumTraits<Scalar>::epsilon() && Scalar(.5) * ratio <= 1.)
            {
                info_ = Success;
                return FtolTooSmall;
            }
            if (delta_ <= NumTraits<Scalar>::epsilon() * xnorm)
            {
                info_ = Success;
                return XtolTooSmall;
            }
            if (gnorm_ <= NumTraits<Scalar>::epsilon())
            {
                info_ = Success;
                return GtolTooSmall;
            }

        } while (ratio < Scalar(1e-4));
        return Running;
    }

    void calcLMParameter()
    {
        // Make a copy of the triangular factor.
        // This copy is modified during call the qrsolv
        jMat s = QRSolver_.matrixR();


        const Scalar dwarf = (std::numeric_limits<Scalar>::min)();


        /* compute and store in x the gauss-newton direction. if the */
        /* jacobian is rank-deficient, obtain a least squares solution. */

        //    const Index rank = qr.nonzeroPivots(); // exactly double(0.)
        const int rank = QRSolver_.rank(); // use a threshold
        xVec wa1 = qTy_;
        wa1.tail(NI-rank).setZero();
        wa1.head(rank) = s.topLeftCorner(rank,rank)
                .template triangularView<Upper>().solve(qTy_.head(rank));

        wa1_ = QRSolver_.colsPermutation()*wa1;

        // initialize the iteration counter.
        // evaluate the function at the origin, and test
        // for acceptance of the gauss-newton direction.
        int iter = 0;
        xVec wa2 = diag_.cwiseProduct(wa1_);
        Scalar dxnorm = wa2.blueNorm();
        Scalar fp = dxnorm - delta_;
        if (fp <= Scalar(0.1) * delta_) {
            par_ = 0.0;
            return;
        }

        // if the jacobian is not rank deficient, the newton
        // step provides a lower bound, parl, for the zero of
        // the function. otherwise set this bound to zero.
        Scalar parl = 0.0;
        Scalar temp;
        if (rank == NI) {
            wa1 = QRSolver_.colsPermutation().inverse() *  diag_.cwiseProduct(wa2)/dxnorm;
            s.template topLeftCorner<NI, NI>().transpose()
                    .template triangularView<Lower>().solveInPlace(wa1);
            temp = wa1.blueNorm();
            parl = fp / delta_ / temp / temp;
        }

        // calculate an upper bound, paru, for the zero of the function.
        for (int j = 0; j < NI; ++j)
        {
            wa1[j] = s.col(j).head(j+1).dot(qTy_.head(j+1)) / diag_[QRSolver_.colsPermutation().indices()(j)];
        }

        Scalar gnorm = wa1.stableNorm();
        Scalar paru = gnorm / delta_;
        if (paru == 0.0)
        {
            paru = dwarf / std::min(delta_, Scalar(0.1));
        }

        // if the input par lies outside of the interval (parl,paru),
        // set par to the closer endpoint.
        par_ = std::max(par_, parl);
        par_ = std::min(par_, paru);
        if (par_ == 0.0)
        {
            par_ = gnorm / dxnorm;
        }

        while (true)
        {
            ++iter;

            // evaluate the function at the current value of par.
            if (par_ == 0.0)
                par_ = std::max(dwarf, Scalar(.001) * paru);
            wa1 = sqrt(par_)* diag_;

            xVec sdiag;
            lmqrsolv(s, QRSolver_.colsPermutation(), wa1, sdiag);

            wa2 = diag_.cwiseProduct(wa1_);
            dxnorm = wa2.blueNorm();
            temp = fp;
            fp = dxnorm - delta_;

            // if the function is small enough, accept the current value
            // of par. also test for the exceptional cases where parl
            // is zero or the number of iterations has reached 10.
            if (abs(fp) <= Scalar(0.1) * delta_ || (parl == 0. && fp <= temp && temp < 0.) || iter == 10)
                break;

            /* compute the newton correction. */
            wa1 = QRSolver_.colsPermutation().inverse() * diag_.cwiseProduct(wa2/dxnorm);
            // we could almost use this here, but the diagonal is outside qr, in sdiag[]
            for (int j = 0; j < NI; ++j)
            {
                wa1[j] /= sdiag[j];
                temp = wa1[j];
                for (Index i = j+1; i < NI; ++i)
                {
                    wa1[i] -= s.coeff(i,j) * temp;
                }
            }
            temp = wa1.blueNorm();
            Scalar parc = fp / delta_ / temp / temp;

            /* depending on the sign of the function, update parl or paru. */
            if (fp > 0.0)
                parl = std::max(parl, par_);
            if (fp < 0.0)
                paru = std::min(paru, par_);

            /* compute an improved estimate for par. */
            par_ = std::max(parl, par_+ parc);
        }
        if (iter == 0)
            par_ = 0.0;
        return;
    }

    void lmqrsolv(
            jMat &s,
            const Eigen::PermutationMatrix<NI> &iPerm,
            const xVec &diag,
            xVec &sdiag)
    {
        /* Local variables */
        Index i, j, k;
        Scalar temp;
        Index n = s.cols();
        xVec  wa;
        JacobiRotation<Scalar> givens;

        /* Function Body */
        // the following will only change the lower triangular part of s, including
        // the diagonal, though the diagonal is restored afterward

        /*     copy r and (q transpose)*b to preserve input and initialize s. */
        /*     in particular, save the diagonal elements of r in x. */
        wa1_ = s.diagonal();
        wa = qTy_;


        s.topLeftCorner(n,n).template triangularView<StrictlyLower>() = s.topLeftCorner(n,n).transpose();
        /*     eliminate the diagonal matrix d using a givens rotation. */
        for (j = 0; j < n; ++j) {

            /*        prepare the row of d to be eliminated, locating the */
            /*        diagonal element using p from the qr factorization. */
            const int l = iPerm.indices()(j);
            if (diag[l] == 0.)
                break;
            sdiag.tail(n-j).setZero();
            sdiag[j] = diag[l];

            /*        the transformations to eliminate the row of d */
            /*        modify only a single element of (q transpose)*b */
            /*        beyond the first n, which is initially zero. */
            Scalar qtbpj = 0.;
            for (k = j; k < n; ++k) {
                /*           determine a givens rotation which eliminates the */
                /*           appropriate element in the current row of d. */
                givens.makeGivens(-s(k,k), sdiag[k]);

                /*           compute the modified diagonal element of r and */
                /*           the modified element of ((q transpose)*b,0). */
                s(k,k) = givens.c() * s(k,k) + givens.s() * sdiag[k];
                temp = givens.c() * wa[k] + givens.s() * qtbpj;
                qtbpj = -givens.s() * wa[k] + givens.c() * qtbpj;
                wa[k] = temp;

                /*           accumulate the transformation in the row of s. */
                for (i = k+1; i<n; ++i) {
                    temp = givens.c() * s(i,k) + givens.s() * sdiag[i];
                    sdiag[i] = -givens.s() * s(i,k) + givens.c() * sdiag[i];
                    s(i,k) = temp;
                }
            }
        }

        /*     solve the triangular system for z. if the system is */
        /*     singular, then obtain a least squares solution. */
        Index nsing;
        for(nsing=0; nsing<n && sdiag[nsing]!=0; nsing++) {}

        wa.tail(n-nsing).setZero();
        s.topLeftCorner(nsing, nsing).transpose()
                .template triangularView<Upper>().solveInPlace(wa.head(nsing));

        // restore
        sdiag = s.diagonal();
        s.diagonal() = wa1_;

        /* permute the components of z back to components of x. */
        wa1_ = iPerm * wa;
    }


    int minimize(xVec& x)
    {
        int status = minimizeInit(x);
        if (status != NotStarted)
        {
            initialized_ = false;
            return status;
        }

        do
        {
            status = minimizeOneStep(x);
        } while (status == Running);

        initialized_ = true;
        return status;
    }

    // Workspace
    yVec y_;
    xVec wa1_, wa2_; // temporary vectors
    yVec wa3_, wa4_; // temporary vectors
    jMat jac_;
    jMat jacR_; // The triangular matrix R from the QR of the jacobian matrix jac_
    xVec diag_;
    xVec qTy_;

    int info_;
    Scalar delta_;
    bool initialized_;
    Scalar ynorm_; // norm of the current output
    Scalar gnorm_;
    int iter_; // iteration counter
    Scalar par_; // levenberg marquardt parameter
    int nfev_; // function evaluator counter
    int njev_; // jacobian evaluator counter

    const functor& f_; // function we are minimizing
    const levenbergMarquardtParameters<Scalar>& params_;
    ColPivHouseholderQR<jMat> QRSolver_;
    Eigen::PermutationMatrix<NI> jacP_;
    std::function<bool(const Scalar*, const Scalar*, Scalar*)> boxplus_ = NULL;  // function pointer to boxplus operator (x = x [+] dx)
};

}
