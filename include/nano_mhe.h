#include <Eigen/Core>

using namespace Eigen;

#ifdef NDEBUG
#define NANO_MHE_ASSERT(...)
#else
#define NANO_MHE_ASSERT(cond, message) \
    if(!(cond)) \
    { \
        throw std::runtime_error(message);\
    }
#endif


// N is the number of nodes
template<typename T, int K>
class MHE_1D
{
static constexpr int NY = (K-1)+K; // number of residuals in estimator
static constexpr int NX = (K*2)+1; // number of states in estimator
typedef Matrix<T, NX, 1> XVec;
typedef Matrix<T, NY, 1> YVec;
typedef Matrix<T, NY, NX> JMat;

public:
    typedef Matrix<T, 2, 1> nVec; // node
    MHE_1D()
    {
        latest_k_ = -1;
        latest_n_ = -1;
        X_.setZero();
    }

    const Map<const nVec> getState(int node)
    {
        NANO_MHE_ASSERT(node <= latest_n_, "Tried acces a node from the future");
        NANO_MHE_ASSERT(node > latest_n_-K, "Tried acces a node beyond buffer");
        NANO_MHE_ASSERT(latest_k_ >= 0 && latest_n_ >= 0, "tried to get state before setting state");

        return Map<const nVec>(X_.data() + 2*n2k(node));
    }

    int add_node(int node, const nVec& xhat)
    {
        NANO_MHE_ASSERT(node == latest_n_ + 1, "Tried to add non-sequential node!");

        latest_k_ = (latest_k_ + 1) % K;
        latest_n_ = node;
        X_.template segment<2>(2*latest_k_) = xhat;
    }

    inline int n2k(int node)
    {
        return ((K + latest_k_) - (latest_n_-node)) % K;
    }

    XVec X_;
    JMat J_;
    int latest_k_; // latest internal index [k]
    int latest_n_; // latest node id



};
