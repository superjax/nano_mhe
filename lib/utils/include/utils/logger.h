#include <fstream>
#include <Eigen/Dense>


template <typename Scalar>
class Logger
{
public:
    Logger(std::string filename)
    {
        file_.open(filename);
    }

    ~Logger()
    {
        file_.close();
    }
    template <typename... T>
    void log(T... data)
    {
        int dummy[sizeof...(data)] = { (file_.write((char*)&data, sizeof(Scalar)), 1)... };
    }

private:
    std::ofstream file_;
};
