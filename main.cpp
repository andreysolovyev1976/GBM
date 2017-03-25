#include <iostream>
#include <vector>
#include <memory>
#include <Eigen/Dense>
#include <exception>
#include <random>

#include "EquityPriceGenerator.h"

using Eigen::MatrixXd;
using Eigen::VectorXd;

using std::unique_ptr;
using std::mt19937_64;

using std::normal_distribution;
using std::exception;

using std::vector;


using std::cout;
using std::endl;

void data_generation_(MatrixXd& X, VectorXd& y);

void straightforward_solution_(const MatrixXd& X, const VectorXd& y);
void svd_solution_(const MatrixXd& X, const VectorXd& y);
void FullPivHouseholderQR_(const MatrixXd& X, const VectorXd& y);
void colPivHouseholderQr_(const MatrixXd& X, const VectorXd& y);
void normal_(const MatrixXd& X, const VectorXd& y);

void error_level_();



int main()
{
    MatrixXd X(3,2);
    VectorXd y(3);

   // data_generation_(X, y);

    X <<  1.0,  2.0, -7.0,
            3.0, 2.0, -3.0;
    y <<  -1.0, 2.0, -3.0;

//    cout << X << endl;
//    cout << y << endl;

    straightforward_solution_(X, y);
    svd_solution_(X, y);
    FullPivHouseholderQR_(X, y);
    colPivHouseholderQr_(X, y);
    normal_(X, y);

//    error_level_();

    return 0;
}


void straightforward_solution_(const MatrixXd& X, const VectorXd& y)
{

    MatrixXd LHS_temp_, RHS_temp_, b;

    LHS_temp_ = X.transpose() * X;
    LHS_temp_ = LHS_temp_.inverse();

    RHS_temp_ = X.transpose() * y;

    b = LHS_temp_ * RHS_temp_;

    std::cout << "The solution using the jacobiSvd is:\n" << b << endl << endl;
}

void svd_solution_(const MatrixXd& X, const VectorXd& y)
{

    MatrixXd b = X.jacobiSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(y);

    cout << "The solution using the jacobiSvd is:\n" << b << endl << endl;

}

void FullPivHouseholderQR_(const MatrixXd& X, const VectorXd& y)
{

    MatrixXd b = X.fullPivHouseholderQr ().solve (y);

//    assert(y.isApprox (X * b));

    cout << "The solution using the QR decomposition (fullPivHouseholderQr) is:\n" << b << endl << endl;
}

void colPivHouseholderQr_(const MatrixXd& X, const VectorXd& y)
{

    MatrixXd b = X.colPivHouseholderQr ().solve (y);

    cout << "The solution using the QR decomposition (colPivHouseholderQr) is:\n" << b << endl << endl;
}

void normal_(const MatrixXd& X, const VectorXd& y)
{
    MatrixXd b = (X.transpose() * X).ldlt().solve(X.transpose() * y);
    cout << "The solution using normal equations is:\n" << b << endl << endl;
}

void error_level_()
{
    MatrixXd A = MatrixXd::Random (100, 100);
    MatrixXd b = MatrixXd::Random (100, 50);
    MatrixXd x = A.fullPivLu ().solve (b);
    double relative_error = (A * x - b).norm () / b.norm (); // norm() is L2 norm
    cout << "The relative error is:\n" << relative_error << endl;
}

void data_generation_(MatrixXd& X, VectorXd& y)
{
    double spot = 50.38;
    double riskFreeRate = 0.025;
    double volatility = 6.25;
    unsigned numTimeSteps = 10;
    unsigned numScenarios = 2;
    int seed = 100;

    EquityPriceGenerator epg(spot, numTimeSteps, 0.5, riskFreeRate, volatility);


    using realVector = vector<double>;
    realVector
            x1 = epg(100),
            x2 = epg(-100);


    VectorXd x1_(x1.size()),
            x2_(x2.size());

    x1_ = VectorXd::Map(&x1[0], x1_.size());
    x2_ = VectorXd::Map(&x2[0], x2_.size());

    y.resize(2);
    X.resize(y.size(), x1_.size());

    y << 12.0, -12.0;
    X.row(0) = x1_;
    X.row(1) = x2_;
}
