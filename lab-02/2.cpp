#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <openblas/lapacke.h>
#include <vector>

using namespace std;

const int N = 6;

void build_matrix(double x0, double x1, double x2, double x3, vector<double>& A)
{
    A.resize(N * N);

    int k = 0;
    for (int i = 0; i < N; i++) {
        A[k * N + i] = pow(x0, i);
    }

    k = 1;
    A[k * N] = 0;
    for (int i = 1; i < N; i++) {
        A[k * N + i] = i * pow(x0, i - 1);
    }

    k = 2;
    for (int i = 0; i < N; i++) {
        A[k * N + i] = pow(x1, i);
    }

    k = 3;
    for (int i = 0; i < N; i++) {
        A[k * N + i] = pow(x2, i);
    }

    k = 4;
    for (int i = 0; i < N; i++) {
        A[k * N + i] = pow(x3, i);
    }

    k = 5;
    A[k * N] = 0;
    for (int i = 1; i < N; i++) {
        A[k * N + i] = i * pow(x3, i - 1);
    }
}

void print_matrix(const vector<double>& A)
{
    cout << "Matrix A:\n";

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            cout << setw(5) << A[i * N + j] << " ";
        }
        cout << "\n";
    }
    cout << "\n";
}

double polynomial(double x, vector<double>& b)
{
    double f = 0.0;
    for (int i = 0; i < N; i++) {
        f += b[i] * pow(x, i);
    }

    return f;
}

double determinant(const vector<double>& LU)
{
    double det_u = 1.0;

    for (int i = 0; i < N; i++) {
        det_u *= LU[i * N + i];
    }

    return det_u;
}

void solve(double x0, double x1, double x2, double x3)
{
    vector<double> A;
    build_matrix(x0, x1, x2, x3, A);
    print_matrix(A);

    vector<double> LU = A;
    vector<double> b = { 0.0, 1.0, 1.0, 0.0, -1.0, 0.0 };

    vector<lapack_int> ipiv(N);

    LAPACKE_dgetrf(LAPACK_ROW_MAJOR, N, N, LU.data(), N, ipiv.data());

    if (determinant(LU) == 0.0) {
        cout << "The system has no unique solution.\n";
        return;
    } else {
        cout << "Determinant = " << determinant(LU) << "\n";
    }

    LAPACKE_dgetrs(LAPACK_ROW_MAJOR, 'N', N, 1, LU.data(), N, ipiv.data(), b.data(), 1);

    ofstream f_coeff("coefficients.txt");
    cout << "\nCoefficients:\n";
    for (int i = 0; i < N; i++) {
        cout << "a" << i << " = " << b[i] << "\n";
        f_coeff << b[i] << "\n";
    }
    f_coeff.close();

    ofstream f_plot("plot.txt");
    double dx = 0.01;
    for (double x = 0; x <= 3; x += dx) {
        f_plot << x << " " << polynomial(x, b) << "\n";
    }
    f_plot.close();
}

int main()
{
    double x0 = 0.0;
    double x1 = 1.0;
    double x2 = 2.0;
    double x3 = 3.0;

    solve(x0, x1, x2, x3);

    return 0;
}