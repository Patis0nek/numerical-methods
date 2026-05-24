#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <openblas/cblas.h>
#include <time.h>
#include <vector>

using namespace std;

double vec_norm(const vector<double>& x)
{
    double s = 0.0;

    for (double v : x) {
        s += v * v;
    }

    return sqrt(s);
}

void build_matrix(vector<double>& A, int n, int gamma)
{
    A.resize(n * n);

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            double d = abs(i - j);
            A[i * n + j] = 1.0 / (d * d * d * d + gamma);
        }
    }
}

void matrix_vector(const vector<double>& A, const vector<double>& x, vector<double>& y, int n)
{
    y.assign(n, 0.0);

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            y[i] += A[i * n + j] * x[j];
        }
    }
}

void save_norms(const vector<double>& A, int n, int m, int gamma)
{
    vector<double> x(n, 1.0), y;
    ofstream file("norms_gamma=" + to_string(gamma) + ".txt");

    for (int i = 1; i <= m; i++) {
        matrix_vector(A, x, y, n);

        file << i << " " << setprecision(15) << vec_norm(y) << "\n";
        x = y;
    }

    file.close();
}

void matrix_multiplication_own(const vector<double>& A, vector<double>& C, int n)
{
    C.assign(n * n, 0.0);

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            for (int k = 0; k < n; k++) {
                C[i * n + j] += A[i * n + k] * A[k * n + j];
            }
        }
    }
}

void matrix_multiplication_blas(const vector<double>& A, vector<double>& C, int n)
{
    C.assign(n * n, 0.0);

    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, n, n, n, 1.0, A.data(), n, A.data(), n, 0.0, C.data(), n);
}

int main()
{
    int n = 2000;
    int m = 30;

    for (int gamma : { 2, 3 }) {
        vector<double> A;

        build_matrix(A, n, gamma);
        save_norms(A, n, m, gamma);

        vector<double> C;

        time_t start, end;

        start = time(NULL);
        matrix_multiplication_own(A, C, n);
        end = time(NULL);

        cout << "Own multiplication gamma = " << gamma << ": " << difftime(end, start) << " s\n";

        start = time(NULL);
        matrix_multiplication_blas(A, C, n);
        end = time(NULL);

        cout << "BLAS multiplication gamma = " << gamma << ": " << difftime(end, start) << " s\n";
    }

    return 0;
}