#include <cmath>
#include <fstream>
#include <iostream>
#include <openblas/lapacke.h>
#include <string>
#include <utility>
#include <vector>

using namespace std;

double f(double x)
{
    return exp(-x * x);
}

double factorial(double n)
{
    double result = 1;
    for (int i = 1; i <= n; i++) {
        result *= i;
    }

    return result;
}

double coefficient(int k)
{
    if (k < 0) {
        return 0.0;
    }

    if (k % 2 == 1) {
        return 0.0;
    }

    int p = k / 2;

    return pow(-1.0, p) / factorial(p);
}

void build_c(int n, vector<double>& c)
{
    c.resize(n + 1);

    for (int k = 0; k <= n; k++) {
        c[k] = coefficient(k);
    }
}

void solve_pade(int N, int M, vector<double>& a, vector<double>& b)
{
    vector<double> c;
    build_c(N + M, c);

    vector<double> A(M * M);
    vector<double> y(M);
    vector<lapack_int> ipiv(M);

    for (int i = 0; i < M; i++) {
        for (int j = 0; j < M; j++) {
            A[j + i * M] = coefficient(N - M + i + j + 1);
        }
        y[i] = -coefficient(N + 1 + i);
    }

    LAPACKE_dsysv(LAPACK_COL_MAJOR, 'U', M, 1, A.data(), M, ipiv.data(), y.data(), M);

    b.resize(M + 1);
    b[0] = 1.0;

    for (int i = 0; i < M; i++) {
        b[M - i] = y[i];
    }

    a.resize(N + 1);

    for (int i = 0; i <= N; i++) {
        a[i] = 0.0;

        for (int j = 0; j <= i; j++) {
            a[i] += coefficient(i - j) * b[j];
        }
    }
}

double polynomial(double x, const vector<double>& p)
{
    double value = 0.0;

    for (int i = p.size() - 1; i >= 0; i--) {
        value = value * x + p[i];
    }

    return value;
}

double R(double x, const vector<double>& a, const vector<double>& b)
{
    return polynomial(x, a) / polynomial(x, b);
}

void save_results(int N, int M)
{
    vector<double> a;
    vector<double> b;

    solve_pade(N, M, a, b);

    string file_name = "R" + to_string(N) + "_" + to_string(M) + ".txt";
    ofstream file(file_name);

    float dx = 0.01;

    for (double x = -5.0; x <= 5.0; x += dx) {
        file << x << " " << f(x) << " " << R(x, a, b) << endl;
    }
    file.close();
}

int main()
{
    vector<pair<int, int>> cases = { { 2, 2 }, { 4, 4 }, { 6, 6 }, { 2, 4 }, { 2, 6 }, { 2, 8 } };

    for (int i = 0; i < cases.size(); i++) {
        save_results(cases[i].first, cases[i].second);
    }

    return 0;
}