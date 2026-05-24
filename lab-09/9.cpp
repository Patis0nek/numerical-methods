#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <openblas/lapacke.h>
#include <vector>

using namespace std;

double f(double x)
{
    return 1.0 / (1.0 + x * x);
}

double second_derivative(double x)
{
    return (6.0 * x * x - 2.0) / pow(x * x + 1.0, 3);
}

void save_nodes(const vector<double>& x, const vector<double>& y, int n)
{
    string filename = "nodes_n_" + to_string(n) + ".txt";
    ofstream file(filename);

    for (int i = 0; i <= n; i++) {
        file << x[i] << " " << y[i] << endl;
    }
    file.close();
}

void save_second_derivatives(const vector<double>& x, const vector<double>& M, int n)
{
    string filename = "second_derivatives_n_" + to_string(n) + ".txt";
    ofstream file(filename);

    for (int i = 0; i <= n; i++) {
        file << x[i] << " " << M[i] << " " << second_derivative(x[i]) << endl;
    }
    file.close();
}

double spline_value(double x_val, const vector<double>& x, const vector<double>& y, const vector<double>& M, int n)
{
    int p = -1;

    for (int i = 0; i <= n - 1; i++) {
        if (x_val >= x[i] && x_val <= x[i + 1]) {
            p = i;
            break;
        }
    }

    if (p >= 0) {
        int i = p;

        double h = x[i + 1] - x[i];

        double A = (y[i + 1] - y[i]) / h - h * (M[i + 1] - M[i]) / 6.0;
        double B = y[i] - M[i] * h * h / 6.0;

        double s = M[i] * pow(x[i + 1] - x_val, 3) / (6.0 * h) + M[i + 1] * pow(x_val - x[i], 3) / (6.0 * h) + A * (x_val - x[i]) + B;

        return s;
    }
    return 0.0;
}

void save_spline(const vector<double>& x, const vector<double>& y, const vector<double>& Mi, int n)
{
    string filename = "spline_n_" + to_string(n) + ".txt";
    ofstream file(filename);

    int M = 199;

    double xmin = x[0];
    double xmax = x[n];
    double dx = (xmax - xmin) / M;

    for (int m = 0; m <= M; m++) {
        double x_val = xmin + dx * m;
        double y_exact = f(x_val);
        double y_spline = spline_value(x_val, x, y, Mi, n);

        file << x_val << " " << y_exact << " " << y_spline << endl;
    }
    file.close();
}

void save_error(const vector<double>& x, const vector<double>& y, const vector<double>& Mi, int n)
{
    string filename = "error_n_" + to_string(n) + ".txt";
    ofstream file(filename);

    int M = 199;

    double xmin = x[0];
    double xmax = x[n];
    double dx = (xmax - xmin) / M;

    for (int m = 0; m <= M; m++) {
        double x_val = xmin + dx * m;
        double y_exact = f(x_val);
        double y_spline = spline_value(x_val, x, y, Mi, n);

        file << x_val << " " << y_exact - y_spline << endl;
    }
    file.close();
}

void cubic_spline(int n)
{
    double xmin = -5.0, xmax = 5.0;

    vector<double> x(n + 1);
    vector<double> y(n + 1);

    double h0 = (xmax - xmin) / n;

    for (int i = 0; i <= n; i++) {
        x[i] = xmin + i * h0;
        y[i] = f(x[i]);
    }

    vector<double> dl(n);
    vector<double> d(n + 1);
    vector<double> du(n);
    vector<double> b(n + 1);

    for (int i = 0; i <= n; i++) {
        d[i] = 0.0;
        b[i] = 0.0;
    }

    for (int i = 0; i < n; i++) {
        dl[i] = 0.0;
        du[i] = 0.0;
    }

    d[0] = 1.0;
    d[n] = 1.0;

    b[0] = 0.0;
    b[n] = 0.0;

    for (int i = 1; i <= n - 1; i++) {
        double h_i = x[i] - x[i - 1];
        double h_ip1 = x[i + 1] - x[i];

        double lambda = h_ip1 / (h_i + h_ip1);
        double mu = 1.0 - lambda;

        d[i] = 2.0;
        dl[i - 1] = mu;
        du[i] = lambda;

        b[i] = 6.0 / (h_i + h_ip1) * ((y[i + 1] - y[i]) / h_ip1 - (y[i] - y[i - 1]) / h_i);
    }

    du[0] = 0.0;
    dl[n - 1] = 0.0;

    lapack_int N = n + 1;
    lapack_int nrhs = 1;
    lapack_int ldb = N;

    LAPACKE_dgtsv(LAPACK_COL_MAJOR, N, nrhs, dl.data(), d.data(), du.data(), b.data(), ldb);

    vector<double> M(n + 1);

    for (int i = 0; i <= n; i++) {
        M[i] = b[i];
    }

    save_nodes(x, y, n);
    save_second_derivatives(x, M, n);
    save_spline(x, y, M, n);
    save_error(x, y, M, n);
}

int main()
{
    cubic_spline(5);
    cubic_spline(10);
    cubic_spline(50);

    return 0;
}