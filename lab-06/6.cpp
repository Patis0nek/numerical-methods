#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <openblas/lapacke.h>
#include <string>
#include <vector>

using namespace std;

const double pi = M_PI;

double calculate_function(double x)
{
    return x * cos(x);
}

double calculate_fit(double x, const vector<double>& c)
{
    double result = 0.0, power = 1.0;

    for (int j = 0; j < (int)c.size(); j++) {
        result += c[j] * power;
        power *= x;
    }

    return result;
}

void generate_data(int m, double xa, double xb, vector<double>& x, vector<double>& y)
{
    x.resize(m);
    y.resize(m);

    for (int i = 0; i <= m - 1; i++) {
        x[i] = xa + i * (xb - xa) / (m - 1);
        y[i] = calculate_function(x[i]);
    }
}

void fill_matrix_A(int m, int n, const vector<double>& x, vector<double>& A)
{
    A.resize(m * n);

    for (int i = 0; i <= m - 1; i++) {
        double power = 1.0;

        for (int j = 0; j <= n - 1; j++) {
            A[i * n + j] = power;
            power *= x[i];
        }
    }
}

void save_vector_xy(const string& fileName, const vector<double>& x, const vector<double>& y)
{
    ofstream file(fileName);

    for (int i = 0; i < (int)x.size(); i++) {
        file << fixed << x[i] << " " << y[i] << "\n";
    }

    file.close();
}

void save_vector(const string& fileName, const vector<double>& v)
{
    ofstream file(fileName);

    for (int i = 0; i < (int)v.size(); i++) {
        file << fixed << v[i] << "\n";
    }

    file.close();
}

void save_matrix(const string& fileName, const vector<double>& A, int m, int n)
{
    ofstream file(fileName);

    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            file << fixed << setw(12) << A[i * n + j];
            if (j < n - 1) {
                file << " ";
            }
        }
        file << "\n";
    }

    file.close();
}

void reconstruct_matrix(int m, int n, const vector<double>& s, const vector<double>& u, const vector<double>& vt, vector<double>& A_reconstructed)
{
    A_reconstructed.assign(m * n, 0.0);

    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            double sum = 0.0;

            for (int k = 0; k < n; k++) {
                sum += s[k] * u[i * n + k] * vt[k * n + j];
            }

            A_reconstructed[i * n + j] = sum;
        }
    }
}

void calculate_coefficients(int m, int n, const vector<double>& y, const vector<double>& s, const vector<double>& u, const vector<double>& vt, vector<double>& c)
{
    c.assign(n, 0.0);

    for (int k = 0; k < n; k++) {
        double scalarProduct = 0.0;

        for (int i = 0; i < m; i++) {
            scalarProduct += u[i * n + k] * y[i];
        }

        for (int j = 0; j < n; j++) {
            c[j] += (scalarProduct / s[k]) * vt[k * n + j];
        }
    }
}

void save_plot_data(const string& fileName, double xa, double xb, const vector<double>& x, const vector<double>& y, const vector<double>& c)
{
    ofstream file(fileName);

    int points = 100;

    for (int i = 0; i < points; i++) {
        double xPlot = xa + i * (xb - xa) / (points - 1);
        double yExact = calculate_function(xPlot);
        double yFit = calculate_fit(xPlot, c);

        file << fixed << xPlot << " " << yExact << " " << yFit << "\n";
    }

    file.close();
}

void solve_case(int m, int n)
{
    double xa = 0.0;
    double xb = 3.0 * pi;

    vector<double> x, y, A;
    vector<double> s(n), u(m * n), vt(n * n);
    vector<double> superb(n - 1);
    vector<double> A_reconstructed;
    vector<double> c;

    generate_data(m, xa, xb, x, y);
    fill_matrix_A(m, n, x, A);

    save_vector_xy("data.txt", x, y);
    save_matrix("A.txt", A, m, n);

    vector<double> A_copy;
    A_copy = A;

    LAPACKE_dgesvd(LAPACK_ROW_MAJOR, 'S', 'S', m, n, A_copy.data(), n, s.data(), u.data(), n, vt.data(), n, superb.data());

    save_vector("singular_values.txt", s);

    reconstruct_matrix(m, n, s, u, vt, A_reconstructed);
    save_matrix("A_reconstructed.txt", A_reconstructed, m, n);

    calculate_coefficients(m, n, y, s, u, vt, c);
    save_vector("coefficients.txt", c);

    save_plot_data("plot.txt", xa, xb, x, y, c);
}

int main()
{
    solve_case(30, 20);

    return 0;
}