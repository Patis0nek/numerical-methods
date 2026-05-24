#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <vector>

using namespace std;

const int n = 500;
const double delta = 1.0;
const int itmax = 10000;
const double eps = pow(10, -8);
const double omega = 1.99;

double scalar_product(const vector<double>& x, const vector<double>& y)
{
    double s = 0.0;

    for (int i = 0; i < (int)x.size(); i++) {
        s += x[i] * y[i];
    }

    return s;
}

double norm_vector(const vector<double>& x)
{
    return sqrt(scalar_product(x, x));
}

void csr_matvec(const vector<double>& a, const vector<int>& col, const vector<int>& row, const vector<double>& x, vector<double>& y)
{
    y.assign(n, 0.0);

    for (int i = 0; i < n; i++) {
        for (int l = row[i]; l < row[i + 1]; l++) {
            int j = col[l];
            y[i] += a[l] * x[j];
        }
    }
}

void fill_matrix_csr(vector<double>& a, vector<int>& col, vector<int>& row, vector<double>& d)
{
    int nnz_max = 3 * n;

    a.resize(nnz_max);
    col.resize(nnz_max);
    row.resize(n + 1);
    d.resize(n);

    int nnz = 0;

    for (int i = 0; i < n; i++) {
        row[i] = -1;

        if (i > 0) {
            a[nnz] = 1.0 / (delta * delta);
            col[nnz] = i - 1;
            row[i] = nnz;
            nnz++;
        }

        a[nnz] = -2.0 / (delta * delta);
        col[nnz] = i;
        d[i] = a[nnz];

        if (row[i] < 0) {
            row[i] = nnz;
        }

        nnz++;

        if (i < n - 1) {
            a[nnz] = 1.0 / (delta * delta);
            col[nnz] = i + 1;
            nnz++;
        }
    }

    row[n] = nnz;

    a.resize(nnz);
    col.resize(nnz);
}

void fill_rho(vector<double>& rho)
{
    rho.resize(n);

    for (int i = 0; i < n; i++) {
        rho[i] = 0.1 * pow(sin(2.0 * M_PI * i / (n - 1)), 10.0);
    }
}

void sor(const vector<double>& d, const vector<double>& rho, vector<double>& v, const vector<double>& a, const vector<int>& col, const vector<int>& row)
{
    for (int i = 0; i < n; i++) {
        double c = 0.0;

        for (int l = row[i]; l < row[i + 1]; l++) {
            int j = col[l];
            c += a[l] * v[j];
        }

        v[i] = v[i] + omega / d[i] * (rho[i] - c);
    }
}

void save_vector(const string& filename, const vector<double>& y)
{
    ofstream file(filename);

    for (int i = 0; i < n; i++) {
        double x = i * delta;
        file << x << " " << y[i] << "\n";
    }
}

int main()
{
    vector<double> a, d, rho, v, Av, r;
    vector<int> col, row;

    fill_matrix_csr(a, col, row, d);
    fill_rho(rho);

    v.resize(n);
    for (int i = 0; i < n; i++) {
        v[i] = 100.0 * rand() / RAND_MAX;
    }

    Av.resize(n);
    r.resize(n);

    ofstream file("iter_test_random.csv");

    int k = 0;
    double norm = 0.0;

    do {
        k++;

        sor(d, rho, v, a, col, row);
        csr_matvec(a, col, row, v, Av);

        for (int i = 0; i < n; i++) {
            r[i] = rho[i] - Av[i];
        }

        norm = norm_vector(r);

        file << k << " " << norm_vector(v) << " " << norm << "\n";

    } while (k < itmax && norm > eps);

    cout << "Iterations (random): " << k << "\n";
    cout << "Final residual norm: " << norm << "\n";

    save_vector("rho_test_random.csv", rho);
    save_vector("v_test_random.csv", v);

    return 0;
}