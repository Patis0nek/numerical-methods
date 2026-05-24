#include <cmath>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <iostream>

using namespace std;

const int n = 6;
const int itmax = 150;
const double eps = pow(10, -8);

double scalar_product(const double a[n], const double b[n])
{
    double s = 0.0;

    for (int i = 0; i < n; i++) {
        s += a[i] * b[i];
    }
    return s;
}

double vector_norm(const double x[n])
{
    return sqrt(scalar_product(x, x));
}

void matrix_vector(const double A[n][n], const double x[n], double y[n])
{
    for (int i = 0; i < n; i++) {
        y[i] = 0.0;

        for (int j = 0; j < n; j++) {
            y[i] += A[i][j] * x[j];
        }
    }
}

void normalize(double x[n])
{
    double norm = vector_norm(x);

    if (norm == 0.0) {
        return;
    }

    for (int i = 0; i < n; i++) {
        x[i] /= norm;
    }
}

void gram_schmidt(double y[n], const double xk[n][n], int k)
{
    for (int m = 0; m < k; m++) {
        double c = 0.0;

        for (int i = 0; i < n; i++) {
            c += xk[m][i] * y[i];
        }

        for (int i = 0; i < n; i++) {
            y[i] -= c * xk[m][i];
        }
    }
}

void fill_matrix(double A[n][n])
{
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            if (j == i) {
                A[i][j] = -2.0;
            } else if (j == i - 1 || j == i + 1) {
                A[i][j] = 1.0;
            } else {
                A[i][j] = 0.0;
            }
        }
    }
}

void print_matrix(const double A[n][n], const string& name)
{
    cout << name << endl;

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            cout << setw(10) << fixed << A[i][j];
        }
        cout << endl;
    }
    cout << endl;
}

void save_matrix(const double A[n][n], const string& filename)
{
    ofstream file(filename);

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            file << setw(10) << fixed << A[i][j];
        }
        file << '\n';
    }
    file.close();
}

int main()
{
    double A[n][n];
    double xk[n][n];
    double lambda_k[n];
    double A_num[n][n];
    double A_inv_num[n][n];

    srand(time(NULL));

    fill_matrix(A);
    print_matrix(A, "Matrix A:");

    for (int i = 0; i < n; i++) {
        lambda_k[i] = 0.0;

        for (int j = 0; j < n; j++) {
            xk[i][j] = 0.0;
            A_num[i][j] = 0.0;
            A_inv_num[i][j] = 0.0;
        }
    }

    for (int k = 0; k < n; k++) {
        double x[n];
        double y[n];
        double Ax[n];

        for (int i = 0; i < n; i++) {
            x[i] = (double)rand() / RAND_MAX;
        }

        normalize(x);

        double lambda_old = pow(10, 5);
        double lambda = 0.0;

        string filename = "lambda_" + to_string(k) + ".txt";
        ofstream file(filename);

        for (int it = 1; it <= itmax; it++) {
            matrix_vector(A, x, y);
            matrix_vector(A, x, Ax);

            double numerator = scalar_product(x, Ax);
            double denominator = scalar_product(x, x);

            lambda = numerator / denominator;

            gram_schmidt(y, xk, k);
            normalize(y);

            for (int i = 0; i < n; i++) {
                x[i] = y[i];
            }

            file << it << " " << lambda << '\n';

            if (fabs((lambda - lambda_old) / lambda_old) < eps) {
                break;
            }

            lambda_old = lambda;
        }

        file.close();

        lambda_k[k] = lambda;

        for (int i = 0; i < n; i++) {
            xk[k][i] = x[i];
        }
    }

    ofstream file("eigenpairs.txt");
    file << fixed;

    for (int k = 0; k < n; k++) {
        file << "lambda_" << k << " = " << lambda_k[k] << '\n';
        file << "x_" << k << " = ";

        for (int i = 0; i < n; i++) {
            file << xk[k][i];

            if (i < n - 1) {
                file << " ";
            }
        }
        file << "\n\n";
    }

    file.close();

    for (int k = 0; k < n; k++) {
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                A_num[i][j] += lambda_k[k] * xk[k][i] * xk[k][j];
                A_inv_num[i][j] += (1.0 / lambda_k[k]) * xk[k][i] * xk[k][j];
            }
        }
    }

    save_matrix(A_num, "A_num.txt");
    save_matrix(A_inv_num, "A_inv_num.txt");

    cout << "Eigenvalues:" << endl;
    for (int k = 0; k < n; k++) {
        cout << "lambda_" << k << " = " << fixed << lambda_k[k] << endl;
    }
    cout << endl;

    print_matrix(A_num, "Reconstructed matrix A_num:");
    print_matrix(A_inv_num, "Reconstructed inverse of matrix A_inv_num:");

    return 0;
}