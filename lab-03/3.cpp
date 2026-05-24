#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <openblas/lapacke.h>
#include <vector>

using namespace std;

double exact_wavefunction(int k, double x)
{
    double a = 0.0;
    double b = 1.0;
    double c = 1.0;

    for (int i = 1; i <= k; i++) {
        c = 2.0 * x * b - 2.0 * (i - 1) * a;
        a = b;
        b = c;
    }

    return c * exp(-0.5 * x * x);
}

void normalize_vector(vector<double>& values, double dx)
{
    double normSquared = 0.0;

    for (double value : values) {
        normSquared += value * value;
    }

    double norm = sqrt(normSquared * dx);

    if (norm == 0.0) {
        return;
    }

    for (double& value : values) {
        value /= norm;
    }
}

int main()
{
    int n = 100;
    double xMax = 10.0;
    double dx = 2.0 * xMax / (n - 1);

    vector<double> grid(n);

    for (int i = 0; i < n; i++) {
        grid[i] = -xMax + i * dx;
    }

    vector<double> mainDiagonal(n);
    vector<double> subDiagonal(n - 1);

    for (int i = 0; i < n; i++) {
        mainDiagonal[i] = 1.0 / (dx * dx) + 0.5 * grid[i] * grid[i];
    }

    for (int i = 0; i < n - 1; i++) {
        subDiagonal[i] = -1.0 / (2.0 * dx * dx);
    }

    vector<double> numericalVec(n * n);

    int INFO = LAPACKE_dstev(LAPACK_COL_MAJOR, 'V', n, mainDiagonal.data(), subDiagonal.data(), numericalVec.data(), n);

    if (INFO != 0) {
        cout << INFO << endl;

        return 0;
    }

    for (int j = 0; j < n; j++) {
        vector<double> eigenvector(n);

        for (int i = 0; i < n; i++) {
            eigenvector[i] = numericalVec[i + j * n];
        }

        normalize_vector(eigenvector, dx);

        for (int i = 0; i < n; i++) {
            numericalVec[i + j * n] = eigenvector[i];
        }
    }

    ofstream numericalValFile("numerical_eigenvalues.txt");
    numericalValFile << fixed << setprecision(4);
    for (int j = 0; j < n; j++) {
        numericalValFile << j << " " << mainDiagonal[j] << "\n";
    }
    numericalValFile.close();

    ofstream numericalVecFile("numerical_eigenvectors.txt");
    numericalVecFile << fixed << setprecision(4);
    for (int i = 0; i < n; i++) {
        numericalVecFile << grid[i];

        for (int j = 0; j < n; j++) {
            numericalVecFile << " " << numericalVec[i + j * n];
        }
        numericalVecFile << "\n";
    }
    numericalVecFile.close();

    vector<double> exactVal(n);
    for (int j = 0; j < n; j++) {
        exactVal[j] = j + 0.5;
    }

    vector<vector<double>> exactVec(n, vector<double>(n));

    for (int j = 0; j < n; j++) {
        for (int i = 0; i < n; i++) {
            exactVec[j][i] = exact_wavefunction(j, grid[i]);
        }

        normalize_vector(exactVec[j], dx);

        double overlap = 0.0;
        for (int i = 0; i < n; i++) {
            overlap += exactVec[j][i] * numericalVec[i + j * n];
        }
        overlap *= dx;

        if (overlap < 0.0) {
            for (int i = 0; i < n; i++) {
                exactVec[j][i] = -exactVec[j][i];
            }
        }
    }

    ofstream exactValFile("exact_eigenvalues.txt");
    exactValFile << fixed << setprecision(4);
    for (int j = 0; j < n; j++) {
        exactValFile << j << " " << exactVal[j] << "\n";
    }
    exactValFile.close();

    ofstream exactVecFile("exact_eigenvectors.txt");
    exactVecFile << fixed << setprecision(4);
    for (int i = 0; i < n; i++) {
        exactVecFile << grid[i];
        for (int j = 0; j < n; j++) {
            exactVecFile << " " << exactVec[j][i];
        }
        exactVecFile << "\n";
    }
    exactVecFile.close();

    return 0;
}