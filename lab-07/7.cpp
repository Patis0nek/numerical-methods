#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>

using namespace std;

double f(double x)
{
    return (x - 1.0) * pow(x - 3.0, 6);
}

double df(double x)
{
    return (7.0 * x - 9.0) * pow(x - 3.0, 5);
}

void bissection_method(double xa, double xb, int itmax, double eps, const string& file_name)
{
    ofstream file(file_name);
    file << fixed;

    int k = 0;
    double xc, delta;

    do {
        k++;

        xc = 0.5 * (xa + xb);

        if (f(xa) * f(xc) < 0.0) {
            xb = xc;
        } else {
            xa = xc;
        }

        delta = fabs(xa - xb);

        file << k << " " << xc << " " << delta << "\n";
    } while (k < itmax && delta > eps);
    file.close();

    cout << "Bissection" << endl;
    cout << "x = " << xc << ", delta = " << delta << ", iterations = " << k << "\n\n";
}

void secant_method(double x0, double x1, int itmax, double eps, const string& file_name)
{
    ofstream file(file_name);
    file << fixed;

    int k = 0;
    double x2, delta;

    do {
        double f0 = f(x0);
        double f1 = f(x1);

        k++;

        x2 = x1 - f1 * ((x1 - x0) / (f1 - f0));

        x0 = x1;
        x1 = x2;

        delta = fabs(x1 - x0);

        file << k << " " << x1 << " " << delta << "\n";
    } while (k < itmax && delta > eps);
    file.close();

    cout << "Secant" << endl;
    cout << "x = " << x1 << ", delta = " << delta << ", iterations = " << k << "\n\n";
}

void newton_method(double x, int itmax, double eps, const string& file_name)
{
    ofstream file(file_name);
    file << fixed;

    int k = 0;
    double x1, delta;

    do {
        k++;

        x1 = x - (f(x) / df(x));
        delta = fabs(x - x1);

        file << k << " " << x1 << " " << delta << "\n";

        x = x1;

    } while (k < itmax && delta > eps);
    file.close();

    cout << "Newton-Raphson" << endl;
    cout << "x = " << x1 << ", delta = " << delta << ", iterations = " << k << "\n\n";
}

int main()
{
    const double eps = pow(10, -12);
    const int itmax_root1 = 30;

    cout << "root (x = 1.0)" << endl;
    bissection_method(0.1, 1.5, itmax_root1, eps, "root1_bissection.txt");
    secant_method(0.1, 0.5, itmax_root1, eps, "root1_secant.txt");
    newton_method(0.1, itmax_root1, eps, "root1_newton.txt");

    const int itmax_root3 = 300;

    cout << "root (x = 3.0)" << endl;
    secant_method(4.0, 6.0, itmax_root3, eps, "root3_secant.txt");
    newton_method(6.0, itmax_root3, eps, "root3_newton.txt");

    return 0;
}