#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

using namespace std;

const int n = 100;
const int m = 4;

const double xmin = -10.0;
const double xmax = 20.0;

const double eps_a = pow(10.0, -4.0);
const double eps_lambda = pow(10.0, -5.0);

const int Kmax = 1000;
const int Lmax = 100;

const double lambdaA0 = 0.0;
const double lambdaB0 = 0.1;
const double r = 0.618;

double p(double x, double* a)
{
    return a[0] * atan(a[1] * x + a[2]) + a[3];
}

double f(int n, double* x, double* y, double* w, double* a)
{
    double result = 0.0;

    for (int i = 0; i < n; i++) {
        double value = y[i] - p(x[i], a);

        result += w[i] * value * value;
    }

    return result;
}

void gradient_f(int n, double* x, double* y, double* w, double* a, double* g)
{
    for (int j = 0; j < m; j++) {
        g[j] = 0.0;
    }

    for (int i = 0; i < n; i++) {
        double t = a[1] * x[i] + a[2];
        double value = y[i] - p(x[i], a);

        double dp_da0 = atan(t);
        double dp_da1 = a[0] * x[i] / (t * t + 1.0);
        double dp_da2 = a[0] / (t * t + 1.0);
        double dp_da3 = 1.0;

        g[0] += -2.0 * w[i] * value * dp_da0;
        g[1] += -2.0 * w[i] * value * dp_da1;
        g[2] += -2.0 * w[i] * value * dp_da2;
        g[3] += -2.0 * w[i] * value * dp_da3;
    }
}

double scalar_product(int n, double* x, double* y)
{
    double result = 0.0;

    for (int i = 0; i < n; i++) {
        result += x[i] * y[i];
    }

    return result;
}

double norm(double* x, double* y)
{
    double result = 0.0;

    for (int i = 0; i < m; i++) {
        result += (x[i] - y[i]) * (x[i] - y[i]);
    }

    return sqrt(result);
}

double f_lambda(int n, double* x, double* y, double* w, double* a, double* u, double lambda)
{
    double b[m];

    for (int i = 0; i < m; i++) {
        b[i] = a[i] + lambda * u[i];
    }

    return f(n, x, y, w, b);
}

double golden_section(int n, double* x, double* y, double* w, double* a, double* u)
{
    double lambdaA = lambdaA0;
    double lambdaB = lambdaB0;

    double lambda = 0.0;
    double s = fabs(lambdaB - lambdaA);

    int l = 0;

    do {
        l++;

        double lambda1 = lambdaA + r * r * (lambdaB - lambdaA);
        double lambda2 = lambdaA + r * (lambdaB - lambdaA);

        if (f_lambda(n, x, y, w, a, u, lambda1) > f_lambda(n, x, y, w, a, u, lambda2)) {
            lambdaA = lambda1;
        } else {
            lambdaB = lambda2;
        }

        lambda = (lambdaA + lambdaB) / 2.0;
        s = fabs(lambdaB - lambdaA);

    } while (s > eps_lambda && l < Lmax);

    return lambda;
}

void conjugate_gradients(int n, double* x, double* y, double* w, double* a, string version)
{
    double g[m], u[m];
    double gold[m];
    double aold[m];

    double s = 0.0;
    int k = 0;

    ofstream file("cg_" + version + ".txt");

    do {
        gradient_f(n, x, y, w, a, g);

        for (int i = 0; i < m; i++) {
            g[i] = -g[i];
        }

        if (k == 0) {
            for (int i = 0; i < m; i++) {
                u[i] = g[i];
            }
        } else {
            double gamma = 0.0;
            double denominator = scalar_product(m, gold, gold);

            if (version == "fr") {
                gamma = scalar_product(m, g, g) / denominator;
            }

            if (version == "pr") {
                double difference[m];

                for (int i = 0; i < m; i++) {
                    difference[i] = g[i] - gold[i];
                }
                gamma = scalar_product(m, difference, g) / denominator;
            }

            for (int i = 0; i < m; i++) {
                u[i] = g[i] + gamma * u[i];
            }
        }

        for (int i = 0; i < m; i++) {
            gold[i] = g[i];
        }

        k++;

        double lambda = golden_section(n, x, y, w, a, u);

        for (int i = 0; i < m; i++) {
            aold[i] = a[i];
        }

        for (int i = 0; i < m; i++) {
            a[i] += lambda * u[i];
        }

        s = norm(a, aold);

        file << k << " " << f(n, x, y, w, a) << " " << s << " " << lambda << " " << a[0] << " " << a[1] << " " << a[2] << " " << a[3] << endl;

    } while (s > eps_a && k < Kmax);

    file.close();
}

void save_fit(string filename, int n, double* x, double* y, double* a_exact, double* a_fit)
{
    ofstream file(filename);

    for (int i = 0; i < n; i++) {
        file << x[i] << " " << y[i] << " " << p(x[i], a_exact) << " " << p(x[i], a_fit) << endl;
    }
    file.close();
}

int main()
{
    srand(time(NULL));

    vector<double> x(n);
    vector<double> y(n);
    vector<double> w(n);

    double a_exact[m] = { 0.5, 10.0, -30.0, 0.7 };

    ofstream data("data.txt");

    for (int i = 0; i < n; i++) {
        double delta = (xmax - xmin) / (n - 1.0);
        double experimental_noise = 0.1 * (((double)rand() / RAND_MAX) - 0.5);

        x[i] = xmin + delta * i;
        y[i] = p(x[i], a_exact) + experimental_noise;
        w[i] = 1.0;

        data << x[i] << " " << y[i] << " " << p(x[i], a_exact) << endl;
    }
    data.close();

    double a_FR[m] = { 1.0, 1.0, 1.0, 1.0 };
    double a_PR[m] = { 1.0, 1.0, 1.0, 1.0 };

    // fr is Fletcher-Reves and pr is Polak-Riberie
    conjugate_gradients(n, x.data(), y.data(), w.data(), a_FR, "fr");
    conjugate_gradients(n, x.data(), y.data(), w.data(), a_PR, "pr");

    save_fit("fit_fr.txt", n, x.data(), y.data(), a_exact, a_FR);
    save_fit("fit_pr.txt", n, x.data(), y.data(), a_exact, a_PR);

    return 0;
}