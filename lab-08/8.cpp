#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

using namespace std;

const double xmin = -5.0;
const double xmax = 5.0;
const double pi = M_PI;

double y(double x)
{
    return 1.0 / (1.0 + x * x);
}

double lagrange(int n, double x, vector<double>& x_nodes, vector<double>& y_nodes)
{
    double W = 0.0;

    for (int j = 0; j <= n; j++) {
        double alpha = 1.0;
        double beta = 1.0;

        for (int i = 0; i <= n; i++) {
            if (abs(i - j) > 0) {
                alpha *= (x - x_nodes[i]);
                beta *= (x_nodes[j] - x_nodes[i]);
            }
        }

        W += y_nodes[j] * alpha / beta;
    }

    return W;
}

void equidistant_nodes(int n, vector<double>& x_nodes, vector<double>& y_nodes)
{
    x_nodes.resize(n + 1);
    y_nodes.resize(n + 1);

    double delta = (xmax - xmin) / n;

    for (int i = 0; i <= n; i++) {
        x_nodes[i] = xmin + delta * i;
        y_nodes[i] = y(x_nodes[i]);
    }
}

void chebyshev_nodes(int n, vector<double>& x_nodes, vector<double>& y_nodes)
{
    x_nodes.resize(n + 1);
    y_nodes.resize(n + 1);

    for (int i = 0; i <= n; i++) {
        x_nodes[i] = (xmax - xmin) / 2.0 * cos((2.0 * i + 1.0) / (2.0 * n + 2.0) * pi) + (xmin + xmax) / 2.0;
        y_nodes[i] = y(x_nodes[i]);
    }
}

void save_interpolation(string filename, int n, vector<double>& x_nodes, vector<double>& y_nodes)
{
    ofstream file(filename);

    file << fixed;

    float delta = 0.01;
    for (double x = xmin; x <= xmax; x += delta) {
        double W = lagrange(n, x, x_nodes, y_nodes);

        file << x << " " << y(x) << " " << W << " " << y(x) - W << endl;
    }
    file.close();
}

void save_nodes(string filename, vector<double>& x_nodes, vector<double>& y_nodes)
{
    ofstream file(filename);

    file << fixed;

    for (int i = 0; i < x_nodes.size(); i++) {
        file << x_nodes[i] << " " << y_nodes[i] << endl;
    }
    file.close();
}

int main()
{
    vector<int> n_equidistant = { 5, 10, 20 };
    vector<int> n_chebyshev = { 20, 50 };

    vector<double> x_nodes;
    vector<double> y_nodes;

    for (int n : n_equidistant) {
        equidistant_nodes(n, x_nodes, y_nodes);

        string name = "equidistant_" + to_string(n);

        save_interpolation(name + ".txt", n, x_nodes, y_nodes);
        save_nodes(name + "_nodes.txt", x_nodes, y_nodes);
    }

    for (int n : n_chebyshev) {
        chebyshev_nodes(n, x_nodes, y_nodes);

        string name = "chebyshev_" + to_string(n);

        save_interpolation(name + ".txt", n, x_nodes, y_nodes);
        save_nodes(name + "_nodes.txt", x_nodes, y_nodes);
    }

    return 0;
}