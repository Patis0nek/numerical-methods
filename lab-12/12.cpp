#include <cmath>
#include <complex>
#include <cstdlib>
#include <fftw3.h>
#include <fstream>
#include <iostream>
#include <vector>

using namespace std;

const double L = 4.0;
const double alpha = 0.5;

double fpure(double x)
{
    double omega = (2.0 * M_PI) / L;

    return sin(5.0 * omega * x) + sin(15.0 * omega * x) + sin(25.0 * omega * x);
}

double random_noise()
{
    return 10.0 * (((double)rand() / (double)RAND_MAX) - 0.5);
}

void solve(int n)
{
    double dx = L / n;

    vector<complex<double>> input(n);
    vector<complex<double>> output(n);
    vector<complex<double>> filtered(n);
    vector<complex<double>> inverse(n);

    string suffix = "N" + to_string(n);
    ofstream noised("fnoised_" + suffix + ".txt");

    for (int i = 0; i < n; i++) {
        double x = dx * i;
        double f = fpure(x) + random_noise();

        input[i] = complex<double>(f, 0.0);
        noised << x << " " << f << " " << fpure(x) << endl;
    }
    noised.close();

    fftw_plan plan_forward = fftw_plan_dft_1d(n, (fftw_complex*)&input[0], (fftw_complex*)&output[0], FFTW_FORWARD, FFTW_ESTIMATE);

    fftw_execute(plan_forward);
    fftw_destroy_plan(plan_forward);

    ofstream re("re_" + suffix + ".txt");
    ofstream im("im_" + suffix + ".txt");

    for (int k = 0; k < n; k++) {
        re << k << " " << real(output[k]) << endl;
        im << k << " " << imag(output[k]) << endl;
    }
    re.close();
    im.close();

    double pmax = 0.0;

    for (int k = 0; k < n; k++) {
        if (abs(output[k]) * abs(output[k]) > pmax) {
            pmax = abs(output[k]) * abs(output[k]);
        }
    }

    for (int k = 0; k < n; k++) {
        if (abs(output[k]) * abs(output[k]) >= alpha * pmax) {
            filtered[k] = output[k];
        } else {
            filtered[k] = complex<double>(0.0, 0.0);
        }
    }

    fftw_plan plan_backward = fftw_plan_dft_1d(n, (fftw_complex*)&filtered[0], (fftw_complex*)&inverse[0], FFTW_BACKWARD, FFTW_ESTIMATE);

    fftw_execute(plan_backward);
    fftw_destroy_plan(plan_backward);

    ofstream denoised("fdenoised_" + suffix + ".txt");

    for (int i = 0; i < n; i++) {
        double x = dx * i;
        inverse[i] /= n;

        denoised << x << " " << real(inverse[i]) << " " << fpure(x) << endl;
    }
    denoised.close();
}

int main()
{
    solve(pow(2, 10));
    solve(pow(2, 12));

    return 0;
}