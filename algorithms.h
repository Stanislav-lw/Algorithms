#ifndef NKALGORITHM_H
#define NKALGORITHM_H

#include <cmath>
#include <complex>
#include <valarray>
#include <vector>

namespace fourier {

using complex = std::complex<double>;
using ctd = double (*)(complex);
inline double realc(complex in) { return in.real(); }
inline double imagc(complex in) { return in.imag(); }
inline double normc(complex in) { return std::norm(in); }
inline double absc(complex in)  { return std::abs(in); }
inline double argc(complex in)  { return std::arg(in); }

namespace tools {
template<typename Functor>
struct y_combinate {
    y_combinate(Functor &&functor) : functor_(functor) {}
    Functor functor_;
    template<typename... Args>
    decltype(auto) operator()(Args&&...args) const {
        return functor_(std::ref(*this), std::forward<Args>(args)...);
    }
};
}

template <template <typename> class Vector, typename Type>
auto cFFT(const Vector<Type> &in, bool normalize = true)
{
    Vector<complex> complexv;
    std::copy(std::begin(in), std::end(in), std::back_inserter(complexv));
    int order = static_cast<int>(ceil(log2(in.size())));
    int base = static_cast<int>(ceil(pow(2, order)));
    int size = static_cast<int>(complexv.size());
    for (int i = size; i < base; ++i) complexv.push_back(0);
    std::valarray<complex> valarray(complexv.data(), base);
    auto fft = tools::y_combinate([](auto &&fft, std::valarray<complex> &v)->void {
        const std::uint32_t N = v.size();
        if (N <= 1) return;
        std::valarray<complex> even = v[std::slice(0, N/2, 2)];
        std::valarray<complex>  odd = v[std::slice(1, N/2, 2)];
        fft(even);
        fft(odd);
        for (std::uint32_t k = 0U; k < N/2; ++k) {
            complex t = std::polar(1.0, -2 * M_PI * k / N) * odd[k];
            v[k    ] = even[k] + t;
            v[k+N/2] = even[k] - t;
        }
    });
    fft(valarray);
    if (normalize) {
        double sqtb = 1.0/sqrt(base);
        valarray *= sqtb;
    }
    Vector<complex> cfft;
    std::copy(std::begin(valarray), std::end(valarray), std::back_inserter(cfft));
    return cfft;
}

namespace windows {

template <template <typename> class Vector, typename Type>
auto hann(const Vector<Type> &in)
{
    Vector<Type> out;
    const std::uint32_t size = in.size();
    for (std::uint32_t i = 0; i < size; ++i) {
        double multiplier = 0.5*(1.0 - cos((2.0*M_PI*i)/(size - 1.0)));
        out.push_back(in[i]*multiplier);
    }
    return out;
}


template <template <typename> class Vector, typename Type>
auto hamming(const Vector<Type> &in)
{
    Vector<Type> out;
    const std::uint32_t size = in.size();
    for (std::uint32_t i = 0; i < size; ++i) {
        double multiplier = 0.54 - 0.46*cos((2.0*M_PI*i)/(size - 1.0));
        out.push_back(in[i]*multiplier);
    }
    return out;
}


template <template <typename> class Vector, typename Type>
auto blackman(const Vector<Type> &in)
{
    Vector<Type> out;
    const std::uint32_t size = in.size();
    for (std::uint32_t i = 0; i < size; ++i) {
        double multiplier = 0.42 - 0.5*cos(2.0*M_PI*i/(size - 1.0)) + 0.08*cos(4.0*M_PI*i/(size - 1.0));
        out.push_back(in[i]*multiplier);
    }
    return out;
}

constexpr std::uint64_t factorial(int n)
{
    return n <= 1 ? 1 : (n * factorial(n - 1));
}

double besselZero(double x)
{
    double sum = 0.0;
    for (int k = 0; k < 8; ++k) {
        std::uint64_t f = factorial(k);
        double increase = std::pow((1.0/f)*std::pow(x*0.5, k), 2.0);
        double fraction = increase/sum;
        if (fraction < 0.001) break;
        sum += increase;
    }
    return sum;
};

template <template <typename > class Vector, typename Type>
auto kaiser(const Vector<Type> &in, double beta = 5.0)
{
    Vector<Type> out;
    const std::uint32_t size = in.size();
    for (std::uint32_t i = 0; i < size; ++i) {
        double multiplier = beta * sqrt(1.0 - pow((2.0*i)/(size - 1.0) - 1.0, 2.0));
        double fraction = besselZero(multiplier)/besselZero(beta);
        out.push_back(in[i]*fraction);
    }
    return out;
}
} // end window namespace
} // end fourier namespace


#endif // NKALGORITHM_H
