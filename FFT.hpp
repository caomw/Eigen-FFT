// Written by Ashton Fagg
// License: Do whatever you please, I take no responsibility.

#ifndef FFT_HPP
#define FFT_HPP

#include <cmath>
#include <complex>

#include <Eigen/Dense>
#include <unsupported/Eigen/FFT>

using Eigen::Matrix;
using Eigen::Dynamic;
using std::complex;



template<typename T> class FFT {
public:
    FFT() {
    }
    
    
    // This performs a 1D FFT on a matrix of type T
    Matrix<complex<T>, Dynamic, Dynamic> fwd1D(const Matrix<T, Dynamic, Dynamic> &X) {
        const long m = X.rows(), n = X.cols();
        
        Matrix<complex<T>, Eigen::Dynamic, Eigen::Dynamic> Y(m,n);
        
        for (long i = 0; i < n; ++i) {
            Matrix<complex<T>, Eigen::Dynamic, 1> Xi = X.col(i);
            Y.col(i) = this->tf.fwd(Xi);
        }
        
        return Y;
    }
    
    
    // Overloaded to ensure we can perform the same on complex matrices as well
    Matrix<complex<T>, Dynamic, Dynamic> fwd1D(const Matrix<complex<T>, Dynamic, Dynamic> &X) {
        const long m = X.rows(), n = X.cols();
        
        Matrix<complex<T>, Eigen::Dynamic, Eigen::Dynamic> Y(m,n);
        
        for (long i = 0; i < n; ++i) {
            Matrix<complex<T>, Eigen::Dynamic, 1> Xi = X.col(i);
            Y.col(i) = this->tf.fwd(Xi);
        }
        
        return Y;
    }
    
    // This performs an inverse 1D FFT on a matrix of complex<T>
    // Note that the returned matrix is still a matrix of complex<T>
    Matrix<complex<T>, Dynamic, Dynamic> inv1D(const Matrix<complex<T>, Dynamic, Dynamic> &Y) {
        const long m = Y.rows(), n = Y.cols();
        
        Matrix<complex<T>, Dynamic, Dynamic> X(m,n);
        
        for (long i = 0; i < n; ++i) {
            Matrix<complex<T>, Eigen::Dynamic, 1> Yi = Y.col(i);
            X.col(i) = this->tf.inv(Yi);
        }
        
        return X;
    }
    
    // Wraps the 1D case to form the 2D fft
    Matrix<complex<T>, Dynamic, Dynamic> fwd2D(const Matrix<complex<T>, Dynamic, Dynamic> &X) {
        Matrix<complex<T>, Eigen::Dynamic, Eigen::Dynamic> t = this->fwd1D(X).transpose();
        
        return this->fwd1D( t ).transpose();
    }
    
    // Wraps the 1D case to form the 2D inverse FFT
    Matrix<complex<T>, Dynamic, Dynamic> inv2D(const Matrix<complex<T>, Dynamic, Dynamic> &Y) {
        return this->inv1D( this->inv1D(Y).transpose() ).transpose();
    }
    
private:
    Eigen::FFT<T> tf; // Underlying eigen FFT to talk to FFTW
};
#endif
