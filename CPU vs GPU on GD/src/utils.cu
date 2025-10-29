#include "utils.hpp"

#include <random>
#include <fstream>
#include <chrono>
#include <cmath>
#include <stdexcept>
#include <iostream>

// CUDA runtime
#include <cuda_runtime.h>

// ------------------- Хост-утилиты -------------------
Dataset make_synthetic_lr(int N, int D, unsigned seed, float noise_std) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> wdist(0.f, 1.f);
    std::normal_distribution<float> xdist(0.f, 1.f);
    std::normal_distribution<float> ndist(0.f, noise_std);

    std::vector<float> X(N * D), y(N), w_true(D);
    for (int j = 0; j < D; ++j) w_true[j] = wdist(rng);
    for (int i = 0; i < N; ++i) {
        float yi = 0.f;
        for (int j = 0; j < D; ++j) {
            float x = xdist(rng);
            X[i*D + j] = x;
            yi += x * w_true[j];
        }
        y[i] = yi + ndist(rng);
    }
    return {std::move(X), std::move(y), N, D};
}

float mse_and_grad_cpu(const Dataset& data, const std::vector<float>& w, std::vector<float>& grad) {
    const int N = data.N, D = data.D;
    std::fill(grad.begin(), grad.end(), 0.f);
    float loss = 0.f;
    for (int i = 0; i < N; ++i) {
        const float* xi = &data.X[i * D];
        float pred = 0.f;
        for (int j = 0; j < D; ++j) pred += xi[j] * w[j];
        float err = pred - data.y[i];
        loss += err * err;
        for (int j = 0; j < D; ++j) grad[j] += (2.f / N) * err * xi[j];
    }
    return loss / N;
}

void write_csv(const std::string& path,
               const std::vector<float>& xs,
               const std::vector<float>& ys1,
               const std::string& col_x,
               const std::string& col_y1,
               const std::vector<float>* ys2,
               const std::string& col_y2) {
    std::ofstream f(path);
    f << col_x << "," << col_y1;
    if (ys2) f << "," << col_y2;
    f << "\n";
    for (size_t i = 0; i < xs.size(); ++i) {
        f << xs[i] << "," << ys1[i];
        if (ys2) f << "," << (*ys2)[i];
        f << "\n";
    }
}

void write_csv_time2(const std::string& path,
                     const std::vector<float>& t_cpu,
                     const std::vector<float>& loss_cpu,
                     const std::vector<float>& t_gpu,
                     const std::vector<float>& loss_gpu,
                     const std::string& col_t_cpu,
                     const std::string& col_loss_cpu,
                     const std::string& col_t_gpu,
                     const std::string& col_loss_gpu) {
    std::ofstream f(path);
    f << col_t_cpu << "," << col_loss_cpu << "," << col_t_gpu << "," << col_loss_gpu << "\n";
    size_t n = std::min(std::min(t_cpu.size(), loss_cpu.size()), std::min(t_gpu.size(), loss_gpu.size()));
    for (size_t i = 0; i < n; ++i) {
        f << t_cpu[i] << "," << loss_cpu[i] << "," << t_gpu[i] << "," << loss_gpu[i] << "\n";
    }
}

void CpuTimer::tic() {
    start_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::high_resolution_clock::now().time_since_epoch()).count();
}
double CpuTimer::toc_ms() const {
    long long end_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::high_resolution_clock::now().time_since_epoch()).count();
    return (end_ns - start_ns) / 1e6;
}

// ------------------- CUDA helpers -------------------
void* cuda_malloc(size_t bytes) { void* p = nullptr; cudaMalloc(&p, bytes); return p; }
void  cuda_free(void* ptr) { cudaFree(ptr); }
void  cuda_memcpy_h2d(void* dst, const void* src, size_t bytes) { cudaMemcpy(dst, src, bytes, cudaMemcpyHostToDevice); }
void  cuda_memcpy_d2h(void* dst, const void* src, size_t bytes) { cudaMemcpy(dst, src, bytes, cudaMemcpyDeviceToHost); }
void  cuda_check_last_error(const char* tag) {
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        std::cerr << "CUDA error after " << tag << ": " << cudaGetErrorString(e) << "\n";
        throw std::runtime_error("CUDA error");
    }
}

// NaN/Inf detector
__global__ void kernel_check_nan_inf(const float* __restrict__ a, int n, int* __restrict__ flag) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float v = a[idx];
        if (!isfinite(v)) atomicExch(flag, 1);
    }
}
bool device_has_nan_or_inf(const float* d_arr, int n) {
    int* d_flag = nullptr; int h_flag = 0;
    cudaMalloc(&d_flag, sizeof(int));
    cudaMemcpy(d_flag, &h_flag, sizeof(int), cudaMemcpyHostToDevice);
    int block = 256; int grid = (n + block - 1) / block;
    kernel_check_nan_inf<<<grid, block>>>(d_arr, n, d_flag);
    cudaDeviceSynchronize();
    cudaMemcpy(&h_flag, d_flag, sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d_flag);
    return h_flag != 0;
}

// ----------- MSE/grad CUDA -----------
__global__ void kernel_pred_err(const float* __restrict__ X, const float* __restrict__ y,
                                const float* __restrict__ w, float* __restrict__ errs,
                                int N, int D) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        const float* xi = X + i * D;
        float pred = 0.f;
        for (int j = 0; j < D; ++j) pred += xi[j] * w[j];
        errs[i] = pred - y[i];
    }
}

__global__ void kernel_grad_reduction(const float* __restrict__ X, const float* __restrict__ errs,
                                      float* __restrict__ grad, int N, int D) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j < D) {
        float acc = 0.f;
        for (int i = 0; i < N; ++i) acc += (2.f / N) * errs[i] * X[i * D + j];
        grad[j] = acc;
    }
}

float mse_and_grad_cuda(const float* d_X, const float* d_y, int N, int D,
                        const float* d_w, float* d_grad) {
    float* d_errs = (float*)cuda_malloc(sizeof(float) * N);

    int block = 256;
    int gridN = (N + block - 1) / block;
    kernel_pred_err<<<gridN, block>>>(d_X, d_y, d_w, d_errs, N, D);

    int gridD = (D + block - 1) / block;
    kernel_grad_reduction<<<gridD, block>>>(d_X, d_errs, d_grad, N, D);

    // loss = mean(err^2)
    std::vector<float> h_errs(N);
    cuda_memcpy_d2h(h_errs.data(), d_errs, sizeof(float) * N);
    float loss = 0.f;
    for (int i = 0; i < N; ++i) loss += h_errs[i] * h_errs[i];
    loss /= N;

    cuda_free(d_errs);
    cuda_check_last_error("mse_and_grad_cuda");
    return loss;
}
