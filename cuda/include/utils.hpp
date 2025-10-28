#pragma once
#include <vector>
#include <string>
#include <cstddef>

struct Dataset {
    // X: N x D (row-major), y: N
    std::vector<float> X;
    std::vector<float> y;
    int N;
    int D;
};

// Синтетика для линейной регрессии: y = X w_true + noise
Dataset make_synthetic_lr(int N, int D, unsigned seed = 42, float noise_std = 0.1f);

// CPU: MSE и градиент по w (grad размером D)
float mse_and_grad_cpu(const Dataset& data, const std::vector<float>& w, std::vector<float>& grad);

// CSV логгер (xs, ys1[, ys2])
void write_csv(const std::string& path,
               const std::vector<float>& xs,
               const std::vector<float>& ys1,
               const std::string& col_x,
               const std::string& col_y1,
               const std::vector<float>* ys2 = nullptr,
               const std::string& col_y2 = "");

// CSV логгер для сравнения по времени: две серии (CPU/GPU)
void write_csv_time2(const std::string& path,
                     const std::vector<float>& t_cpu,
                     const std::vector<float>& loss_cpu,
                     const std::vector<float>& t_gpu,
                     const std::vector<float>& loss_gpu,
                     const std::string& col_t_cpu = "time_cpu_ms",
                     const std::string& col_loss_cpu = "loss_cpu",
                     const std::string& col_t_gpu = "time_gpu_ms",
                     const std::string& col_loss_gpu = "loss_gpu");

// Простой CPU-таймер
struct CpuTimer {
    void tic();
    double toc_ms() const; // milliseconds
private:
    long long start_ns = 0;
};

// ---- CUDA helpers (реализованы в utils.cu) ----
void* cuda_malloc(size_t bytes);
void  cuda_free(void* ptr);
void  cuda_memcpy_h2d(void* dst, const void* src, size_t bytes);
void  cuda_memcpy_d2h(void* dst, const void* src, size_t bytes);
void  cuda_check_last_error(const char* tag);

// Проверка NaN/Inf на устройстве
bool device_has_nan_or_inf(const float* d_arr, int n);

// MSE и градиент по w на GPU (вызывает CUDA-ядра)
float mse_and_grad_cuda(const float* d_X, const float* d_y, int N, int D,
                        const float* d_w, float* d_grad);
