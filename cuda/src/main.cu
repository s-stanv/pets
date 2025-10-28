#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>
#include <cuda_runtime.h>

#include "utils.hpp"
#include "optimizers.cuh"

// struct OptimizerConfig { float lr, beta1, beta2, eps, grad_clip; int max_iters; };
// Уменьшены значения по умолчанию, чтобы запуск завершался быстрее.
OptimizerConfig cfg_sgd  { 1e-2f, 0.9f, 0.999f, 1e-8f, 0.0f, 200 };
OptimizerConfig cfg_rms  { 1e-2f, 0.9f, 0.99f,  1e-8f, 0.0f, 200 };
// Adam обычно требует меньший lr. Увеличим итерации для уверенной сходимости.
// Adam: немного меньший lr и слегка больше итераций, чтобы дойти до малых лоссов без чрезмерного роста времени
OptimizerConfig cfg_adam { 5e-3f, 0.9f, 0.999f, 1e-8f, 0.0f, 800 };

int main() {
    // --- Данные ---
    const int N = 20000;  // samples
    const int D = 128;    // features
    auto data = make_synthetic_lr(N, D, 42, 0.1f);

    // --- CPU baseline ---
    std::vector<float> w0(D, 0.f), w1(D, 0.f), w2(D, 0.f);
    std::vector<float> g(D, 0.f), m(D, 0.f), v(D, 0.f);

    std::vector<float> t_sgd_cpu, loss_sgd_cpu;
    std::vector<float> t_rms_cpu, loss_rms_cpu;
    std::vector<float> t_adam_cpu, loss_adam_cpu;

    // SGD CPU
    {
        auto w = w0; std::fill(m.begin(), m.end(), 0.f); std::fill(v.begin(), v.end(), 0.f);
        CpuTimer t; t.tic();
        for (int it = 1; it <= cfg_sgd.max_iters; ++it) {
            float loss = mse_and_grad_cpu(data, w, g);
            grad_clip_global_cpu(g.data(), D, cfg_sgd.grad_clip);
            sgd_cpu(w.data(), g.data(), D, cfg_sgd.lr);
            if (it % 10 == 0) { t_sgd_cpu.push_back((float)t.toc_ms()); loss_sgd_cpu.push_back(loss); }
            if (it % 50 == 0) std::cout << "[CPU][SGD] iter=" << it << ", loss=" << loss << "\n";
        }
        double ms = t.toc_ms();
        std::cout << "CPU SGD time (ms): " << ms << "\n";
    }

    // RMSProp CPU
    {
        auto w = w1; std::fill(m.begin(), m.end(), 0.f); std::fill(v.begin(), v.end(), 0.f);
        CpuTimer t; t.tic();
        for (int it = 1; it <= cfg_rms.max_iters; ++it) {
            float loss = mse_and_grad_cpu(data, w, g);
            grad_clip_global_cpu(g.data(), D, cfg_rms.grad_clip);
            rmsprop_cpu(w.data(), g.data(), v.data(), D, cfg_rms.lr, cfg_rms.beta2, cfg_rms.eps);
            if (it % 10 == 0) { t_rms_cpu.push_back((float)t.toc_ms()); loss_rms_cpu.push_back(loss); }
            if (it % 50 == 0) std::cout << "[CPU][RMSProp] iter=" << it << ", loss=" << loss << "\n";
        }
        double ms = t.toc_ms();
        std::cout << "CPU RMSProp time (ms): " << ms << "\n";
    }

    // Adam CPU
    {
        auto w = w2; std::fill(m.begin(), m.end(), 0.f); std::fill(v.begin(), v.end(), 0.f);
        CpuTimer t; t.tic();
        for (int it = 1; it <= cfg_adam.max_iters; ++it) {
            float loss = mse_and_grad_cpu(data, w, g);
            grad_clip_global_cpu(g.data(), D, cfg_adam.grad_clip);
            adam_cpu(w.data(), g.data(), m.data(), v.data(), D,
                     cfg_adam.lr, cfg_adam.beta1, cfg_adam.beta2, cfg_adam.eps, it);
            if (it % 10 == 0) { t_adam_cpu.push_back((float)t.toc_ms()); loss_adam_cpu.push_back(loss); }
            if (it % 50 == 0) std::cout << "[CPU][Adam] iter=" << it << ", loss=" << loss << "\n";
        }
        double ms = t.toc_ms();
        std::cout << "CPU Adam time (ms): " << ms << "\n";
    }

    // --- GPU ---
    float *d_X = (float*)cuda_malloc(sizeof(float) * N * D);
    float *d_y = (float*)cuda_malloc(sizeof(float) * N);
    cuda_memcpy_h2d(d_X, data.X.data(), sizeof(float) * N * D);
    cuda_memcpy_h2d(d_y, data.y.data(), sizeof(float) * N);

    float *d_w = (float*)cuda_malloc(sizeof(float) * D);
    float *d_grad = (float*)cuda_malloc(sizeof(float) * D);
    float *d_m = (float*)cuda_malloc(sizeof(float) * D);
    float *d_v = (float*)cuda_malloc(sizeof(float) * D);

    std::vector<float> t_sgd_gpu, loss_sgd_gpu;
    std::vector<float> t_rms_gpu, loss_rms_gpu;
    std::vector<float> t_adam_gpu, loss_adam_gpu;

    // SGD CUDA
    {
        std::vector<float> h_w(D, 0.f), zero(D, 0.f);
        cuda_memcpy_h2d(d_w, h_w.data(), sizeof(float) * D);
        cuda_memcpy_h2d(d_m, zero.data(), sizeof(float) * D);
        cuda_memcpy_h2d(d_v, zero.data(), sizeof(float) * D);
        cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
        cudaEventRecord(e0);
        CpuTimer tgpu; tgpu.tic();
        for (int it = 1; it <= cfg_sgd.max_iters; ++it) {
            float loss = mse_and_grad_cuda(d_X, d_y, N, D, d_w, d_grad);
            grad_clip_global_cuda(d_grad, D, cfg_sgd.grad_clip);
            if (device_has_nan_or_inf(d_grad, D) || device_has_nan_or_inf(d_w, D)) {
                std::cerr << "[NaN-trap] SGD detected NaN/Inf at iter " << it << "\n";
                break;
            }
            sgd_cuda(d_w, d_grad, D, cfg_sgd.lr);
            if (it % 10 == 0) { t_sgd_gpu.push_back((float)tgpu.toc_ms()); loss_sgd_gpu.push_back(loss); }
            if (it % 50 == 0) std::cout << "[GPU][SGD] iter=" << it << ", loss=" << loss << "\n";
        }
        cudaEventRecord(e1); cudaEventSynchronize(e1);
        float ms = 0; cudaEventElapsedTime(&ms, e0, e1);
        std::cout << "GPU SGD time (ms): " << ms << "\n";
        cudaEventDestroy(e0); cudaEventDestroy(e1);
    }

    // RMSProp CUDA
    {
        std::vector<float> h_w(D, 0.f), zero(D, 0.f);
        cuda_memcpy_h2d(d_w, h_w.data(), sizeof(float) * D);
        cuda_memcpy_h2d(d_m, zero.data(), sizeof(float) * D);
        cuda_memcpy_h2d(d_v, zero.data(), sizeof(float) * D);
        cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
        cudaEventRecord(e0);
        CpuTimer tgpu; tgpu.tic();
        for (int it = 1; it <= cfg_rms.max_iters; ++it) {
            float loss = mse_and_grad_cuda(d_X, d_y, N, D, d_w, d_grad);
            grad_clip_global_cuda(d_grad, D, cfg_rms.grad_clip);
            if (device_has_nan_or_inf(d_grad, D) || device_has_nan_or_inf(d_w, D)) {
                std::cerr << "[NaN-trap] RMSProp detected NaN/Inf at iter " << it << "\n";
                break;
            }
            rmsprop_cuda(d_w, d_grad, d_v, D, cfg_rms.lr, cfg_rms.beta2, cfg_rms.eps);
            if (it % 10 == 0) { t_rms_gpu.push_back((float)tgpu.toc_ms()); loss_rms_gpu.push_back(loss); }
            if (it % 50 == 0) std::cout << "[GPU][RMSProp] iter=" << it << ", loss=" << loss << "\n";
        }
        cudaEventRecord(e1); cudaEventSynchronize(e1);
        float ms = 0; cudaEventElapsedTime(&ms, e0, e1);
        std::cout << "GPU RMSProp time (ms): " << ms << "\n";
        cudaEventDestroy(e0); cudaEventDestroy(e1);
    }

    // Adam CUDA
    {
        std::vector<float> h_w(D, 0.f), zero(D, 0.f);
        cuda_memcpy_h2d(d_w, h_w.data(), sizeof(float) * D);
        cuda_memcpy_h2d(d_m, zero.data(), sizeof(float) * D);
        cuda_memcpy_h2d(d_v, zero.data(), sizeof(float) * D);
        cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
        cudaEventRecord(e0);
        CpuTimer tgpu; tgpu.tic();
        for (int it = 1; it <= cfg_adam.max_iters; ++it) {
            float loss = mse_and_grad_cuda(d_X, d_y, N, D, d_w, d_grad);
            grad_clip_global_cuda(d_grad, D, cfg_adam.grad_clip);
            if (device_has_nan_or_inf(d_grad, D) || device_has_nan_or_inf(d_w, D)) {
                std::cerr << "[NaN-trap] Adam detected NaN/Inf at iter " << it << "\n";
                break;
            }
            adam_cuda(d_w, d_grad, d_m, d_v, D, cfg_adam.lr, cfg_adam.beta1, cfg_adam.beta2, cfg_adam.eps, it);
            if (it % 10 == 0) { t_adam_gpu.push_back((float)tgpu.toc_ms()); loss_adam_gpu.push_back(loss); }
            if (it % 50 == 0) std::cout << "[GPU][Adam] iter=" << it << ", loss=" << loss << "\n";
        }
        cudaEventRecord(e1); cudaEventSynchronize(e1);
        float ms = 0; cudaEventElapsedTime(&ms, e0, e1);
        std::cout << "GPU Adam time (ms): " << ms << "\n";
        cudaEventDestroy(e0); cudaEventDestroy(e1);
    }

    // --- Логи ---
#ifdef _WIN32
    system("if not exist logs mkdir logs");
#else
    system("mkdir -p logs");
#endif
    write_csv_time2("logs/sgd.csv",     t_sgd_cpu,  loss_sgd_cpu,  t_sgd_gpu,  loss_sgd_gpu);
    write_csv_time2("logs/rmsprop.csv", t_rms_cpu,  loss_rms_cpu,  t_rms_gpu,  loss_rms_gpu);
    write_csv_time2("logs/adam.csv",    t_adam_cpu, loss_adam_cpu, t_adam_gpu, loss_adam_gpu);

    cuda_free(d_X); cuda_free(d_y); cuda_free(d_w); cuda_free(d_grad); cuda_free(d_m); cuda_free(d_v);

    std::cout << "Done. Logs saved to ./logs/*.csv\n";
    return 0;
}
