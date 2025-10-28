#pragma once
#include <cstddef>

struct OptimizerConfig {
    float lr = 1e-2f;     // Learning rate
    float beta1 = 0.9f;   // Adam
    float beta2 = 0.999f; // Adam/RMSProp
    float eps = 1e-8f;    // Numerical stability
    float grad_clip = 0.0f; // 0 = off
    int   max_iters = 2000;
};

// CPU
void sgd_cpu(float* w, float* grad, size_t D, float lr);
void rmsprop_cpu(float* w, float* grad, float* v, size_t D, float lr, float beta2, float eps);
void adam_cpu(float* w, float* grad, float* m, float* v, size_t D,
              float lr, float beta1, float beta2, float eps, int t);

void grad_clip_global_cpu(float* grad, size_t D, float clip_norm);

// GPU (launchers)
void sgd_cuda(float* d_w, const float* d_grad, size_t D, float lr);
void rmsprop_cuda(float* d_w, const float* d_grad, float* d_v, size_t D,
                  float lr, float beta2, float eps);
void adam_cuda(float* d_w, const float* d_grad, float* d_m, float* d_v, size_t D,
               float lr, float beta1, float beta2, float eps, int t);

void grad_clip_global_cuda(float* d_grad, size_t D, float clip_norm);
