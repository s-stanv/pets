#include "optimizers.cuh"
#include <cmath>
#include <cuda_runtime.h>

// ---------------- CPU ----------------
void sgd_cpu(float* w, float* grad, size_t D, float lr) {
    for (size_t j = 0; j < D; ++j) w[j] -= lr * grad[j];
}

void rmsprop_cpu(float* w, float* grad, float* v, size_t D, float lr, float beta2, float eps) {
    for (size_t j = 0; j < D; ++j) {
        v[j] = beta2 * v[j] + (1 - beta2) * grad[j] * grad[j];
        w[j] -= lr * grad[j] / (std::sqrt(v[j]) + eps);
    }
}

void adam_cpu(float* w, float* grad, float* m, float* v, size_t D,
              float lr, float beta1, float beta2, float eps, int t) {
    for (size_t j = 0; j < D; ++j) {
        m[j] = beta1 * m[j] + (1 - beta1) * grad[j];
        v[j] = beta2 * v[j] + (1 - beta2) * grad[j] * grad[j];
        float mhat = m[j] / (1 - std::pow(beta1, t));
        float vhat = v[j] / (1 - std::pow(beta2, t));
        w[j] -= lr * mhat / (std::sqrt(vhat) + eps);
    }
}

void grad_clip_global_cpu(float* grad, size_t D, float clip_norm) {
    if (clip_norm <= 0) return;
    double norm2 = 0.0; for (size_t j = 0; j < D; ++j) norm2 += (double)grad[j] * grad[j];
    double n = std::sqrt(norm2);
    if (n > clip_norm && n > 0) {
        double s = clip_norm / n;
        for (size_t j = 0; j < D; ++j) grad[j] = (float)(grad[j] * s);
    }
}

// ---------------- CUDA ----------------
__global__ void k_sgd(float* w, const float* grad, size_t D, float lr) {
    size_t j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j < D) w[j] -= lr * grad[j];
}

__global__ void k_rmsprop(float* w, const float* grad, float* v, size_t D,
                          float lr, float beta2, float eps) {
    size_t j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j < D) {
        float vj = beta2 * v[j] + (1 - beta2) * grad[j] * grad[j];
        v[j] = vj;
        w[j] -= lr * grad[j] / (sqrtf(vj) + eps);
    }
}

__global__ void k_adam(float* w, const float* grad, float* m, float* v, size_t D,
                       float lr, float beta1, float beta2, float eps, int t) {
    size_t j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j < D) {
        float mj = beta1 * m[j] + (1 - beta1) * grad[j];
        float vj = beta2 * v[j] + (1 - beta2) * grad[j] * grad[j];
        m[j] = mj; v[j] = vj;
        float mhat = mj / (1.f - powf(beta1, t));
        float vhat = vj / (1.f - powf(beta2, t));
        w[j] -= lr * mhat / (sqrtf(vhat) + eps);
    }
}

__global__ void k_reduce_sum_squares(const float* grad, size_t D, double* out) {
    __shared__ double sh[256];
    size_t tid = threadIdx.x;
    size_t j = blockIdx.x * blockDim.x + tid;
    double v = 0.0;
    if (j < D) v = (double)grad[j] * grad[j];
    sh[tid] = v; __syncthreads();
    for (int s = blockDim.x/2; s > 0; s >>= 1) {
        if (tid < s) sh[tid] += sh[tid + s];
        __syncthreads();
    }
    if (tid == 0) atomicAdd(out, sh[0]);
}

__global__ void k_scale(float* grad, size_t D, float s) {
    size_t j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j < D) grad[j] *= s;
}

void grad_clip_global_cuda(float* d_grad, size_t D, float clip_norm) {
    if (clip_norm <= 0) return;
    double* d_acc = nullptr; cudaMalloc(&d_acc, sizeof(double));
    double zero = 0.0; cudaMemcpy(d_acc, &zero, sizeof(double), cudaMemcpyHostToDevice);
    int block = 256; int grid = (int)((D + block - 1) / block);
    k_reduce_sum_squares<<<grid, block>>>(d_grad, D, d_acc);
    cudaDeviceSynchronize();
    double h_acc = 0; cudaMemcpy(&h_acc, d_acc, sizeof(double), cudaMemcpyDeviceToHost);
    cudaFree(d_acc);
    double n = std::sqrt(h_acc);
    if (n > clip_norm && n > 0) {
        float s = (float)(clip_norm / n);
        k_scale<<<grid, block>>>(d_grad, D, s);
    }
}

void sgd_cuda(float* d_w, const float* d_grad, size_t D, float lr) {
    int block = 256; int grid = (int)((D + block - 1) / block);
    k_sgd<<<grid, block>>>(d_w, d_grad, D, lr);
}

void rmsprop_cuda(float* d_w, const float* d_grad, float* d_v, size_t D,
                  float lr, float beta2, float eps) {
    int block = 256; int grid = (int)((D + block - 1) / block);
    k_rmsprop<<<grid, block>>>(d_w, d_grad, d_v, D, lr, beta2, eps);
}

void adam_cuda(float* d_w, const float* d_grad, float* d_m, float* d_v, size_t D,
               float lr, float beta1, float beta2, float eps, int t) {
    int block = 256; int grid = (int)((D + block - 1) / block);
    k_adam<<<grid, block>>>(d_w, d_grad, d_m, d_v, D, lr, beta1, beta2, eps, t);
}
