// SPDX-License-Identifier: Apache-2.0
// Independently integrated from MIT BEVFusion commit 326653dc; see ../NOTICE.
#include <cuda.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>

__global__ void forward_kernel(int d, int h, int w, int c, int intervals,
                               const float* __restrict__ values,
                               const int* __restrict__ geometry,
                               const int* __restrict__ starts,
                               const int* __restrict__ lengths,
                               float* __restrict__ output) {
  const int linear = blockIdx.x * blockDim.x + threadIdx.x;
  const int interval = linear / c;
  const int channel = linear % c;
  if (interval >= intervals) return;
  const int start = starts[interval];
  const int length = lengths[interval];
  const int* coordinate = geometry + start * 4;
  const float* input = values + start * c + channel;
  float sum = 0.0f;
  for (int index = 0; index < length; ++index) sum += input[index * c];
  const int offset = coordinate[3] * d * h * w * c +
                     coordinate[2] * h * w * c +
                     coordinate[1] * w * c + coordinate[0] * c + channel;
  output[offset] = sum;
}

__global__ void backward_kernel(int d, int h, int w, int c, int intervals,
                                const float* __restrict__ output_gradient,
                                const int* __restrict__ geometry,
                                const int* __restrict__ starts,
                                const int* __restrict__ lengths,
                                float* __restrict__ values_gradient) {
  const int linear = blockIdx.x * blockDim.x + threadIdx.x;
  const int interval = linear / c;
  const int channel = linear % c;
  if (interval >= intervals) return;
  const int start = starts[interval];
  const int length = lengths[interval];
  const int* coordinate = geometry + start * 4;
  const int offset = coordinate[3] * d * h * w * c +
                     coordinate[2] * h * w * c +
                     coordinate[1] * w * c + coordinate[0] * c + channel;
  const float value = output_gradient[offset];
  for (int index = 0; index < length; ++index)
    values_gradient[(start + index) * c + channel] = value;
}

void launch_bev_pool_forward(int, int d, int h, int w, int, int c,
                             int intervals, const float* values,
                             const int* geometry, const int* starts,
                             const int* lengths, float* output) {
  const int threads = 256;
  const int blocks = (intervals * c + threads - 1) / threads;
  forward_kernel<<<blocks, threads, 0, c10::cuda::getCurrentCUDAStream()>>>(
      d, h, w, c, intervals, values, geometry, starts, lengths, output);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void launch_bev_pool_backward(int, int d, int h, int w, int, int c,
                              int intervals, const float* output_gradient,
                              const int* geometry, const int* starts,
                              const int* lengths, float* values_gradient) {
  const int threads = 256;
  const int blocks = (intervals * c + threads - 1) / threads;
  backward_kernel<<<blocks, threads, 0, c10::cuda::getCurrentCUDAStream()>>>(
      d, h, w, c, intervals, output_gradient, geometry, starts, lengths,
      values_gradient);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
