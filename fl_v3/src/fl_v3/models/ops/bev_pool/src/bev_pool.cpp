// SPDX-License-Identifier: Apache-2.0
// Independently integrated from MIT BEVFusion commit 326653dc; see ../NOTICE.
#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>

void launch_bev_pool_forward(int b, int d, int h, int w, int n, int c,
                             int intervals, const float* values,
                             const int* geometry, const int* starts,
                             const int* lengths, float* output);
void launch_bev_pool_backward(int b, int d, int h, int w, int n, int c,
                              int intervals, const float* output_gradient,
                              const int* geometry, const int* starts,
                              const int* lengths, float* values_gradient);

namespace {
void check_common(const at::Tensor& values, const at::Tensor& geometry,
                  const at::Tensor& starts, const at::Tensor& lengths) {
  TORCH_CHECK(values.is_cuda() && geometry.is_cuda() && starts.is_cuda() && lengths.is_cuda(),
              "BEV pooling tensors must be CUDA");
  TORCH_CHECK(values.scalar_type() == at::kFloat, "values must be float32");
  TORCH_CHECK(geometry.scalar_type() == at::kInt && starts.scalar_type() == at::kInt &&
              lengths.scalar_type() == at::kInt, "geometry/intervals must be int32");
  TORCH_CHECK(values.is_contiguous() && geometry.is_contiguous() &&
              starts.is_contiguous() && lengths.is_contiguous(),
              "BEV pooling tensors must be contiguous");
  TORCH_CHECK(values.dim() == 2 && geometry.dim() == 2 &&
              geometry.size(0) == values.size(0) && geometry.size(1) == 4,
              "invalid BEV pooling values/geometry shape");
  TORCH_CHECK(starts.dim() == 1 && lengths.sizes() == starts.sizes(),
              "invalid BEV pooling intervals");
}
}  // namespace

at::Tensor forward(const at::Tensor& values, const at::Tensor& geometry,
                   const at::Tensor& starts, const at::Tensor& lengths,
                   int64_t b, int64_t d, int64_t h, int64_t w) {
  check_common(values, geometry, starts, lengths);
  const at::cuda::OptionalCUDAGuard guard(at::device_of(values));
  auto output = torch::zeros({b, d, h, w, values.size(1)}, values.options());
  launch_bev_pool_forward(static_cast<int>(b), static_cast<int>(d), static_cast<int>(h),
                          static_cast<int>(w), static_cast<int>(values.size(0)),
                          static_cast<int>(values.size(1)), static_cast<int>(starts.size(0)),
                          values.data_ptr<float>(), geometry.data_ptr<int>(),
                          starts.data_ptr<int>(), lengths.data_ptr<int>(),
                          output.data_ptr<float>());
  return output;
}

at::Tensor backward(const at::Tensor& output_gradient, const at::Tensor& geometry,
                    const at::Tensor& starts, const at::Tensor& lengths,
                    int64_t b, int64_t d, int64_t h, int64_t w) {
  TORCH_CHECK(output_gradient.is_cuda() && geometry.is_cuda() && starts.is_cuda() && lengths.is_cuda(),
              "BEV pooling tensors must be CUDA");
  TORCH_CHECK(output_gradient.scalar_type() == at::kFloat &&
              geometry.scalar_type() == at::kInt && starts.scalar_type() == at::kInt &&
              lengths.scalar_type() == at::kInt, "invalid BEV pooling backward dtypes");
  TORCH_CHECK(output_gradient.is_contiguous() && geometry.is_contiguous() &&
              starts.is_contiguous() && lengths.is_contiguous(),
              "BEV pooling backward tensors must be contiguous");
  TORCH_CHECK(geometry.dim() == 2 && geometry.size(1) == 4 && starts.dim() == 1 &&
              lengths.sizes() == starts.sizes(), "invalid BEV pooling backward metadata");
  TORCH_CHECK(output_gradient.dim() == 5 && output_gradient.size(0) == b &&
              output_gradient.size(1) == d && output_gradient.size(2) == h &&
              output_gradient.size(3) == w,
              "invalid BEV pooling output-gradient shape");
  const at::cuda::OptionalCUDAGuard guard(at::device_of(output_gradient));
  auto gradient = torch::zeros({geometry.size(0), output_gradient.size(-1)}, output_gradient.options());
  launch_bev_pool_backward(static_cast<int>(b), static_cast<int>(d), static_cast<int>(h),
                           static_cast<int>(w), static_cast<int>(geometry.size(0)),
                           static_cast<int>(output_gradient.size(-1)),
                           static_cast<int>(starts.size(0)), output_gradient.data_ptr<float>(),
                           geometry.data_ptr<int>(), starts.data_ptr<int>(), lengths.data_ptr<int>(),
                           gradient.data_ptr<float>());
  return gradient;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("forward", &forward, "S10 BEV pooling forward");
  module.def("backward", &backward, "S10 BEV pooling backward");
}
