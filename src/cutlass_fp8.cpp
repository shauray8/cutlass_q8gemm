#include <torch/extension.h>
#include <cuda_fp8.h>
#include <cutlass/cutlass.h>
#include <vector>

torch::Tensor cutlass_fp8gemm_batched_kernel(
    torch::Tensor a,
    torch::Tensor b,
    int m, int n, int k,
    bool trans_a,
    bool trans_b,
    int lda, int ldb, int ldc,
    int out_h, int out_w,
    bool weight_is_a,
    torch::optional<torch::Tensor> bias,
    torch::Tensor input_scale,
    torch::Tensor weight_scale,
    const std::string& epilogue_str
);

torch::Tensor cutlass_fp8gemm_batched_impl_custom(
    torch::Tensor a,
    torch::Tensor b,
    int m, int n, int k,
    int lda, int ldb, int ldc,
    int out_h, int out_w,
    bool trans_a,
    bool trans_b,
    bool weight_is_a,
    torch::optional<torch::Tensor> bias,
    torch::Tensor input_scale,
    torch::Tensor weight_scale,
    const std::string& epilogue_str
) {
    return cutlass_fp8gemm_batched_kernel(
        a, b, m, n ,k , trans_a, trans_b, 
        lda, ldb, ldc, out_h, out_w, weight_is_a,
        bias, input_scale, weight_scale, epilogue_str
    );
}

PYBIND11_MODULE(TORCH_EXTENSTION_NAME, m){
    m.def("cutlass_fp8gemm_batched_impl_custom",
        &cutlass_fp8gemm_batched_impl_custom,
        "CUTLASS FP8 GEMM IMPLIMENTATION");
    /*
    m.def("convert_to_fp8",
        &convert_to_fp8,
        "Convert tensor to fp8 E4M3 format");
    */
}
