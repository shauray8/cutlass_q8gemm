#include <torch/extension.h>
#include <cuda_fp8.h>
#include <cutlass/cutlass.h>
#include <cutlass/gemm/device/gemm_grouped.h>
#include <cutlass/gemm/kernel/default_gemm_grouped.h>
#include <cutlass/epilogue/thread/linear_combination.h>
#include "cutlass/gemm/kernel/gemm_grouped_per_group_scale.h"
#include "cutlass/gemm/kernel/default_gemm_grouped_per_group_scale.h"

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
    const std::string& epilogue_str) {
    
    using ElementA = cutlass::float_e4m3_t;
    using ElementB = cutlass::float_e4m3_t;
    using ElementOutput = cutlass::bfloat16_t;
    using ElementOutputFp8Accum = cutlass::float_e4m3_t;
    using ElementAccumulator = float;
    using ElementCompute = float;

    using LayoutA = cutlass::layout::RowMajor;
    using LayoutB = cutlass::layout::ColumnMajor;
    using LayoutC = cutlass::layout::RowMajor;

    using GemmConfig = cutlass::gemm::GemmShape<64, 128, 64>;
    using ThreadblockShape = cutlass::gemm::GemmShape<64, 32, 64>;
    using WarpShape = cutlass::gemm::GemmShape<16, 8, 32>;
    
    using EpilogueOp = typename cutlass::epilogue::thread::LinearCombination<
        ElementOutput,
        128 / cutlass::sizeof_bits<ElementOutput>::value,
        ElementAccumulator,
        ElementCompute>;

    using GemmKernel = typename cutlass::gemm::kernel::DefaultGemmGroupedPerGroupScale<
        ElementA, LayoutA, cutlass::ComplexTransform::kNone, 32,
        ElementB, LayoutB, cutlass::ComplexTransform::kNone, 32,
        ElementOutput, LayoutC,
        ElementAccumulator,
        cutlass::arch::OpClassTensorOp,
        cutlass::arch::Sm89,
        GemmConfig,
        ThreadblockShape,
        WarpShape,
        EpilogueOp,
        cutlass::gemm::threadblock::GemmBatchedIdentityThreadblockSwizzle,
        4>::GemmKernel;

    using Gemm = cutlass::gemm::device::GemmGrouped<GemmKernel>;

    int64_t batch_size = (a.dim() == 3) ? a.size(0) : b.size(0);
    if (out_h == -1) out_h = m;
    if (out_w == -1) out_w = n;
    
    auto output = torch::empty({batch_size, out_h, out_w}, 
        torch::TensorOptions()
            .dtype(torch::kBFloat16)
            .device(a.device()));

    auto outputFp8 = torch::empty({batch_size, out_h, out_w}, 
        torch::TensorOptions()
            .dtype(torch::kBFloat16) // not sure what to use here for fp8 accum
            .device(a.device()));

    float* input_scale_ptr = input_scale.data_ptr<float>();
    float* weight_scale_ptr = weight_scale.data_ptr<float>();

    typename Gemm::Arguments args{
        {m, n, k},
        {
            static_cast<ElementA*>(a.data_ptr()),
            lda == -1 ? (trans_a ? k : m) : lda
        },
        {
            static_cast<ElementB*>(b.data_ptr()),
            ldb == -1 ? (trans_b ? n : k) : ldb
        },
        {
            static_cast<ElementOutput*>(output.data_ptr()),
            ldc == -1 ? n : ldc
        },
        {
            bias.has_value() ? 
                static_cast<ElementOutput*>(bias.value().data_ptr()) : nullptr,
            bias.has_value() ? n : 0
        },
        input_scale_ptr,            
        weight_scale_ptr,           
        batch_size,                 
        trans_a,                    
        trans_b                     
    };

    Gemm gemm_op;
    cutlass::Status status = gemm_op.initialize(args);
    if (status != cutlass::Status::kSuccess) {
        throw std::runtime_error("CUTLASS GEMM initialization failed");
    }

    status = gemm_op();
    if (status != cutlass::Status::kSuccess) {
        throw std::runtime_error("CUTLASS GEMM execution failed");
    }

    return output;
}
