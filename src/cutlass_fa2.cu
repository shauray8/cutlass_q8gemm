int run_grouped(Options& options) {
  using AttentionKernel = typename cutlass::gemm::kernel::DefaultFMHAGrouped<
    cutlass::half_t,      // scalar_t
    cutlass::arch::Sm80,  // ArchTag
    true,                 // Memory is aligned
    kQueriesPerBlock,
    kKeysPerBlock,
    kMaxK,
    GroupScheduleMode_
  >::FMHAKernel;

  using FMHA = cutlass::gemm::device::GemmGrouped<AttentionKernel>;
