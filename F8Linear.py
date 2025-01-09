import torch 
import torch.nn as nn
from typing import Optional, Union
import cutlass_fp8

class F8Linear(nn.Linear):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device: Optional[Union[torch.device, str]],
        dtype: torch.dtype = torch.float8_e4m3fn,
        epilogue_str: str = "NONE",
    )
        super().__init__(in_features, out_features, bias=bias, device=device, dtype=torch.float16)
        self._epilogue_str = epilogue_str
        self.has_bias = bias
        self.heas_cehcked_weight = False
    
        self.register_buffer("weight_scale", torch.ones(1, dtype=torch.float32))
        self.register_buffer('input_scale', torch.ones(1, dtype=torch.float32))

    def to_fp8(self, tensor=torch.Tensor) -> torch.Tensor:
        return cutlass_fp8.convert_to_fp8(tensor)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.has_checked_weight:
            self.weight.data = self.to_fp8(self.weight.data)
            if self.bias is not None:
                self.bias.data = self.bias.data.to(torch.bfloat16)
            self.has_checked_weight = True

        x_fp8 = self.to_fp8(x)

        output = cutlass_fp8.cutlass_fp8_matmul(
            x_fp8,
            self.weight,
            self.bias,
            self._epilogue_str,
            self.has_bias,
            self.input_scale,
            self.weight_scale,
        )

        return output

def cutlass_fp8_matmul(
    input_tensor: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    epilogue_str: str,
    has_bias: bool,
    input_scale: torch.Tensor,
    weight_scale: torch.Tensor,
) -> torch.Tensor:
    
    batch_size=input_tensor.size(0)
    m = input_tensor.size(1) if input_tensor.dim() == 3 else 1
    k = input_tensor.size(-1)
    n = weight.size(0)

    lda = k
    ldb = n
    ldc = n

    return cutlass_fp8.cutlass_fp8gemm_batched_impl_custom(
        input_tensor,
        weight,
        m, n, k,
        lda, ldb, ldc,
        -1,
        -1,
        False,
        True,
        False,
        bias if has_bias else None,
        input_scale,
        weight_scale,
        epilogue_str
    )
    
