import torch
import os


def check_bfloat16_cublas_support():
    """Test if the GPU supports bfloat16 in cublasGemmStridedBatchedEx operations specifically"""
    if not torch.cuda.is_available():
        print("CUDA not available")
        return False

    try:
        # Create tensors that will specifically trigger cublasGemmStridedBatchedEx
        # Use larger dimensions and batch size to ensure the specific CUBLAS path is taken
        batch_size = 8
        seq_len = 64
        hidden_size = 128

        # Create example inputs similar to transformer attention operations
        query = torch.randn(batch_size, seq_len, hidden_size, device="cuda").to(torch.bfloat16)
        key = torch.randn(batch_size, seq_len, hidden_size, device="cuda").to(torch.bfloat16)
        value = torch.randn(batch_size, seq_len, hidden_size, device="cuda").to(torch.bfloat16)

        # This will trigger cublasGemmStridedBatchedEx internally
        # First compute query @ key.transpose(-2, -1) which uses batched matmul
        attn_weights = torch.bmm(query, key.transpose(1, 2))
        torch.cuda.synchronize()

        # Then compute attn_weights @ value which uses another batched matmul
        attn_output = torch.bmm(attn_weights.softmax(dim=-1), value)
        torch.cuda.synchronize()

        print("GPU fully supports bfloat16 in cublasGemmStridedBatchedEx")
        return True

    except RuntimeError as e:
        if "CUBLAS_STATUS_NOT_SUPPORTED" in str(e):
            print(f"bfloat16 not fully supported: {str(e)}")
            return False
        else:
            print(f"Other error occurred: {str(e)}")
            raise

    # set the device to GPU if available, otherwise use CPU


def set_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        # if using Apple MPS, fall back to CPU for unsupported ops
        os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
    else:
        device = torch.device("cpu")

    if device.type == "cuda":
        if check_bfloat16_cublas_support():
            print("Using bfloat16 precision")  # use bfloat16 for the entire notebook
            torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
        else:
            print(
                "bfloat16 not supported, using float32 instead")  # older GPUs like Titan X do not support CUDNN ops with bfloat16...
            torch.autocast("cuda", dtype=torch.float32).__enter__()
        # torch.autocast("cuda", dtype=torch.float32).__enter__()
        # turn on tfloat32 for Ampere GPUs
        # (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
        if torch.cuda.get_device_properties(0).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
    elif device.type == "mps":
        print(
            "\nSupport for MPS devices is preliminary. SAM 2 is trained with CUDA and might "
            "give numerically different outputs and sometimes degraded performance on MPS. "
            "See e.g. https://github.com/pytorch/pytorch/issues/84936 for a discussion."
        )

    return device
