#!/usr/bin/env python3
import time
import torch
import torch.nn as nn

def test_pytorch_install():
    # 1) Version info
    print(f"PyTorch version: {torch.__version__}")
    print(f"Built with CUDA: {torch.version.cuda}")
    
    # 2) CUDA availability and device query
    cuda_ok = torch.cuda.is_available()
    print(f"CUDA available: {cuda_ok}")
    if cuda_ok:
        print(f"  Device count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"    [{i}] {torch.cuda.get_device_name(i)}")
    
    # 3) Basic tensor operations & timing
    device = torch.device("cuda" if cuda_ok else "cpu")
    a = torch.randn(2000, 2000, device=device)
    b = torch.randn(2000, 2000, device=device)
    start = time.time()
    c = a @ b   # matrix multiply
    torch.cuda.synchronize() if cuda_ok else None
    print(f"Matrix multiply on {device}: {time.time() - start:.4f}s")
    
    # 4) Autograd sanity check
    x = torch.tensor(3.0, requires_grad=True, device=device)
    y = x * x + 2 * x + 1
    y.backward()
    print(f"Autograd: d(y)/d(x) at x=3 → {x.grad.item()}  (should be 2*x + 2 = 8)")
    
    # 5) Simple nn.Module forward
    model = nn.Sequential(
        nn.Linear(10, 20),
        nn.ReLU(),
        nn.Linear(20, 5)
    ).to(device)
    inp = torch.randn(4, 10, device=device)
    out = model(inp)
    print(f"NN forward: input shape {inp.shape} → output shape {out.shape}")

if __name__ == "__main__":
    test_pytorch_install()
