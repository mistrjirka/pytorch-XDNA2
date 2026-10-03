import torch
import xdna_train

xdna_train.register_xdna_device()
device = torch.device("xdna")

model = torch.nn.Sequential(
    torch.nn.Linear(768, 3072, bias=False),
    torch.nn.GELU(),
    torch.nn.Linear(3072, 768, bias=False),
).bfloat16().to(device)

x = torch.randn(256, 768, dtype=torch.bfloat16, device=device)
target = torch.zeros(256, 768, dtype=torch.bfloat16, device=device)
opt = torch.optim.Adam(model.parameters(), lr=1e-4)

opt.zero_grad(set_to_none=True)
loss = torch.nn.functional.mse_loss(model(x).float(), target.float())
loss.backward()
opt.step()
print("loss", float(loss.detach().cpu()))
