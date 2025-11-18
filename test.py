import torch
import torch.nn as nn
import torch.optim as optim
from RLAlg.alg.gan import GAN

device = torch.device("cuda:0")

net = nn.Linear(4, 1).to(device)
optimimizer = optim.Adam(net.parameters(), lr=1e-4)

for param in net.parameters():
    print(param)

inputs = torch.rand(4, 4).to(device)
inputs.requires_grad_(True)

outputs = net(inputs)

optimimizer.zero_grad()

grad = torch.autograd.grad(
            outputs=outputs,
            inputs=inputs,
            grad_outputs=torch.ones_like(outputs),
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]

grad_norm = torch.sum(torch.square(grad), dim=-1)
grad_penalty = torch.mean(grad_norm)

grad_penalty.backward()
optimimizer.step()
print("===============")
for param in net.parameters():
    print(param)