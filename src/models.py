import torch
import torch.nn as nn

# [1] PINN 모델 아키텍처 정의
class PressurePINN(nn.Module):
    def __init__(self):
        super(PressurePINN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(4, 128), nn.Tanh(),
            nn.Linear(128, 128), nn.Tanh(),
            nn.Linear(128, 128), nn.Tanh(),
            nn.Linear(128, 64), nn.Tanh(),
            nn.Linear(64, 1)
        )
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight, gain=0.1)
                nn.init.constant_(m.bias, 0.0)

    def forward(self, x, y, z, t):
        inputs = torch.cat([x, y, z, t], dim=1)
        return self.net(inputs) + 30.0 # 초기 저류층 압력 기반 변환

# [2] 물리 법칙 제약 손실 함수
def compute_physics_loss(model, coords, k_map):
    coords.requires_grad = True
    p = model(coords[:, [0]], coords[:, [1]], coords[:, [2]], coords[:, [3]])

    grads = torch.autograd.grad(p, coords, grad_outputs=torch.ones_like(p), create_graph=True)[0]
    dp_dx, dp_dy, dp_dz, dp_dt = grads[:, [0]], grads[:, [1]], grads[:, [2]], grads[:, [3]]

    flux_x, flux_y, flux_z = k_map * dp_dx, k_map * dp_dy, k_map * dp_dz

    p_xx = torch.autograd.grad(flux_x, coords, grad_outputs=torch.ones_like(flux_x), create_graph=True)[0][:, [0]]
    p_yy = torch.autograd.grad(flux_y, coords, grad_outputs=torch.ones_like(flux_y), create_graph=True)[0][:, [1]]
    p_zz = torch.autograd.grad(flux_z, coords, grad_outputs=torch.ones_like(flux_z), create_graph=True)[0][:, [2]]

    source = 3.5 * torch.exp(-((coords[:, [0]]-32)**2 + (coords[:, [1]]-32)**2 + (coords[:, [2]]-32)**2) / 120.0)
    loss_pde = 20.0 * torch.mean((p_xx + p_yy + p_zz - dp_dt + source)**2)
    loss_safety = torch.mean(torch.nn.functional.relu(p - 45.0)**2)

    return loss_pde, loss_safety

# [3] XAI를 위한 Integrated Gradients 알고리즘
def compute_integrated_gradients(model, input_coords, baseline, steps=50):
    model.eval()
    delta = input_coords - baseline
    integrated_grads = torch.zeros_like(input_coords)
    alphas = torch.linspace(0, 1, steps).to(input_coords.device)

    for alpha in alphas:
        path_input = (baseline + alpha * delta).clone().detach().requires_grad_(True)
        out = model(path_input[:, [0]], path_input[:, [1]], path_input[:, [2]], path_input[:, [3]])
        grad = torch.autograd.grad(out, path_input, grad_outputs=torch.ones_like(out))[0]
        integrated_grads += grad / steps

    return delta * integrated_grads
