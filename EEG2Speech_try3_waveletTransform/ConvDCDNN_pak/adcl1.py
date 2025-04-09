import torch

class ImprovedADCL(torch.optim.Optimizer):
    def __init__(
        self,
        params,
        lr=1e-3,
        betas=(0.9, 0.999),
        activation_fn=None,
        vlr_clamp=1.0,
        weight_decay=1e-5,
        trapezoidal=False,
        omicron=1e-6
    ):
        defaults = {
            'lr': lr,
            'betas': betas,
            'activation_fn': activation_fn,
            'vlr_clamp': vlr_clamp,
            'weight_decay': weight_decay,
            'trapezoidal': trapezoidal,
            'omicron': omicron
        }
        super().__init__(params, defaults)

    def step(self, closure=None):
        if closure is None:
            raise RuntimeError('ImprovedADCL optimizer requires a closure.')

        with torch.enable_grad():
            loss = closure()

        for group in self.param_groups:
            beta1, beta2 = group['betas']
            for p in group['params']:
                if p.grad is not None:
                    # 动量和自适应学习率
                    state = self.state[p]
                    if 'm' not in state:
                        state['m'] = torch.zeros_like(p.grad)
                        state['v'] = torch.zeros_like(p.grad)
                        state['step'] = 0

                    state['step'] += 1
                    m_t = state['m'] = beta1 * state['m'] + (1 - beta1) * p.grad
                    v_t = state['v'] = beta2 * state['v'] + (1 - beta2) * torch.square(p.grad)

                    # 偏差修正
                    m_hat = m_t / (1 - beta1 ** state['step'])
                    v_hat = v_t / (1 - beta2 ** state['step'])

                    # 计算学习率
                    if group['activation_fn'] is None:
                        vlr = group['lr'] * loss / (torch.sum(v_hat) + group['omicron'])
                    else:
                        vlr = group['lr'] * group['activation_fn'](loss) / (torch.sum(v_hat) + group['omicron'])

                    # 学习率钳制
                    vlr = min(vlr, group['vlr_clamp'])

                    # L2 正则化
                    if group['weight_decay'] > 0:
                        p.grad = p.grad.add(p, alpha=group['weight_decay'])

                    # 参数更新
                    p.data -= vlr * m_hat / (torch.sqrt(v_hat) + group['omicron'])

        return loss