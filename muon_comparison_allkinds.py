class Muon(torch.optim.Optimizer):
    """
    Muon - MomentUm Orthogonalized by Newton-schulz
    支持 default/double_momentum/svd_momentum 三种模式
    """
    def __init__(self, params, lr=0.02, momentum=0.95, nesterov=True, ns_steps=5, rank=0, world_size=1, type='default'):
        self.rank = rank
        self.world_size = world_size
        self.type = type
        # 支持 momentum 为单值或元组
      
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps)
        params: list[Tensor] = [*params]
        param_groups = []
        for size in {p.numel() for p in params}:
            b = torch.empty(world_size, size, dtype=torch.bfloat16, device="cuda")
            group = dict(params=[p for p in params if p.numel() == size],
                         update_buffer=b, update_buffer_views=[b[i] for i in range(world_size)])
            param_groups.append(group)
        super().__init__(param_groups, defaults)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            update_buffer: Tensor = group["update_buffer"]
            update_buffer_views: list[Tensor] = group["update_buffer_views"]
            params: list[Tensor] = group["params"]
            handle = None
            params_world = None

            def update_prev():
                handle.wait()
                for p_world, g_world in zip(params_world, update_buffer_views):
                    p_world.add_(g_world.view_as(p_world),
                                 alpha=-group["lr"] * max(1, p_world.size(-2) / p_world.size(-1))**0.5)

            for base_i in range(len(params))[::self.world_size]:
                if base_i + self.rank < len(params):
                    p = params[base_i + self.rank]
                    g = p.grad
                    assert g is not None
                    state = self.state[p]
                    if "momentum_buffer" not in state:
                        state["momentum_buffer"] = torch.zeros_like(g)
                    buf: Tensor = state["momentum_buffer"]

                    # 下面根据 type 选择不同的动量和正交化策略
                    if self.type == 'default':
                        buf.lerp_(g, 1 - group["momentum"])
                        g_muon = g.lerp_(buf, group["momentum"]) if group["nesterov"] else buf
                        g_muon = zeropower_via_newtonschulz5(g_muon, steps=group["ns_steps"]).flatten()
                    elif self.type == 'double_momentum':
                        # momentum[1] 用于 buf，momentum[0] 用于 nesterov
                        if group["nesterov"]:
                            g_muon = g.lerp_(buf, group["momentum"][0])
                        else:
                            g_muon = buf
                        g_muon = zeropower_via_newtonschulz5(g_muon, steps=group["ns_steps"]).flatten()
                        buf.lerp_(g, 1 - group["momentum"][1]) #更新buf放在计算 g_muon 之后
                    elif self.type == 'svd_momentum':
                        buf.lerp_(g, 1 - group["momentum"])
                        g_ortho = zeropower_via_newtonschulz5(g, steps=group["ns_steps"])
                        buf_ortho = zeropower_via_newtonschulz5(buf, steps=group["ns_steps"])
                        g_muon = g_ortho.lerp_(buf_ortho, group["momentum"]).flatten()
                      
                    elif self.type == 'svd_momentum_v2':
                        g_ortho = zeropower_via_newtonschulz5(g, steps=group["ns_steps"]).to(dtype=torch.float32)
                        buf.lerp_(g_ortho, 1 - group["momentum"])
                        g_muon = g_ortho.lerp_(buf, group["momentum"]).flatten().to(dtype=torch.bfloat16) if group["nesterov"] else buf.flatten()
                        
                    else:
                        raise ValueError(f"Unknown Muon type: {self.type}")
                else:
                    g_muon = update_buffer_views[self.rank]
                if base_i > 0:
                    update_prev()
                handle = dist.all_gather_into_tensor(update_buffer, g_muon, async_op=True)
                params_world = params[base_i : base_i + self.world_size]
            update_prev()
