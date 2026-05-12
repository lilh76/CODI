from typing import Optional, Dict
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
import os
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.schedulers.scheduling_ddim import DDIMScheduler

import diffuser.utils as utils
from diffuser.models.helpers import Losses, apply_conditioning


class GaussianDiffusionWrapped(nn.Module):
    def __init__(
        self,
        model,
        n_agents: int,
        horizon: int,
        history_horizon: int,
        observation_dim: int,
        action_dim: int,
        use_inv_dyn: bool = True,
        use_fwd_dyn: bool = True,
        use_rwd_model: bool = True,
        use_legal_model: bool = True,
        use_label_model: bool = False,
        label_model_path: str = "",
        discrete_action: bool = True,
        num_actions: int = 5,  # for discrete action space
        n_timesteps: int = 1000,
        clip_denoised: bool = False,
        predict_epsilon: bool = True,
        action_weight: float = 1.0,
        hidden_dim: int = 256,
        loss_discount: float = 1.0,
        loss_weights: np.ndarray = None,
        state_loss_weight: float = None,
        opponent_loss_weight: float = None,
        returns_condition: bool = False,
        condition_guidance_w: float = 1.2,
        returns_loss_guided: bool = False,
        loss_guidence_w: float = 0.1,
        value_diffusion_model: nn.Module = None,
        train_only_inv: bool = False,
        share_inv: bool = False,  # True
        joint_inv: bool = True,  # False
        share_fwd: bool = False,
        joint_fwd: bool = True,
        data_encoder: utils.Encoder = utils.IdentityEncoder(),
        **kwargs,
    ):
        assert action_dim > 0
        assert (
            not returns_condition or not returns_loss_guided
        ), "Can't do both returns conditioning and returns loss guidence"

        super().__init__()
        self.n_agents = n_agents
        self.horizon = horizon
        self.history_horizon = history_horizon
        self.observation_dim = observation_dim
        self.action_dim = action_dim # 1
        self.state_loss_weight = state_loss_weight
        self.opponent_loss_weight = opponent_loss_weight
        self.discrete_action = discrete_action
        self.num_actions = num_actions
        self.transition_dim = observation_dim + action_dim
        self.model = model
        self.use_inv_dyn = use_inv_dyn
        self.train_only_inv = train_only_inv
        self.use_fwd_dyn = use_fwd_dyn
        self.use_rwd_model = use_rwd_model
        self.use_legal_model = use_legal_model
        self.use_label_model = use_label_model
        self.label_model_path = label_model_path
        self.share_inv = share_inv
        self.joint_inv = joint_inv
        self.share_fwd = share_fwd
        self.joint_fwd = joint_fwd
        self.data_encoder = data_encoder

        if self.use_inv_dyn:
            self.inv_model = self._build_inv_model(
                hidden_dim,
                output_dim=action_dim if not discrete_action else num_actions,
            )
        if self.use_fwd_dyn:
            self.fwd_model = self._build_fwd_model(
                hidden_dim, 
                action_dim=action_dim if not discrete_action else num_actions,
            )
        if self.use_rwd_model:
            self.rwd_model = self._build_rwd_model(
                hidden_dim,
                action_dim=action_dim if not discrete_action else num_actions,
            )
        if self.use_legal_model:
            self.legal_model = self._build_legal_model(
                hidden_dim,
                output_dim=action_dim if not discrete_action else num_actions,
            )
        if self.use_label_model and self.label_model_path != "":
            assert os.path.exists(self.label_model_path)
            data = torch.load(self.label_model_path)
            self.label_model = self._build_label_model(self.observation_dim, num_actions)
            self.label_model.load_state_dict(data['model'])
            print(f"Loaded label model {self.label_model_path}")

        self.returns_condition = returns_condition
        self.condition_guidance_w = condition_guidance_w

        self.returns_loss_guided = returns_loss_guided
        self.loss_guidence_w = loss_guidence_w
        self.value_diffusion_model = value_diffusion_model
        if self.value_diffusion_model is not None:
            self.value_diffusion_model.requires_grad_(False)

        self.n_timesteps = int(n_timesteps)
        self.clip_denoised = clip_denoised
        self.predict_epsilon = predict_epsilon

        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=self.n_timesteps,
            clip_sample=True,
            prediction_type="epsilon",
            beta_schedule="squaredcos_cap_v2",
        )
        self.use_ddim_sample = False

        # get loss coefficients and initialize objective
        loss_weights = self.get_loss_weights(loss_discount, action_weight)
        loss_type = "state_l2" if self.use_inv_dyn else "l2"
        self.loss_fn = Losses[loss_type](loss_weights)

    def _build_inv_model(self, hidden_dim: int, output_dim: int):
        if self.joint_inv:
            print("\n USE JOINT INV \n")
            inv_model = nn.Sequential(
                nn.Linear(self.n_agents * (2 * self.observation_dim), hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, self.n_agents * output_dim),
            )

        elif self.share_inv:
            print("\n USE SHARED INV \n")
            inv_model = nn.Sequential(
                nn.Linear(2 * self.observation_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, output_dim),
            )

        else:
            print("\n USE INDEPENDENT INV \n")
            inv_model = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Linear(2 * self.observation_dim, hidden_dim),
                        nn.ReLU(),
                        nn.Linear(hidden_dim, hidden_dim),
                        nn.ReLU(),
                        nn.Linear(hidden_dim, output_dim),
                        nn.Softmax(dim=-1) if self.discrete_action else nn.Identity(),
                    )
                    for _ in range(self.n_agents)
                ]
            )

        return inv_model
    
    def _build_fwd_model(self, hidden_dim: int, action_dim: int):
        # P(s, a) -> s'
        if self.joint_fwd:
            print("\nUSE JOINT FWD\n")
            fwd_model = nn.Sequential(
                nn.Linear(self.n_agents * (action_dim + self.observation_dim), hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, self.n_agents * self.observation_dim),
            )
        elif self.share_fwd:
            print("\nUSE SHARED FWD\n")
            fwd_model = nn.Sequential(
                nn.Linear(action_dim + self.observation_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, self.observation_dim),
            )
        else: 
            print("\nUSE INDEPENDENT FWD\n")
            fwd_model = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Linear(action_dim + self.observation_dim, hidden_dim),
                        nn.ReLU(),
                        nn.Linear(hidden_dim, hidden_dim),
                        nn.ReLU(),
                        nn.Linear(hidden_dim, self.observation_dim),
                    )
                    for _ in range(self.n_agents)
                ]
            )

        return fwd_model

    def _build_rwd_model(self, hidden_dim: int, action_dim: int):
        # R(s, a) -> r
        rwd_model = nn.Sequential(
            # nn.Linear(self.n_agents * (action_dim + self.observation_dim), hidden_dim),
            nn.Linear(self.n_agents * (action_dim + self.observation_dim * 2), hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim * 4),
            nn.ReLU(),
            nn.Linear(hidden_dim * 4, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim * 1),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        return rwd_model

    def _build_legal_model(self, hidden_dim: int, output_dim: int):
        if self.joint_inv:
            legal_model = nn.Sequential(
                nn.Linear(self.n_agents * (self.observation_dim), hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, self.n_agents * output_dim),
            )

        elif self.share_inv:
            legal_model = nn.Sequential(
                nn.Linear(self.observation_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, output_dim),
            )

        else:
            legal_model = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Linear(self.observation_dim, hidden_dim),
                        nn.ReLU(),
                        nn.Linear(hidden_dim, hidden_dim),
                        nn.ReLU(),
                        nn.Linear(hidden_dim, output_dim),
                        nn.Softmax(dim=-1) if self.discrete_action else nn.Identity(),
                    )
                    for _ in range(self.n_agents)
                ]
            )

        return legal_model

    def _build_label_model(self, obs_dim: int, action_dim: int):

        class LabelModel(nn.Module):
            def __init__(self, obs_dim, action_dim, n_agents, hidden_dim=256):
                super().__init__()
                self.n_agents = n_agents
                self.input_dim = obs_dim + action_dim
                self.hidden_dim = hidden_dim

                self.rnn = nn.GRU(
                    input_size=self.input_dim,
                    hidden_size=hidden_dim, # // 2,
                    batch_first=True,
                )

                self.rnn_dropout = nn.Dropout(0.2)

                self.mlp = nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim // 2),
                    nn.ReLU(),
                    nn.Dropout(0.2),

                    nn.Linear(hidden_dim // 2, hidden_dim // 4),
                    nn.ReLU(),
                    nn.Dropout(0.2),

                    nn.Linear(hidden_dim // 4, 1),
                )

            def forward(self, x):
                # x: [bs, h, na, dim]
                bs, h, na, dim = x.shape

                x = x.view(bs * na, h, dim)      # [bs*na, h, dim]
                _, h_n = self.rnn(x)             # [1, bs*na, hidden_dim // 2]
                h_n = self.rnn_dropout(h_n)

                agent_repr = h_n.squeeze(0).view(bs, na, self.hidden_dim)  # [bs, na, hidden_dim // 2]

                out = self.mlp(agent_repr)       # [bs, na, 1]
                return out

        return LabelModel(obs_dim, action_dim, self.n_agents)

    def set_ddim_scheduler(self, n_ddim_steps: int = 15):
        self.ddim_noise_scheduler = DDIMScheduler(
            num_train_timesteps=self.n_timesteps,
            clip_sample=True,
            prediction_type="epsilon",
            beta_schedule="squaredcos_cap_v2",
        )
        self.ddim_noise_scheduler.set_timesteps(n_ddim_steps)
        self.use_ddim_sample = True

    def get_loss_weights(self, discount: float, action_weight: Optional[float] = None):
        """
        sets loss coefficients for trajectory

        discount   : float
            multiplies t^th timestep of trajectory loss by discount**t
        """

        if self.use_inv_dyn:
            dim_weights = torch.ones(self.observation_dim, dtype=torch.float32)
        else:
            dim_weights = torch.ones(self.transition_dim, dtype=torch.float32)

        # decay loss with trajectory timestep: discount**t
        discounts = discount ** torch.arange(self.horizon, dtype=torch.float)
        discounts = discounts / discounts.mean()
        discounts = torch.cat([torch.zeros(self.history_horizon), discounts])
        loss_weights = torch.einsum("h,t->ht", discounts, dim_weights)
        loss_weights = loss_weights.unsqueeze(1).expand(-1, self.n_agents, -1).clone()

        # manually set a0 weight
        if not self.use_inv_dyn:
            loss_weights[self.history_horizon, :, : self.action_dim] = action_weight
        return loss_weights

    # ------------------------------------------ sampling ------------------------------------------#

    def get_model_output(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        returns: Optional[torch.Tensor] = None,
        env_ts: Optional[torch.Tensor] = None,
        attention_masks: Optional[torch.Tensor] = None,
        use_composition_tech: Optional[bool] = False,
    ):
        if self.returns_condition:
            epsilon_uncond = self.model(x, t, returns=returns, env_timestep=env_ts, attention_masks=attention_masks,
                force_dropout=True,
            )
            epsilon = epsilon_uncond.clone()
            if use_composition_tech:
                n_agents = x.shape[2] # [bs, h2, n_agents, obs_shape]
                for i in range(n_agents):
                    epsilon_cond = self.model(x, t, returns=returns, env_timestep=env_ts, attention_masks=attention_masks,
                        use_dropout=False, use_composition_tech=True, compose_agent_idx=i,
                    )
                    epsilon += self.condition_guidance_w * (epsilon_cond - epsilon_uncond)
            else:
                epsilon_cond = self.model(x, t, returns=returns, env_timestep=env_ts, attention_masks=attention_masks,
                    use_dropout=False,
                )
                epsilon += self.condition_guidance_w * (epsilon_cond - epsilon_uncond)

        else:
            epsilon = self.model(x, t, env_timestep=env_ts, attention_masks=attention_masks)

        return epsilon

    @torch.no_grad()
    def conditional_sample(
        self,
        cond: Dict[str, torch.Tensor],
        returns: Optional[torch.Tensor] = None,
        env_ts: Optional[torch.Tensor] = None,
        horizon: int = None,
        attention_masks: Optional[torch.Tensor] = None,
        verbose: bool = False,  # True
        return_diffusion: bool = False,
        use_composition_tech: bool = False,
    ):
        """
        conditions : [ (time, state), ... ]
        """

        batch_size = cond["x"].shape[0]
        horizon = horizon or self.horizon + self.history_horizon
        shape = (batch_size, horizon, self.n_agents, self.observation_dim)

        device = list(cond.values())[0].device
        if self.use_ddim_sample:
            scheduler = self.ddim_noise_scheduler
        else:
            scheduler = self.noise_scheduler

        x = 0.5 * torch.randn(shape, device=device)  # 0.5 for low tempurature sampling

        if return_diffusion:
            diffusion = [x]

        # set step values
        # scheduler.set_timesteps(self.num_inference_steps)
        timesteps = scheduler.timesteps

        progress = utils.Progress(len(timesteps)) if verbose else utils.Silent()
        for t in timesteps:
            # 1. apply conditioning
            x = apply_conditioning(x, cond)
            x = self.data_encoder(x)

            # 2. predict model output
            ts = torch.full((batch_size,), t, device=device, dtype=torch.long)
            model_output = self.get_model_output(
                x, ts, returns, env_ts, attention_masks, use_composition_tech
            )

            # 3. compute previous image: x_t -> x_t-1
            x = scheduler.step(model_output, t, x).prev_sample

            progress.update({"t": t})
            if return_diffusion:
                diffusion.append(x)

        # finally make sure conditioning is enforced
        x = apply_conditioning(x, cond)
        x = self.data_encoder(x)

        progress.close()
        if return_diffusion:
            return x, torch.stack(diffusion, dim=1)
        else:
            return x

    # ------------------------------------------ training ------------------------------------------#

    def p_losses(
        self,
        x_start: torch.Tensor, # [bs, h, n_agents, *shape]
        cond: Dict[str, torch.Tensor],
        t: torch.Tensor,
        loss_masks: torch.Tensor,
        attention_masks: Optional[torch.Tensor] = None,
        returns: Optional[torch.Tensor] = None,
        env_ts: Optional[torch.Tensor] = None,
        acs: Optional[torch.Tensor] = None,
    ):
        
        if type(returns) == tuple:
            if hasattr(self, "label_model"):
                with torch.no_grad():
                    returns = list(returns)
                    bs, h, n_agents, _ = returns[1].shape
                    # print(returns[1].shape)
                    # print(returns[1][:10, 0, :, 0])
                    acs_onehot = torch.nn.functional.one_hot(acs.squeeze(-1).long(), num_classes=self.num_actions)
                    inputs = torch.cat([acs_onehot, x_start], dim=-1)
                    labels_predict = self.label_model(inputs) # [bs, n_agents, 1]
                    # print(labels_predict)
                    labels_predict = F.softmax(labels_predict, dim=1)
                    # print(labels_predict[:10, :, 0])
                    # print(labels_predict.shape)
                    labels_predict = labels_predict.unsqueeze(1).repeat(1, h, 1, 1)
                    returns[1] = labels_predict # should be [bs, h, n_agents, 1]

        noise = torch.randn_like(x_start)

        x_noisy = self.noise_scheduler.add_noise(x_start, noise, t)
        x_noisy = apply_conditioning(x_noisy, cond)
        x_noisy = self.data_encoder(x_noisy)

        epsilon = self.model(
            x_noisy,
            t,
            returns=returns,
            env_timestep=env_ts,
            attention_masks=attention_masks,
        )

        if not self.predict_epsilon:
            epsilon = apply_conditioning(epsilon, cond)
            epsilon = self.data_encoder(epsilon)

        assert noise.shape == epsilon.shape
        # print("++++++++++++++++++++++++++++++++++++++++++++")
        # print(epsilon[0, 0, 0, :])
        # print("#############################################")
        # print(noise)

        if self.predict_epsilon:
            loss, info = self.loss_fn(epsilon, noise)
        else:
            loss, info = self.loss_fn(epsilon, x_start)
        # print("---------------------------------------------")
        # print(loss)

        if "agent_idx" in cond.keys() and self.opponent_loss_weight is not None:
            opponent_loss_weight = torch.ones_like(loss) * self.opponent_loss_weight
            indices = (
                cond["agent_idx"]
                .to(torch.long)[..., None]
                .repeat(
                    1, opponent_loss_weight.shape[1], 1, opponent_loss_weight.shape[-1]
                )
            )
            opponent_loss_weight.scatter_(dim=2, index=indices, value=1)
            loss = loss * opponent_loss_weight

        # TODO(zbzhu): Check these two '.mean()'
        loss = (
            (loss * loss_masks).mean(dim=[1, 2]) / loss_masks.mean(dim=[1, 2])
        ).mean()

        if self.returns_loss_guided:
            returns_loss = self.r_losses(x_noisy, t, epsilon, cond)
            info["returns_loss"] = returns_loss
            loss = loss + returns_loss * self.loss_guidence_w

        return loss, info

    def r_losses(self, x_t, t, noise, cond):
        b = x_t.shape[0]
        t = t.detach().to(torch.int64)
        x_recon = self.predict_start_from_noise(x_t, t, noise)

        if self.clip_denoised:
            x_recon.clamp_(-1.0, 1.0)
        else:
            assert RuntimeError()

        model_mean, _, model_log_variance = self.q_posterior(
            x_start=x_recon, x_t=x_t, t=t
        )

        noise = 0.5 * torch.randn_like(x_t)
        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x_t.shape) - 1)))

        x_t_minus_1 = (
            model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise
        )
        x_t_minus_1 = apply_conditioning(x_t_minus_1, cond)
        x_t_minus_1 = self.data_encoder(x_t_minus_1)

        # in value_diffusion_model, t is trained as t - 1
        value_pred = self.value_diffusion_model(x_t_minus_1, t)

        # value_pred = torch.clamp(value_pred, 0.0, 400.0)
        return -1.0 * value_pred.mean()  # maximize value

    def compute_inv_loss(
        self,
        x: torch.Tensor,
        loss_masks: torch.Tensor,
        legal_actions: Optional[torch.Tensor] = None, # [bs, h2, n_agents, n_actions]
    ):
        info = {}
        x_t = x[:, :-1, :, self.action_dim :]  # (bs, seq - 1, na, obs_dim)
        a_t = x[:, :-1, :, : self.action_dim]  # (bs, seq - 1, na, action_dim)
        x_t_1 = x[:, 1:, :, self.action_dim :]  # (bs, seq - 1, na, obs_dim)

        x_comb_t = torch.cat([x_t, x_t_1], dim=-1)  # (bs, seq - 1, na, 2 * obs_dim)
        x_comb_t = x_comb_t.reshape(-1, x_comb_t.shape[2], 2 * self.observation_dim)  # (bs * (seq - 1), na, 2 * obs_dim)
        a_t = a_t.reshape(-1, a_t.shape[2], self.action_dim)  # (bs * (seq - 1), na, action_dim)
        masks_t = loss_masks[:, 1:].reshape(-1, loss_masks.shape[2])
        if legal_actions is not None:
            legal_actions_t = legal_actions[:, :-1].reshape(-1, *legal_actions.shape[2:])

        if self.joint_inv or self.share_inv:
            if self.joint_inv:
                # print(x_comb_t.shape)
                # print(x_comb_t.reshape(x_comb_t.shape[0], -1).shape)
                pred_a_t = self.inv_model(
                    x_comb_t.reshape(x_comb_t.shape[0], -1)  # (b a) f
                )
                pred_a_t = pred_a_t.reshape(x_comb_t.shape[0], x_comb_t.shape[1], -1)
            else:
                pred_a_t = self.inv_model(x_comb_t)

            # print(pred_a_t.shape)  # (15, 2, 0)
            # print(pred_a_t)

            # print(legal_actions.shape)  # (1, 8, 3, 9)
            # print(pred_a_t.shape)  # (7, 3, 0)

            if legal_actions is not None:
                pred_a_t[legal_actions_t == 0] = -1e10
            if self.discrete_action:
                # print(pred_a_t)
                # print(pred_a_t.shape)
                inv_loss = (
                    F.cross_entropy(
                        pred_a_t.reshape(-1, pred_a_t.shape[-1]),
                        a_t.reshape(-1).long(),
                        reduction="none",
                    )
                    * masks_t.reshape(-1)
                ).mean() / masks_t.mean()
                inv_acc = (
                    (pred_a_t.argmax(dim=-1, keepdim=True) == a_t)
                    .to(dtype=float)
                    .squeeze(-1)
                    * masks_t
                ).mean() / masks_t.mean()
                info["inv_acc"] = inv_acc
            else:
                inv_loss = (
                    F.mse_loss(pred_a_t, a_t, reduction="none") * masks_t.unsqueeze(-1)
                ).mean() / masks_t.mean()

        else:
            inv_loss = 0.0
            for i in range(self.n_agents):
                pred_a_t = self.inv_model[i](x_comb_t[:, i])
                if self.discrete_action:
                    inv_loss += (
                        F.cross_entropy(
                            pred_a_t, a_t[:, i].reshape(-1).long(), reduction="none"
                        )
                        * masks_t[:, i]
                    ).mean() / masks_t[:, i].mean()
                else:
                    inv_loss += (
                        F.mse_loss(pred_a_t, a_t[:, i]) * masks_t[:, i].unsqueeze(-1)
                    ).mean() / masks_t[:, i].mean()

        return inv_loss, info
    
    def compute_fwd_loss(
        self,
        x: torch.Tensor,
        loss_masks: torch.Tensor,
    ):  
        x_t = x[:, :-1, :, self.action_dim :]  # (bs, seq - 1, na, obs_dim)
        a_t = x[:, :-1, :, : self.action_dim]  # (bs, seq - 1, na, action_dim)
        x_t_1 = x[:, 1:, :, self.action_dim :]  # (bs, seq - 1, na, obs_dim)

        if not self.discrete_action:
            xa_t = x[:, :-1, :, :]
        else:
            a_onehot_t = torch.nn.functional.one_hot(torch.squeeze(a_t, -1).to(torch.int64), num_classes=self.num_actions).type(torch.FloatTensor).to(x.device)
            xa_t = torch.cat([a_onehot_t, x_t], dim=-1)  # (bs, seq - 1, na, num_actions + obs_dim)

        xa_t = xa_t.reshape(xa_t.shape[0] * xa_t.shape[1], xa_t.shape[2], xa_t.shape[3])  # (bs * (seq - 1), na, num_actions + obs_dim)
        x_t_1 = x_t_1.reshape(x_t_1.shape[0] * x_t_1.shape[1], x_t_1.shape[2], x_t_1.shape[3])
        masks_t = loss_masks[:, 1:].reshape(-1, loss_masks.shape[2])
        
        if self.joint_fwd or self.share_fwd:
            if self.joint_fwd:
                pred_next_obs = self.fwd_model(
                    xa_t.reshape(xa_t.shape[0], -1)
                ).reshape(xa_t.shape[0], xa_t.shape[1], -1)  # (bs * (seq - 1), na, obs_dim)
            else:
                pred_next_obs = self.fwd_model(xa_t)

            fwd_loss = (
                F.mse_loss(pred_next_obs, x_t_1, reduction="none")
                * masks_t.unsqueeze(-1)
            ).mean() / masks_t.mean()
        else:
            fwd_loss = 0.0
            for i in range(self.n_agents):
                pred_next_obs = self.fwd_model[i](xa_t[:, i])
                fwd_loss += (
                    F.mse_loss(pred_next_obs, x_t_1[:, i])
                    * masks_t[:, i].unsqueeze(-1)
                ).mean() / masks_t[:, i].mean()

        return fwd_loss
    
    def compute_rwd_loss(
        self,
        x: torch.Tensor,
        reward: torch.Tensor,
        loss_masks: torch.Tensor,
    ):
        # x [bs, h, na, obs_shape]
        # reward [bs, h, 1]
        # x_t = x[..., self.action_dim :]  # (bs, h, na, obs_dim)
        # a_t = x[..., : self.action_dim]  # (bs, h, na, action_dim)
        x_t = x[:, :-1, :, self.action_dim :]  # (bs, seq - 1, na, obs_dim)
        a_t = x[:, :-1, :, : self.action_dim]  # (bs, seq - 1, na, action_dim)
        x_t_1 = x[:, 1:, :, self.action_dim :]  # (bs, seq - 1, na, obs_dim)
        x_t = torch.cat([x_t, x_t_1], dim=-1)  # (bs, seq - 1, na, 2 * obs_dim)
        reward = reward[:, :-1] # [bs, h-1, 1]
        loss_masks = loss_masks[:, :-1] # [bs, h - 1, na, 1]
        
        if not self.discrete_action:
            assert 0
            xa_t = x
        else:
            a_onehot_t = torch.nn.functional.one_hot(torch.squeeze(a_t, -1).to(torch.int64), num_classes=self.num_actions).type(torch.FloatTensor).to(x.device)
            xa_t = torch.cat([a_onehot_t, x_t], dim=-1)  # (bs, h, na, num_actions + obs_dim)

        xa_t = xa_t.reshape(xa_t.shape[0] * xa_t.shape[1], xa_t.shape[2], xa_t.shape[3])  # (bs * h, na, num_actions + obs_dim)
        reward = reward.reshape(-1, 1) # (bs * h, 1)
        masks_t = loss_masks.reshape(-1, loss_masks.shape[2]) # (bs * h, n_agents)
        pred_reward = self.rwd_model(
            xa_t.reshape(xa_t.shape[0], -1) # [bs * h, n_agents * (n_actions + obs_shape)]
        )  # (bs * h, 1)
        
        rwd_loss = (
                F.mse_loss(pred_reward, reward, reduction="none")
                * masks_t[:, 0].unsqueeze(-1)
            ).mean() / masks_t[:, 0].mean()
        return rwd_loss
    
    def compute_legal_loss(
        self,
        x: torch.Tensor,
        loss_masks: torch.Tensor,
        legal_actions: Optional[torch.Tensor] = None,  # [bs, h2, n_agents, n_actions]
    ):
        info = {}
        x_t = x[:, :-1, :, self.action_dim:]  # (bs, seq - 1, na, obs_dim)
        if legal_actions is not None:
            legal_actions_t = legal_actions[:, :-1]  # (bs, seq - 1, na, n_actions)
        else:
            return torch.tensor(0.0, device=x.device), info

        x_t = x_t.reshape(-1, x_t.shape[2], self.observation_dim)  # (bs * (seq - 1), na, obs_dim)
        legal_actions_t = legal_actions_t.reshape(-1, legal_actions_t.shape[2], legal_actions_t.shape[3])  # (bs * (seq - 1), na, n_actions)
        masks_t = loss_masks[:, :-1].reshape(-1, loss_masks.shape[2])  # (bs * (seq - 1), na)

        if self.joint_inv or self.share_inv:
            if self.joint_inv:
                pred_legal = self.legal_model(
                    x_t.reshape(x_t.shape[0], -1)  # (b, na * obs_dim)
                )
                pred_legal = pred_legal.reshape(x_t.shape[0], x_t.shape[1], -1)  # (b, na, n_actions)
            else:
                pred_legal = self.legal_model(x_t)

            legal_loss = (
                F.binary_cross_entropy_with_logits(
                    pred_legal, 
                    legal_actions_t.float(), 
                    reduction='none'
                ) * masks_t.unsqueeze(-1)
            ).mean() / masks_t.mean()

            pred_binary = (torch.sigmoid(pred_legal) > 0.5).float()
            legal_acc = (
                (pred_binary == legal_actions_t).float().mean(dim=-1) * masks_t
            ).sum() / masks_t.sum()
            info["legal_acc"] = legal_acc

        else:
            legal_loss = 0.0
            for i in range(self.n_agents):
                pred_legal = self.legal_model[i](x_t[:, i])
                loss_i = (
                    F.binary_cross_entropy_with_logits(
                        pred_legal, 
                        legal_actions_t[:, i].float(), 
                        reduction='none'
                    ) * masks_t[:, i].unsqueeze(-1)
                ).mean() / masks_t[:, i].mean()
                legal_loss += loss_i

        return legal_loss, info
    
    

    def loss(
        self,
        x: torch.Tensor,
        cond: Dict[str, torch.Tensor],
        loss_masks: torch.Tensor,
        loss_masks_world_models: torch.Tensor,
        attention_masks: Optional[torch.Tensor] = None,
        returns: Optional[torch.Tensor] = None,
        rewards: Optional[torch.Tensor] = None,
        env_ts: Optional[torch.Tensor] = None,
        legal_actions: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ):
        
        if self.train_only_inv:
            assert self.use_inv_dyn, "If train_only_inv, must use inv_dyn"
            info = {}
        else:
            batch_size = len(x)
            t = torch.randint(
                0,
                self.noise_scheduler.config.num_train_timesteps,
                (batch_size,),
                device=x.device,
            ).long()

            if labels is not None:
                returns = (returns, labels)
                
            if self.use_inv_dyn:
                diffuse_loss, info = self.p_losses(
                    x[..., self.action_dim :],
                    cond,
                    t,
                    loss_masks,
                    attention_masks,
                    returns,
                    env_ts,
                    x[..., : self.action_dim],
                )
            else:
                diffuse_loss, info = self.p_losses(
                    x,
                    cond,
                    t,
                    loss_masks,
                    attention_masks,
                    returns,
                    env_ts,
                )

        losses = [diffuse_loss]
        if self.use_inv_dyn:
            inv_loss, inv_info = self.compute_inv_loss(x, loss_masks_world_models, legal_actions)
            info = {**info, **inv_info}
            info["inv_loss"] = inv_loss
            if self.train_only_inv:
                return inv_loss, info
            losses.append(inv_loss)
        if self.use_fwd_dyn:
            fwd_loss = self.compute_fwd_loss(x, loss_masks_world_models)
            info["fwd_loss"] = fwd_loss
            losses.append(fwd_loss)
        if self.use_rwd_model:
            rwd_loss = self.compute_rwd_loss(x, rewards, loss_masks_world_models)
            info["rwd_loss"] = rwd_loss
            losses.append(rwd_loss)
        if self.use_legal_model:
            legal_loss, legal_info = self.compute_legal_loss(x, loss_masks_world_models, legal_actions)
            info = {**info, **legal_info}
            info["legal_loss"] = legal_loss
            losses.append(legal_loss)
        info["diff_loss"] = losses[0]
        
        # loss = sum(losses)
        loss = sum(losses) / len(losses)

        # loss = self.compute_rwd_loss(x, rewards, loss_masks_world_models)
        # info = {"rwd_loss": loss}


        return loss, info

    def forward(self, cond, *args, **kwargs):
        return self.conditional_sample(cond=cond, *args, **kwargs)


# class ValueDiffusion(GaussianDiffusion):
#     def __init__(self, *args, clean_only=False, **kwargs):
#         assert "value" in kwargs["loss_type"]
#         super().__init__(*args, **kwargs)
#         if clean_only:
#             print("[ models/diffusion ] Info: Only train on clean samples!")
#         self.clean_only = clean_only
#         self.sqrt_alphas_cumprod = torch.cat(
#             [
#                 torch.ones(1, device=self.betas.device),
#                 torch.sqrt(self.alphas_cumprod[:-1]),
#             ]
#         )
#         self.sqrt_one_minus_alphas_cumprod = torch.cat(
#             [
#                 torch.zeros(1, device=self.betas.device),
#                 torch.sqrt(1 - self.alphas_cumprod[:-1]),
#             ]
#         )

#     def loss(self, x, cond, returns=None):
#         batch_size = len(x)
#         t = torch.randint(0, self.n_timesteps, (batch_size,), device=x.device).long()
#         value_loss, info = self.p_losses(x, cond, returns, t - 1)
#         value_loss = value_loss.mean()
#         return value_loss, info

#     def p_losses(self, x_start, cond, target, t):
#         if self.clean_only:
#             pred = self.model(x_start, torch.zeros_like(t))

#         else:
#             t = t + 1
#             noise = torch.randn_like(x_start)

#             # since self.sqrt_alphas_cumprod and xxx is changed in __init__(),
#             # x_noisy here is x_t_minus_1
#             x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
#             x_noisy = apply_conditioning(x_noisy, cond)
#             x_noisy = self.data_encoder(x_noisy)
#             pred = self.model(x_noisy, t)

#         loss, info = self.loss_fn(pred, target)
#         return loss, info

#     def forward(self, x, t):
#         return self.model(x, t)
