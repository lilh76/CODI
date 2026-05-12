from typing import Optional, Dict
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.schedulers.scheduling_ddim import DDIMScheduler

import diffuser.utils as utils
from diffuser.models.helpers import Losses, apply_conditioning

from .call_llm import LLM
from tqdm import tqdm
import re
import os

class LabelModel(nn.Module):
    """
    label model
        input: [bs, h, na, obs_dim + action_dim]
        output: [bs, na, 1]
    """
    def __init__(self, hidden_dim, obs_dim, action_dim, n_agents, dropout):
        super().__init__()
        self.n_agents = n_agents
        self.input_dim = obs_dim + action_dim
        self.hidden_dim = hidden_dim

        self.rnn = nn.GRU(
            input_size=self.input_dim,
            hidden_size=hidden_dim, 
            batch_first=True,
            num_layers=1,
            bidirectional=False,
        )
        
        d_rate = dropout # alternatives: 0, 0.1, 0.2
        self.rnn_dropout = nn.Dropout(0)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(d_rate),

            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(d_rate),

            nn.Linear(hidden_dim // 4, 1),
        )

    def forward(self, x):
        # x: [bs, h, na, dim]
        bs, h, na, dim = x.shape
        #if torch.isnan(x).any() or torch.isinf(x).any():
        #    raise ValueError("NaN / Inf detected in x")
        x = x.contiguous().view(bs * na, h, dim)      # [bs*na, h, dim]
        _, h_n = self.rnn(x)             # [1, bs*na, hidden_dim // 2]
        h_n = self.rnn_dropout(h_n)
        h_n_squeezed = h_n.squeeze(0)
        agent_repr = h_n.squeeze(0).contiguous().view(bs, na, self.hidden_dim)  # [bs, na, hidden_dim // 2]

        out = self.mlp(agent_repr)       # [bs, na, 1]
        return out

class LabelModelWrapper(nn.Module):
    def __init__(
        self,
        n_agents: int,
        horizon: int,
        history_horizon: int,
        observation_dim: int,
        action_dim: int,
        discrete_action: bool = True,
        num_actions: int = 5,  # for discrete action space
        hidden_dim: int = 256,
        dropout: float = 0.0,
        **kwargs,
    ):
        assert action_dim > 0

        super().__init__()
        self.n_agents = n_agents
        self.horizon = horizon
        self.history_horizon = history_horizon
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.discrete_action = discrete_action
        self.num_actions = num_actions
        self.transition_dim = observation_dim + action_dim

        self.label_model = self._build_label_model(
            hidden_dim, dropout,
            obs_dim=self.observation_dim, 
            action_dim=action_dim if not discrete_action else num_actions,
        )

    def _build_label_model(self, hidden_dim: int, dropout: float, obs_dim: int, action_dim: int):
        return LabelModel(hidden_dim, obs_dim, action_dim, self.n_agents, dropout)
    
    def _infer_smac_layout(self, obs_dim: int):
        """
        Heuristically infer a simplified SMAC obs layout, ignoring unit-type onehot.
        """
        # try without shield
        for has_shield in (0, 1):
            per_block = 5 + has_shield
            # obs_dim = 4 + n_enemy*per_block + (n_agents-1)*per_block + (1+has_shield)
            numer = obs_dim - 4 - (self.n_agents - 1) * per_block - (1 + has_shield)
            if numer > 0 and numer % per_block == 0:
                n_enemy = numer // per_block
                if n_enemy >= 1:
                    return per_block, int(n_enemy), has_shield
        # fallback: assume common 3m layout: no shield, 3 enemies
        return 5, 3, 0

    def _build_llm_prompt_smac(self, 
        x_item: torch.Tensor, 
        rewards_item: Optional[torch.Tensor],
        loss_masks_world_models_item: Optional[torch.Tensor] = None
    ) -> str:
        """
        Note: currently only support for SMAC 3m environment is validated.
        """
        import numpy as np
        import math
        import torch as _torch

        # --- shapes & conversions ---
        T = int(x_item.shape[0])
        N = int(x_item.shape[1])
        act_dim = int(self.action_dim)
        obs_dim = int(self.observation_dim)

        def to_np(tensor):
            if isinstance(tensor, _torch.Tensor):
                return tensor.detach().cpu().numpy()
            return np.array(tensor)
        x_np = to_np(x_item)

        # -------------------------
        # Parse per-trajectory mask (determines if the timestep is valid)
        # expected shape: [T, N, 1] or [T, N]
        mask_np = None
        if loss_masks_world_models_item is not None:
            mask_np = to_np(loss_masks_world_models_item)
            # collapse last dim if needed
            if mask_np.ndim == 3 and mask_np.shape[-1] == 1:
                mask_np = mask_np[..., 0]   # now [T, N]

        # --- header & legend ---
        header_lines = [
            "You are an expert StarCraft II micro-management coach.",
            f"This is a joint trajectory segment with T={T} timesteps and N={N} friendly agents.",
            "Task: pick EXACTLY ONE agent (0-based index) that you judge to be the most expert-like.",
            "IMPORTANT: The last line of your OUTPUT should be STRICTLY and ONLY one index in the format: [k]  (k must be between 0 and {}).".format(N - 1),
            "",
            "Action legend:",
            "  0 = no-op (only available for dead agents)",
            "  1 = stop",
            "  2-5 = move (cardinal directions)",
            "  >=6 = attack(enemy_{id-6})",
            "",
            "Primary signals (in order):",
            "  1) Timely engagement: attacking when enemies are attackable and causing HP drops.",
            "  2) Purposeful positioning: moves that reduce distance to enemies.",
            "  3) Team coordination: focusing same enemy as teammates when appropriate.",
            "  4) Survivability: staying alive and keeping health/shield.",
            "Tie-break (if needed): higher attack efficiency → earlier first attack → higher remaining health → better team overlap.",
        ]

        if rewards_item is not None:
            try:
                total_r = float(to_np(rewards_item).sum())
                header_lines.append(f"Context: team total reward over this segment ≈ {total_r:.3f}.")
            except Exception:
                pass

        # --- try to infer SMAC layout (per-enemy block size, number enemies, shield) ---
        try:
            per_block, n_enemy, has_shield = self._infer_smac_layout(obs_dim)
        except Exception:
            per_block, n_enemy, has_shield = None, None, None

        # conservative access helpers
        def safe_get(obs_arr, idx, default=float('nan')):
            return obs_arr[idx] if 0 <= idx < len(obs_arr) else default

        # action extraction robust to scalar or one-hot/vector
        def extract_action_id(a_slice):
            # a_slice is a numpy array possibly scalar or vector
            try:
                a_arr = np.array(a_slice)
                if a_arr.shape == () or a_arr.size == 1:
                    return int(a_arr.item())
                # If vector: treat as logits/one-hot -> argmax
                return int(int(np.argmax(a_arr)))
            except Exception:
                # fallback
                return 0

        # qualitative distance label
        def dist_label(d):
            try:
                if math.isnan(d):
                    return "unknown distance"
                if d < 0.08:
                    return "very close"
                if d < 0.25:
                    return "close"
                if d < 0.6:
                    return "medium"
                return "far"
            except Exception:
                return "unknown distance"

        # generate per-agent info
        agent_blocks = []
        # We need per-step actions across agents for team attribution (who attacked same enemy at same t)
        actions_all = [[extract_action_id(x_np[t, i, :act_dim]) for i in range(N)] for t in range(T)]

        # if we have layout info, compute absolute indices
        if per_block is not None and n_enemy is not None:
            enemy_start = 4
            ally_start = enemy_start + n_enemy * per_block
            own_start = ally_start + (N - 1) * per_block
        else:
            enemy_start = ally_start = own_start = None

        agents_dead_finally = np.zeros(N, dtype=np.bool)

        for i in range(N):
            # per-agent accumulators
            attack_count = 0
            damage_causing_attacks = 0
            first_attack_t = None
            moves_towards = 0
            move_count = 0
            seen_enemy_steps = 0
            nearest_distances = []
            own_hp_traj = []
            own_shield_traj = [] if has_shield else None
            focus_counts = np.zeros(n_enemy if n_enemy is not None else 0, dtype=int) if n_enemy else None

            timeline_lines = []

            for t in range(T):

                # check if the timestep is valid
                if mask_np is not None:
                    # mask: 1 -> valid (game continuing, but the agent can be dead), 0 -> invalid
                    is_valid = float(mask_np[t, i]) > 0.5
                else:
                    is_valid = True
                
                if not is_valid:
                    continue

                # extract action
                a_id = actions_all[t][i]
                a_label = ("idle/no-op" if a_id <= 0 else
                        "stop" if a_id == 1 else
                        "move" if 2 <= a_id <= 5 else
                        f"attack(enemy_{a_id - 6})")

                # observation slice for this agent
                obs = to_np(x_np[t, i, act_dim: act_dim + obs_dim]).astype(float)

                # read own hp / shield if layout known
                own_hp = float('nan')
                own_sh = None
                if own_start is not None:
                    # own_start is absolute index in obs vector (0-based)
                    if 0 <= own_start < obs_dim:
                        # obs is relative (0..obs_dim-1) so index = own_start
                        own_hp = safe_get(obs, own_start)
                        if has_shield:
                            own_sh = safe_get(obs, own_start + 1)
                    else:
                        # fallback: try last dims
                        own_hp = safe_get(obs, -1)
                        if has_shield:
                            own_sh = safe_get(obs, -2)
                else:
                    # fallback: guess last value is hp
                    own_hp = safe_get(obs, -1)
                own_hp_traj.append(float(np.nan_to_num(own_hp, nan=0.0)))
                if has_shield and own_sh is not None:
                    own_shield_traj.append(float(np.nan_to_num(own_sh, nan=0.0)))


                # scan enemies to determine visibility, nearest dist, hp for this agent at this step
                any_visible = False
                min_dist = None
                per_enemy_hp = {}
                per_enemy_visible = {}
                for e in range(n_enemy if n_enemy is not None else 0):
                    s = enemy_start + e * per_block
                    attackable = safe_get(obs, s + 0, 0.0)
                    dist = safe_get(obs, s + 1, float('nan'))
                    hp_e = safe_get(obs, s + 4, float('nan'))
                    # visible if hp>0 or attackable flag set
                    visible = (not math.isnan(hp_e) and hp_e > 0.0) or (attackable > 0.5)
                    per_enemy_visible[e] = visible
                    per_enemy_hp[e] = hp_e
                    if visible:
                        any_visible = True
                        if min_dist is None or (not math.isnan(dist) and dist < min_dist):
                            min_dist = dist
                        focus_counts[e] += 1 if focus_counts is not None else 0
                if any_visible:
                    seen_enemy_steps += 1
                if min_dist is not None and not math.isnan(min_dist):
                    nearest_distances.append(min_dist)

                # Build event description for this step
                step_events = []
                # own hp change vs previous step
                if t == 0:
                    prev_own_hp = None
                else:
                    prev_own_hp = own_hp_traj[-2] if len(own_hp_traj) >= 2 else None
                own_hp_note = ""
                if prev_own_hp is not None and not math.isnan(prev_own_hp):
                    if own_hp < prev_own_hp - 1e-6:
                        own_hp_note = f"own HP dropped {prev_own_hp:.2f}→{own_hp:.2f}"
                    elif own_hp > prev_own_hp + 1e-6:
                        own_hp_note = f"own HP rose {prev_own_hp:.2f}→{own_hp:.2f}"

                # check if the agent is now alive

                if own_hp == float('nan'):
                    print(f'[WARNING] agent {i} has nan health at timestep {t}')
                elif own_hp <= 0.0:
                    dead_hint = f"t={t}: "
                    if prev_own_hp is not None and prev_own_hp > 0:
                        dead_hint += f"own HP dropped {prev_own_hp:.2f}→{own_hp:.2f} "
                    dead_hint += 'DEAD.'
                    timeline_lines.append(dead_hint)
                    agents_dead_finally[i] = True
                    continue

                # action-specific logic
                if a_id >= 6:
                    attack_count += 1
                    if first_attack_t is None:
                        first_attack_t = t
                    target = a_id - 6
                    # check enemy hp now and next step (as seen by the same agent)
                    hp_now = per_enemy_hp.get(target, float('nan'))
                    hp_next = float('nan')
                    if t + 1 < T:
                        obs_next = to_np(x_np[t + 1, i, act_dim: act_dim + obs_dim]).astype(float)
                        hp_next = safe_get(obs_next, enemy_start + target * per_block + 4, float('nan')) if enemy_start is not None else float('nan')
                    # attribute other attackers at same timestep
                    same_step_attackers = [j for j in range(N) if (actions_all[t][j] >= 6 and actions_all[t][j] - 6 == target)]
                    attacker_list = same_step_attackers
                    attacker_note = "solo" if len(attacker_list) == 1 else f"joint by {attacker_list}"
                    damage_note = ""
                    if not math.isnan(hp_now) and not math.isnan(hp_next) and hp_next < hp_now - 1e-6:
                        damage_causing_attacks += 1
                        damage_note = f" enemy_{target} HP {hp_now:.2f}→{hp_next:.2f}"
                    # enemy visibility/distance note
                    dist_note = ""
                    if target in per_enemy_visible and per_enemy_visible[target]:
                        d = safe_get(obs, enemy_start + target * per_block + 1, float('nan'))
                        dist_note = f", {dist_label(d)}"
                    step_events.append(f"t={t}: attack(enemy_{target}) — {attacker_note}{dist_note}{damage_note}")
                elif 2 <= a_id <= 5:
                    # movement: check whether reduced nearest enemy distance next step
                    move_count += 1
                    purposeful = ""
                    # compute min dist now and next from this agent perspective
                    curr_min = min_dist if min_dist is not None else float('nan')
                    next_min = float('nan')
                    if t + 1 < T:
                        obs_next = to_np(x_np[t + 1, i, act_dim: act_dim + obs_dim]).astype(float)
                        # compute next step min dist
                        md = None
                        for e in range(n_enemy if n_enemy is not None else 0):
                            dnext = safe_get(obs_next, enemy_start + e * per_block + 1, float('nan')) if enemy_start is not None else float('nan')
                            if not math.isnan(dnext):
                                md = dnext if md is None else min(md, dnext)
                        if md is not None:
                            next_min = md
                    if not math.isnan(curr_min) and not math.isnan(next_min):
                        if next_min < curr_min - 1e-6:
                            purposeful = "moved closer to nearest enemy"
                            moves_towards += 1
                        elif next_min > curr_min + 1e-6:
                            purposeful = "moved away from nearest enemy"
                    note = f" ({purposeful})" if purposeful else ""
                    vis_note = f" saw enemies ({'yes' if any_visible else 'no'})"
                    step_events.append(f"t={t}: move{note};{vis_note}; own HP {own_hp:.2f}")
                else:
                    # idle/stop/other
                    if a_id <= 1:
                        step_events.append(f"t={t}: {a_label}; own HP {own_hp:.2f}")
                    else:
                        step_events.append(f"t={t}: {a_label}; own HP {own_hp:.2f}")

                # append own HP change if exists and not already mentioned
                if own_hp_note and "own HP" not in step_events[-1]:
                    step_events[-1] = step_events[-1] + f"; {own_hp_note}"
                timeline_lines.extend(step_events)

            # --- aggregate metrics ---
            seen_fraction = seen_enemy_steps / max(T, 1)
            avg_nearest = float(min(nearest_distances)) if nearest_distances else float('nan')
            attack_efficiency = damage_causing_attacks / attack_count if attack_count > 0 else 0.0
            move_effectiveness = moves_towards / move_count if move_count > 0 else 0.0

            # team overlap: fraction of attack steps where at least one teammate attacked same enemy
            team_overlap_count = 0
            if attack_count > 0:
                # compute again across timeline quickly
                for t in range(T):
                    a_id = actions_all[t][i]
                    if a_id >= 6:
                        target = a_id - 6
                        other_attackers = [j for j in range(N) if j != i and actions_all[t][j] >= 6 and (actions_all[t][j] - 6) == target]
                        if len(other_attackers) > 0:
                            team_overlap_count += 1
            team_overlap = team_overlap_count / attack_count if attack_count > 0 else 0.0

            # health summary (start -> end -> min)
            hp_start = own_hp_traj[0] if len(own_hp_traj) > 0 else float('nan')
            hp_end = own_hp_traj[-1] if len(own_hp_traj) > 0 else float('nan')
            hp_min = float(np.min(own_hp_traj)) if len(own_hp_traj) > 0 else float('nan')

            # style label heuristics
            if attack_count >= max(1, int(0.4 * T)) and attack_efficiency >= 0.5:
                style_label = "consistent engager"
            elif move_effectiveness > 0.4 and attack_count > 0:
                style_label = "mobile harasser"
            elif np.isnan(hp_min) or hp_min <= 0.01:
                style_label = "vulnerable (low HP / died)"
            elif (len(own_hp_traj) > 0) and ((hp_start - hp_end) < 0 and attack_count == 0):
                style_label = "healed / recovered"
            elif (len(own_hp_traj) > 0) and (attack_count == 0 and (T - (idle_cnt if 'idle_cnt' in locals() else 0)) > 0):
                style_label = "mostly passive"
            else:
                style_label = "mixed actions"

            # --- build human-readable agent block ---
            # concise scorecard with natural names
            scorecard_lines = [
                f"First attack time: {first_attack_t if first_attack_t is not None else 'never'}",
                f"Total attacks: {attack_count}",
                f"Damage-causing attacks (observed HP drops): {damage_causing_attacks}",
                f"Attack efficiency (damage-causing / total attacks): {attack_efficiency:.2f}",
                f"Moves that reduced distance to enemies / total moves: {moves_towards}/{move_count} (effectiveness {move_effectiveness:.2f})",
                f"Steps with enemy visible: {seen_enemy_steps}/{T} (≈{seen_fraction:.2f})",
                f"Team overlap on attacks (fraction): {team_overlap:.2f}",
                f"Health: start {hp_start:.2f} → end {hp_end:.2f}, min {hp_min:.2f}",
            ]

            block_lines = [
                f"Agent {i} — {style_label}.",
                "  Scorecard: " + "; ".join(scorecard_lines) + ".",
                "  Short summary: " + (
                    f"Attacked {attack_count} times with {damage_causing_attacks} attacks that coincided with observed enemy HP drops; "
                    f"moved closer to enemies on {moves_towards} moves; saw enemies in {seen_enemy_steps}/{T} steps."
                ),
                "  Timeline (key events; we show actions, important HP changes, visibility & qualitative distances):",
            ]
            # compress timeline a little: if long, show first 4 and last 3 with ellipsis
            if len(timeline_lines) <= 12:
                for s in timeline_lines:
                    block_lines.append("    - " + s)
            else:
                for s in timeline_lines[:4]:
                    block_lines.append("    - " + s)
                block_lines.append("    - ...")
                for s in timeline_lines[-3:]:
                    block_lines.append("    - " + s)

            agent_blocks.append("\n".join(block_lines))

        # --- assemble prompt ---
        prompt_parts = []
        prompt_parts.extend(header_lines)
        prompt_parts.append("")
        prompt_parts.append("Per-agent reports (commentator voice):")
        for b in agent_blocks:
            prompt_parts.append("")
            prompt_parts.append(b)

        prompt_parts.append("")
        prompt_parts.append("Decision hints (for your deliberation):")
        prompt_parts.append("  - Prefer agents that attack when enemies are attackable and whose attacks coincide with enemy HP drops.")
        prompt_parts.append("  - Prefer moves that reduce distance to enemies (purposeful approach).")
        prompt_parts.append("  - Prefer agents that contribute to team focus (overlap) rather than lone, ineffective attacks.")
        prompt_parts.append("")
        prompt_parts.append("FINAL INSTRUCTION: Derive your analysis in no more than 100 words based on the information above.")
        prompt_parts.append("The last line of your output should be ONLY the index of the most expert-like agent, in the format [k] without any other words.")
        info = dict()
        info['all_agents_alive'] = not np.any(agents_dead_finally)
        info['all_agents_dead'] = np.all(agents_dead_finally)

        return "\n".join(prompt_parts), info

    def _build_llm_prompt_smacv2(self, 
            x_item: torch.Tensor, 
            rewards_item: Optional[torch.Tensor],
            loss_masks_world_models_item: Optional[torch.Tensor] = None
        ) -> str:

        """
        Note: currently only support for SMAC-v2 10gen_zerg (3v4 variant) environment is validated.
        """

        import numpy as np
        import math
        import torch as _torch

        # --- shapes & conversions ---
        T = int(x_item.shape[0])
        N = int(x_item.shape[1])  # Should be 3 for 3v4
        act_dim = int(self.action_dim)
        obs_dim = int(self.observation_dim)  # Should be 58 for 3v4

        def to_np(tensor):
            if isinstance(tensor, _torch.Tensor):
                return tensor.detach().cpu().numpy()
            return np.array(tensor)
        x_np = to_np(x_item)

        # -------------------------
        # Parse per-trajectory mask
        mask_np = None
        if loss_masks_world_models_item is not None:
            mask_np = to_np(loss_masks_world_models_item)
            if mask_np.ndim == 3 and mask_np.shape[-1] == 1:
                mask_np = mask_np[..., 0]

        # --- SMAC-v2 10gen_zerg specific layout ---
        # Based on 58-dim observation for 3v4 zerg environment
        move_feats_dim = 4
        n_enemies = 4
        n_allies = 2  # N-1 = 3-1 = 2
        per_enemy_block = 8  # attackable(1) + distance(1) + rel_x(1) + rel_y(1) + health(1) + unit_type(3)
        per_ally_block = 8   # visible(1) + distance(1) + rel_x(1) + rel_y(1) + health(1) + unit_type(3)
        per_own_block = 6    # health(1) + unit_type(3) + own_pos(2)
        
        # Calculate indices
        enemy_start = move_feats_dim  # 4
        ally_start = enemy_start + n_enemies * per_enemy_block  # 4 + 4*8 = 36
        own_start = ally_start + n_allies * per_ally_block  # 36 + 2*8 = 52

        # --- header & legend ---
        header_lines = [
            "You are an expert StarCraft II micro-management coach analyzing a Zerg vs Zerg battle.",
            f"This is a joint trajectory segment with T={T} timesteps and N={N} friendly Zerg agents.",
            "Environment: SMAC-v2 10gen_zerg (3 friendly Zerg units vs 4 enemy Zerg units).",
            "Task: pick EXACTLY ONE agent (0-based index) that you judge to be the most expert-like.",
            "IMPORTANT: The last line of your OUTPUT should be STRICTLY and ONLY one index in the format: [k]  (k must be between 0 and {}).".format(N - 1),
            "",
            "Zerg Unit Types:",
            "  - Zergling: Fast melee attacker, low health, high damage up close",
            "  - Hydralisk: Ranged attacker, medium health, good dps",
            "  - Baneling: Suicide unit, explodes for area damage",
            "",
            "Action legend:",
            "  0 = no-op (only available for dead agents)",
            "  1 = stop", 
            "  2-5 = move (cardinal directions)",
            "  >=6 = attack(enemy_{id-6})  (targets enemy 0-3)",
            "",
            "Primary signals (in order):",
            "  1) Unit-type appropriate tactics: Zerglings close distance, Hydralisks maintain range, Banelings seek multi-kills",
            "  2) Timely engagement: attacking when enemies are attackable and causing HP drops",
            "  3) Target priority: focusing vulnerable enemies or high-value targets first",
            "  4) Survivability: avoiding unnecessary damage while dealing damage",
            "  5) Team coordination: focusing same enemy as teammates when appropriate",
            "Tie-break (if needed): higher attack efficiency → better target selection → higher remaining health → earlier first attack.",
        ]

        if rewards_item is not None:
            try:
                total_r = float(to_np(rewards_item).sum())
                header_lines.append(f"Context: team total reward over this segment ≈ {total_r:.3f}.")
            except Exception:
                pass

        # Helper functions
        def safe_get(obs_arr, idx, default=float('nan')):
            return obs_arr[idx] if 0 <= idx < len(obs_arr) else default

        def extract_action_id(a_slice):
            try:
                a_arr = np.array(a_slice)
                if a_arr.shape == () or a_arr.size == 1:
                    return int(a_arr.item())
                return int(np.argmax(a_arr))
            except Exception:
                return 0

        def dist_label(d):
            try:
                if math.isnan(d):
                    return "unknown distance"
                if d < 0.08:
                    return "very close"
                if d < 0.25:
                    return "close"
                if d < 0.6:
                    return "medium"
                return "far"
            except Exception:
                return "unknown distance"

        def get_unit_type_name(type_vec):
            """Convert 3-dim unit type vector to name"""
            try:
                if len(type_vec) >= 3:
                    if type_vec[0] > 0.5:
                        return "Zergling"
                    elif type_vec[1] > 0.5:
                        return "Hydralisk"
                    elif type_vec[2] > 0.5:
                        return "Baneling"
                return "Unknown"
            except:
                return "Unknown"

        # Generate per-agent info
        agent_blocks = []
        actions_all = [[extract_action_id(x_np[t, i, :act_dim]) for i in range(N)] for t in range(T)]
        agents_dead_finally = np.zeros(N, dtype=np.bool)

        for i in range(N):
            attack_count = 0
            damage_causing_attacks = 0
            first_attack_t = None
            moves_towards = 0
            move_count = 0
            seen_enemy_steps = 0
            nearest_distances = []
            own_hp_traj = []
            own_positions = []
            focus_counts = np.zeros(n_enemies, dtype=int)
            timeline_lines = []

            # Get agent's unit type (once at start)
            if T > 0:
                obs_first = to_np(x_np[0, i, act_dim: act_dim + obs_dim]).astype(float)
                unit_type_vec = [safe_get(obs_first, own_start + 1 + k, 0) for k in range(3)]
                agent_unit_type = get_unit_type_name(unit_type_vec)
            else:
                agent_unit_type = "Unknown"

            for t in range(T):
                if mask_np is not None:
                    is_valid = float(mask_np[t, i]) > 0.5
                else:
                    is_valid = True
                
                if not is_valid:
                    continue

                # Extract action
                a_id = actions_all[t][i]
                a_label = ("idle/no-op" if a_id <= 0 else
                        "stop" if a_id == 1 else
                        "move" if 2 <= a_id <= 5 else
                        f"attack(enemy_{a_id - 6})")

                # Observation slice
                obs = to_np(x_np[t, i, act_dim: act_dim + obs_dim]).astype(float)

                # Read own features
                own_hp = safe_get(obs, own_start, float('nan'))
                own_type_vec = [safe_get(obs, own_start + 1 + k, 0) for k in range(3)]
                own_pos_x = safe_get(obs, own_start + 4, float('nan'))
                own_pos_y = safe_get(obs, own_start + 5, float('nan'))
                
                own_hp_traj.append(float(np.nan_to_num(own_hp, nan=0.0)))
                if not math.isnan(own_pos_x) and not math.isnan(own_pos_y):
                    own_positions.append((own_pos_x, own_pos_y))

                # Scan enemies
                any_visible = False
                min_dist = None
                per_enemy_hp = {}
                per_enemy_visible = {}
                per_enemy_types = {}
                
                for e in range(n_enemies):
                    s = enemy_start + e * per_enemy_block
                    attackable = safe_get(obs, s + 0, 0.0)
                    dist = safe_get(obs, s + 1, float('nan'))
                    hp_e = safe_get(obs, s + 4, float('nan'))
                    type_vec = [safe_get(obs, s + 5 + k, 0) for k in range(3)]
                    
                    visible = (not math.isnan(hp_e) and hp_e > 0.0) or (attackable > 0.5)
                    per_enemy_visible[e] = visible
                    per_enemy_hp[e] = hp_e
                    per_enemy_types[e] = get_unit_type_name(type_vec)
                    
                    if visible:
                        any_visible = True
                        if min_dist is None or (not math.isnan(dist) and dist < min_dist):
                            min_dist = dist
                        focus_counts[e] += 1
                
                if any_visible:
                    seen_enemy_steps += 1
                if min_dist is not None and not math.isnan(min_dist):
                    nearest_distances.append(min_dist)

                # Build event description
                step_events = []
                
                # Check if agent died
                if own_hp <= 0.0:
                    prev_own_hp = own_hp_traj[-2] if len(own_hp_traj) >= 2 else None
                    dead_hint = f"t={t}: "
                    if prev_own_hp is not None and prev_own_hp > 0:
                        dead_hint += f"own HP dropped {prev_own_hp:.2f}→{own_hp:.2f} "
                    dead_hint += 'DEAD.'
                    timeline_lines.append(dead_hint)
                    agents_dead_finally[i] = True
                    continue

                # Own HP change
                own_hp_note = ""
                if t > 0:
                    prev_own_hp = own_hp_traj[-2] if len(own_hp_traj) >= 2 else None
                    if prev_own_hp is not None and not math.isnan(prev_own_hp):
                        if own_hp < prev_own_hp - 1e-6:
                            own_hp_note = f"own HP dropped {prev_own_hp:.2f}→{own_hp:.2f}"
                        elif own_hp > prev_own_hp + 1e-6:
                            own_hp_note = f"own HP rose {prev_own_hp:.2f}→{own_hp:.2f}"

                # Action-specific logic
                if a_id >= 6:
                    attack_count += 1
                    if first_attack_t is None:
                        first_attack_t = t
                    target = a_id - 6
                    
                    # Check enemy HP change
                    hp_now = per_enemy_hp.get(target, float('nan'))
                    hp_next = float('nan')
                    if t + 1 < T:
                        obs_next = to_np(x_np[t + 1, i, act_dim: act_dim + obs_dim]).astype(float)
                        hp_next = safe_get(obs_next, enemy_start + target * per_enemy_block + 4, float('nan'))
                    
                    # Team coordination
                    same_step_attackers = [j for j in range(N) if (actions_all[t][j] >= 6 and actions_all[t][j] - 6 == target)]
                    attacker_list = same_step_attackers
                    attacker_note = "solo" if len(attacker_list) == 1 else f"joint by {attacker_list}"
                    
                    # Damage note
                    damage_note = ""
                    if not math.isnan(hp_now) and not math.isnan(hp_next) and hp_next < hp_now - 1e-6:
                        damage_causing_attacks += 1
                        damage_note = f" enemy_{target} HP {hp_now:.2f}→{hp_next:.2f}"
                    
                    # Enemy info
                    dist_note = ""
                    enemy_type_note = ""
                    if target in per_enemy_visible and per_enemy_visible[target]:
                        d = safe_get(obs, enemy_start + target * per_enemy_block + 1, float('nan'))
                        dist_note = f", {dist_label(d)}"
                        enemy_type_note = f" ({per_enemy_types[target]})"
                    
                    step_events.append(f"t={t}: attack(enemy_{target}{enemy_type_note}) — {attacker_note}{dist_note}{damage_note}")
                    
                elif 2 <= a_id <= 5:
                    # Movement analysis
                    move_count += 1
                    purposeful = ""
                    curr_min = min_dist if min_dist is not None else float('nan')
                    next_min = float('nan')
                    
                    # Check if movement reduced distance
                    if t + 1 < T:
                        obs_next = to_np(x_np[t + 1, i, act_dim: act_dim + obs_dim]).astype(float)
                        md = None
                        for e in range(n_enemies):
                            dnext = safe_get(obs_next, enemy_start + e * per_enemy_block + 1, float('nan'))
                            if not math.isnan(dnext):
                                md = dnext if md is None else min(md, dnext)
                        if md is not None:
                            next_min = md
                    
                    if not math.isnan(curr_min) and not math.isnan(next_min):
                        if next_min < curr_min - 1e-6:
                            purposeful = "moved closer to nearest enemy"
                            moves_towards += 1
                        elif next_min > curr_min + 1e-6:
                            purposeful = "moved away from nearest enemy"
                    
                    note = f" ({purposeful})" if purposeful else ""
                    vis_note = f" saw enemies ({'yes' if any_visible else 'no'})"
                    step_events.append(f"t={t}: move{note};{vis_note}; own HP {own_hp:.2f}")
                else:
                    if a_id <= 1:
                        step_events.append(f"t={t}: {a_label}; own HP {own_hp:.2f}")
                    else:
                        step_events.append(f"t={t}: {a_label}; own HP {own_hp:.2f}")

                # Add HP change if not already mentioned
                if own_hp_note and "own HP" not in step_events[-1]:
                    step_events[-1] = step_events[-1] + f"; {own_hp_note}"
                timeline_lines.extend(step_events)

            # --- Aggregate metrics ---
            seen_fraction = seen_enemy_steps / max(T, 1)
            avg_nearest = float(np.mean(nearest_distances)) if nearest_distances else float('nan')
            attack_efficiency = damage_causing_attacks / attack_count if attack_count > 0 else 0.0
            move_effectiveness = moves_towards / move_count if move_count > 0 else 0.0

            # Team coordination
            team_overlap_count = 0
            if attack_count > 0:
                for t in range(T):
                    a_id = actions_all[t][i]
                    if a_id >= 6:
                        target = a_id - 6
                        other_attackers = [j for j in range(N) if j != i and actions_all[t][j] >= 6 and (actions_all[t][j] - 6) == target]
                        if len(other_attackers) > 0:
                            team_overlap_count += 1
            team_overlap = team_overlap_count / attack_count if attack_count > 0 else 0.0

            # Health summary
            hp_start = own_hp_traj[0] if len(own_hp_traj) > 0 else float('nan')
            hp_end = own_hp_traj[-1] if len(own_hp_traj) > 0 else float('nan')
            hp_min = float(np.min(own_hp_traj)) if len(own_hp_traj) > 0 else float('nan')

            # Style label with Zerg-specific heuristics
            if agent_unit_type == "Zergling":
                if attack_count >= max(1, int(0.5 * T)) and avg_nearest < 0.2:
                    style_label = "aggressive Zergling engager"
                elif move_effectiveness > 0.6:
                    style_label = "mobile Zergling flanker"
                else:
                    style_label = "Zergling"
            elif agent_unit_type == "Hydralisk":
                if attack_count >= max(1, int(0.4 * T)) and attack_efficiency >= 0.5:
                    style_label = "consistent Hydralisk dps"
                elif avg_nearest > 0.3 and attack_count > 0:
                    style_label = "positioning Hydralisk"
                else:
                    style_label = "Hydralisk"
            elif agent_unit_type == "Baneling":
                if attack_count > 0 and damage_causing_attacks > 0:
                    style_label = "effective Baneling bomber"
                elif attack_count == 0 and move_effectiveness > 0.5:
                    style_label = "maneuvering Baneling"
                else:
                    style_label = "Baneling"
            else:
                if attack_count >= max(1, int(0.4 * T)) and attack_efficiency >= 0.5:
                    style_label = "consistent engager"
                elif move_effectiveness > 0.4 and attack_count > 0:
                    style_label = "mobile harasser"
                elif np.isnan(hp_min) or hp_min <= 0.01:
                    style_label = "vulnerable (low HP / died)"
                else:
                    style_label = "mixed actions"

            # Target focus analysis
            focus_analysis = ""
            if focus_counts.sum() > 0:
                primary_target = np.argmax(focus_counts)
                focus_analysis = f"Most focused on enemy_{primary_target} ({focus_counts[primary_target]} attacks)"

            # --- Build agent block ---
            scorecard_lines = [
                f"Unit type: {agent_unit_type}",
                f"First attack time: {first_attack_t if first_attack_t is not None else 'never'}",
                f"Total attacks: {attack_count}",
                f"Damage-causing attacks: {damage_causing_attacks}",
                f"Attack efficiency: {attack_efficiency:.2f}",
                f"Movement effectiveness: {move_effectiveness:.2f}",
                f"Enemy visibility: {seen_enemy_steps}/{T} ({seen_fraction:.2f})",
                f"Team coordination: {team_overlap:.2f}",
                f"Health: {hp_start:.2f}→{hp_end:.2f} (min {hp_min:.2f})",
            ]
            if focus_analysis:
                scorecard_lines.append(focus_analysis)

            block_lines = [
                f"Agent {i} ({agent_unit_type}) — {style_label}.",
                "  Metrics: " + "; ".join(scorecard_lines) + ".",
                "  Summary: " + (
                    f"{agent_unit_type} with {attack_count} attacks ({damage_causing_attacks} caused damage); "
                    f"{moves_towards} purposeful moves; saw enemies {seen_fraction:.1%} of time."
                ),
                "  Timeline:",
            ]

            # Compress timeline if too long
            if len(timeline_lines) <= 12:
                for s in timeline_lines:
                    block_lines.append("    - " + s)
            else:
                for s in timeline_lines[:4]:
                    block_lines.append("    - " + s)
                block_lines.append("    - ...")
                for s in timeline_lines[-3:]:
                    block_lines.append("    - " + s)

            agent_blocks.append("\n".join(block_lines))

        # --- Assemble final prompt ---
        prompt_parts = []
        prompt_parts.extend(header_lines)
        prompt_parts.append("")
        prompt_parts.append("Per-agent reports:")
        for b in agent_blocks:
            prompt_parts.append("")
            prompt_parts.append(b)

        prompt_parts.append("")
        prompt_parts.append("Decision guidance (Zerg-specific):")
        prompt_parts.append("  - Zerglings: Should aggressively close distance and engage")
        prompt_parts.append("  - Hydralisks: Should maintain optimal range and deal consistent damage")  
        prompt_parts.append("  - Banelings: Should seek high-value multi-kills or finish key targets")
        prompt_parts.append("  - All: Should coordinate focus fire when advantageous")
        prompt_parts.append("")
        prompt_parts.append("FINAL: Provide brief analysis (≤100 words). Then the last line of your OUTPUT should ONLY be the index k of the agent chosen as expert, written in the form [k] .")
        
        info = {
            'all_agents_alive': not np.any(agents_dead_finally),
            'all_agents_dead': np.all(agents_dead_finally)
        }

        return "\n".join(prompt_parts), info

    
    def _build_llm_prompt_mpe_spread(
        self,
        x_item: "torch.Tensor",
        rewards_item: Optional["torch.Tensor"],
        loss_masks_world_models_item: Optional["torch.Tensor"] = None,
    ) -> str:
        """
        Note: currently only support for MPE spread3, spread4, spread5 environments is validated.
        """
        import numpy as np
        import torch as _torch
        from collections import Counter

        def to_np(t):
            if isinstance(t, _torch.Tensor):
                return t.detach().cpu().numpy()
            return np.array(t)

        x_np = to_np(x_item)
        T = int(x_np.shape[0])
        N = int(x_np.shape[1])
        act_dim = int(self.action_dim)
        obs_dim = int(self.observation_dim)

        # optional mask -> [T,N]
        mask_np = None
        if loss_masks_world_models_item is not None:
            mask_np = to_np(loss_masks_world_models_item)
            if mask_np.ndim == 3 and mask_np.shape[-1] == 1:
                mask_np = mask_np[..., 0]
            if mask_np.shape != (T, N):
                mask_np = mask_np.reshape(T, N)

        # action extraction
        def extract_action_id(a_slice):
            a_arr = np.array(a_slice)
            if a_arr.shape == () or a_arr.size == 1:
                return int(a_arr.item())
            return int(np.argmax(a_arr))

        action_legend = {
            0: "no-op",
            1: "move left",
            2: "move right",
            3: "move down",
            4: "move up",
        }

        # simple_spread layout: n_landmarks == n_agents
        n_landmarks = N
        entity_start = 4
        entity_len = n_landmarks * 2

        # compute effective observation length (vel(2)+pos(2)+landmarks(2*n_landmarks)+other_agents(2*(N-1)))
        effective_obs_len = 2 + 2 + n_landmarks * 2 + (N - 1) * 2  # equals 2 + 4*N; for N=3 -> 14

        # collect actions
        actions_all = [[extract_action_id(x_np[t, i, :act_dim]) for i in range(N)] for t in range(T)]

        # agent absolute positions (obs[2:4]) -- use effective obs slice (exclude action + tail one-hot)
        agent_pos = np.zeros((T, N, 2), dtype=float)
        for t in range(T):
            for i in range(N):
                obs_full = x_np[t, i, act_dim: act_dim + obs_dim].astype(float)
                obs_eff = obs_full[:effective_obs_len]
                agent_pos[t, i, :] = obs_eff[2:4]

        # per-step nearest distances and individual rewards
        per_step_nearest_idx = np.zeros((T, N), dtype=int)
        per_step_nearest_dist = np.zeros((T, N), dtype=float)
        per_step_ind_rewards = np.zeros((T, N), dtype=float)
        for t in range(T):
            for i in range(N):
                obs_full = x_np[t, i, act_dim: act_dim + obs_dim].astype(float)
                obs = obs_full[:effective_obs_len]  # <-- use only effective obs (avoid one-hot tail)
                entity_rel = obs[entity_start: entity_start + entity_len].reshape(n_landmarks, 2)
                dists = np.sqrt(np.sum(entity_rel ** 2, axis=-1))  # distances to landmarks
                nearest_idx = int(np.argmin(dists))
                nearest_dist = float(np.min(dists))
                per_step_nearest_idx[t, i] = nearest_idx
                per_step_nearest_dist[t, i] = nearest_dist
                per_step_ind_rewards[t, i] = - nearest_dist

        # collision threshold (agent size 0.15 -> threshold 0.3)
        agent_size = 0.15
        collision_threshold = agent_size * 2.0

        # Build per-agent blocks (no cumulative returns)
        agent_blocks = []
        for i in range(N):
            avg_nearest = float(np.nanmean(per_step_nearest_dist[:, i]))
            coverage_counts = np.bincount(per_step_nearest_idx[:, i], minlength=n_landmarks)
            fav_landmark = int(np.argmax(coverage_counts))
            coverage_fraction = float(coverage_counts.max() / max(1, coverage_counts.sum()))

            collision_steps = 0
            for t in range(T):
                for j in range(N):
                    if j == i:
                        continue
                    dist_ij = float(np.linalg.norm(agent_pos[t, i, :] - agent_pos[t, j, :]))
                    if dist_ij < collision_threshold:
                        collision_steps += 1
                        break

            timeline_lines = []
            for t in range(T):
                a_id = actions_all[t][i]
                a_label = action_legend.get(a_id, f"act_{a_id}")
                curr_dist = per_step_nearest_dist[t, i]
                if t == 0:
                    improvement = 0.0
                else:
                    prev_dist = per_step_nearest_dist[t - 1, i]
                    # improvement = previous - current: positive => got closer (good)
                    improvement = float(prev_dist - curr_dist)
                indiv_r = per_step_ind_rewards[t, i]
                # timeline: explicitly show improvement with clear parenthetical meaning
                timeline_lines.append(
                    f"    - t={t}: action={a_label};  Δd={-improvement:+.3f}; nearest landmark id={per_step_nearest_idx[t, i]}; distance to nearest landmark={curr_dist:.3f};"
                )
            
            delta_d = (per_step_nearest_dist[T-1, i] - per_step_nearest_dist[0, i])
            if abs(delta_d) < 0.0005:
                dist_prompt = "almost doesn't change"
            elif delta_d > 0:
                dist_prompt = f"INCREASES by {abs(delta_d): .3f}"
            else:
                dist_prompt = f"DECREASES by {abs(delta_d): .3f}"
            
            scorecard_lines = [
                f"Agent {i} — navigation report.",
                "  Scorecard:",
                f"    Favorite landmark index (most often nearest): {fav_landmark} (fraction {coverage_fraction:.2f})",
                f"    THROUGHOUT TIMESTEPS, DISTANCE {dist_prompt}", 
                f"    Collision steps observed: {collision_steps}",
            ]

            short_summary = f"collisions {collision_steps}; favorite landmark {fav_landmark}."
            block = scorecard_lines + ["  Short summary: " + short_summary, "  Timeline:"] + timeline_lines
            agent_blocks.append("\n".join(block))

        # team overlap fraction
        team_overlap_count = 0
        team_valid_steps = 0
        for t in range(T):
            if mask_np is not None:
                if mask_np[t].sum() < 0.5:
                    continue
            team_valid_steps += 1
            nearest_idxs = [int(per_step_nearest_idx[t, i]) for i in range(N)]
            cnt = Counter(nearest_idxs)
            if any(v > 1 for v in cnt.values()):
                team_overlap_count += 1
        team_overlap_frac = float(team_overlap_count / max(1, team_valid_steps))

        # Header: explicitly state small distance is better and definition of improvement
        header_lines = [
            "You are an expert cooperative-navigation coach for multi-agent teams.",
            f"This is a joint trajectory segment with T={T} timesteps and N={N} agents.",
            "Important context: among these agents, EXACTLY ONE is following a true expert strategy;",
            "the OTHER agents are random policies that at each step pick an action uniformly at random (each action equally likely).",
            "Task: pick EXACTLY ONE agent (0-based index) that you judge to be the most expert-like.",
            "IMPORTANT: The last line of your OUTPUT should be STRICTLY and ONLY one index in the format: [k]  (k must be between 0 and {}).".format(N - 1),
            "",
            "Action legend:",
            "  0 = no-op",
            "  1 = move left",
            "  2 = move right",
            "  3 = move down",
            "  4 = move up",
            "",
            # Crucial unambiguous statements about distance & improvement
            "IMPORTANT NOTES (CRITICAL):",
            "  - Δd = delta of the distance to nearest landmark.",
            "    * If Δd < 0  => agent moved CLOSER compared with previous step  => THIS IS GOOD.",
            "    * If Δd > 0  => agent moved AWAY compared with previous step  => THIS IS BAD.",
            "  - The following BEHAVIOR FEATURES (in PRIORITY ORDER) can be evidence of an expert-like agent: "
            "    1. if the distance to the nearest landmark is already reasonably small, keeps it so small consistently",
            "    2. if the distance is large, achieves largest DECREASEMENT of DISTANCE THROUGHOUT TIMESTEPS", 
            "    3. if the distance is large, chooses a series of purposeful actions that reduce the distance gradually / smoothly", 
            "    4. doesn't change favorite / nearest landmark frequently",
            "    5. avoids colliding with teammates",
            "  - IF TIES APPEAR, BREAK THEM BY STRICTLY FOLLOWING THE PRIORITY ORDER ABOVE. "
        ]

        if rewards_item is not None:
            total_r = float(np.sum(to_np(rewards_item)))
            header_lines.append(f"Context: team total reward over this segment ≈ {total_r:.3f}.")

        # assemble prompt
        prompt_parts = header_lines + [""] + agent_blocks + [
            "",
            "Team-level stats:",
            f"  Fraction of valid steps with >1 agents focusing same landmark (overlap): {team_overlap_frac:.3f}",
            "",
            "FINAL INSTRUCTION: Derive your analysis in no more than 100 words based on the information above.",
            "The last line of your output should be ONLY the index of the most expert-like agent, in the format [k].",
        ]

        return "\n".join(prompt_parts), dict()

    def _build_llm_prompt_mpe_tag(
        self,
        x_item: "torch.Tensor",
        rewards_item: "Optional[torch.Tensor]" = None,
        loss_masks_world_models_item: "Optional[torch.Tensor]" = None,
    ) -> str:
        """
        Note: currently only support for MPE tag3 environment is validated.
        """
        import numpy as np
        import torch as _torch
        from collections import Counter

        def to_np(t):
            return t.detach().cpu().numpy() if isinstance(t, _torch.Tensor) else np.array(t)

        x_np = to_np(x_item)
        T = int(x_np.shape[0])
        N = int(x_np.shape[1])
        act_dim = int(self.action_dim)
        obs_dim = int(self.observation_dim)

        # optional mask -> [T, N] (use assertion style if present)
        mask_np = None
        if loss_masks_world_models_item is not None:
            mask_np = to_np(loss_masks_world_models_item)
            if mask_np.ndim == 3 and mask_np.shape[-1] == 1:
                mask_np = mask_np[..., 0]
            # if provided shape differs, reshape by assertion (no try/except)
            assert mask_np.shape[0] == T, "mask first dim mismatch"
            if mask_np.shape != (T, N):
                mask_np = mask_np.reshape(T, N)


        num_adversaries = N
        num_good = N
        adversary_indices = list(range(num_adversaries))
        good_indices = list(range(num_adversaries, num_adversaries + num_good))


        expected_obs_dim = 2 + 2 + (num_adversaries + num_good - 1) * 2 + num_good * 2
        assert obs_dim - N == expected_obs_dim

        def extract_action_id(a_slice):
            a_arr = np.array(a_slice)
            if a_arr.shape == () or a_arr.size == 1:
                return int(a_arr.item())
            return int(np.argmax(a_arr))

        action_legend = {
            0: "no-op",
            1: "move left",
            2: "move right",
            3: "move down",
            4: "move up",
        }

        def action_label(a_id):
            return action_legend.get(a_id, f"act_{a_id}")

        # observation slicing constants (per asserted layout)
        other_start = 4
        other_len = (num_adversaries + num_good - 1) * 2  # flattened length == (2N - 1)*2
        # good_vels start at other_start + other_len (not needed for distance computations)

        # precompute actions_all (T x num_adversaries)
        actions_all = [[extract_action_id(x_np[t, i, :act_dim]) for i in range(N)] for t in range(T)]

        # allocate arrays for per-adversary across timesteps
        per_adv_nearest_idx = np.full((num_adversaries, T), -1, dtype=int)    # stores global good index
        per_adv_nearest_dist = np.full((num_adversaries, T), np.nan, dtype=float)
        per_adv_nearest_rel = np.full((num_adversaries, T, 2), np.nan, dtype=float)

        # For each adversary (global index a_idx in [0..N-1]), parse its obs and compute distances to all good agents
        for t in range(T):
            for a_local, a_idx in enumerate(adversary_indices):
                obs = x_np[t, a_idx, act_dim: act_dim + obs_dim].astype(float)

                # extract flattened other positions (defensive: exact sizing guaranteed by assertion)
                end_other = other_start + other_len
                other_flat = obs[other_start:end_other]
                # reshape into (total_other_count, 2) where total_other_count = num_adversaries + num_good - 1 = 2N - 1
                other_pos = other_flat.reshape(num_adversaries + num_good - 1, 2)

                # Among all good agents (global indices N..2N-1), compute distance
                best_g = -1
                best_d = None
                best_rel = np.array([np.nan, np.nan], dtype=float)
                for g in good_indices:
                    # map global index g to index in other_pos: if g < a_idx => k = g else k = g - 1
                    if g < a_idx:
                        k = g
                    else:
                        k = g - 1
                    rel = other_pos[k]
                    d = float(np.linalg.norm(rel))
                    if best_d is None or d < best_d:
                        best_d = d
                        best_g = int(g)
                        best_rel = rel.copy()
                # store
                per_adv_nearest_idx[a_local, t] = best_g
                per_adv_nearest_dist[a_local, t] = float(best_d)
                per_adv_nearest_rel[a_local, t, :] = best_rel

        # compute Δd: Δd_t = dist_t - dist_{t-1}; Δd < 0 => moved closer => GOOD for adversary
        per_adv_delta = np.zeros_like(per_adv_nearest_dist)
        if T > 0:
            per_adv_delta[:, 0] = 0.0
        for tt in range(1, T):
            per_adv_delta[:, tt] = per_adv_nearest_dist[:, tt] - per_adv_nearest_dist[:, tt - 1]

        # team coordination & per-adversary close counts
        close_radius = 0.20
        team_coord_steps = 0
        adv_coord_particip = np.zeros((num_adversaries,), dtype=int)
        adv_close_counts = np.zeros((num_adversaries,), dtype=int)

        for t in range(T):
            counts_per_good = {g: 0 for g in good_indices}
            for a_local in range(num_adversaries):
                gidx = int(per_adv_nearest_idx[a_local, t])
                d = per_adv_nearest_dist[a_local, t]
                if gidx >= 0 and not np.isnan(d) and d <= close_radius:
                    counts_per_good[gidx] += 1
            if any(v >= 2 for v in counts_per_good.values()):
                team_coord_steps += 1
                for a_local in range(num_adversaries):
                    gidx = int(per_adv_nearest_idx[a_local, t])
                    d = per_adv_nearest_dist[a_local, t]
                    if gidx >= 0 and not np.isnan(d) and d <= close_radius and counts_per_good[gidx] >= 2:
                        adv_coord_particip[a_local] += 1
            for a_local in range(num_adversaries):
                d = per_adv_nearest_dist[a_local, t]
                if not np.isnan(d) and d <= close_radius:
                    adv_close_counts[a_local] += 1

        # Build textual blocks for each adversary
        adv_blocks = []
        for a_local, a_idx in enumerate(adversary_indices):
            # average nearest dist (if present)
            valid_mask = ~np.isnan(per_adv_nearest_dist[a_local, :])
            avg_nearest = float(np.nanmean(per_adv_nearest_dist[a_local, :])) if np.any(valid_mask) else float('nan')

            # favorite good target
            cnts = Counter([int(per_adv_nearest_idx[a_local, t]) for t in range(T) if per_adv_nearest_idx[a_local, t] >= 0])
            if len(cnts) == 0:
                fav_good = None
                fav_frac = 0.0
            else:
                fav_good, fav_times = cnts.most_common(1)[0]
                fav_frac = fav_times / max(1, sum(cnts.values()))

            # improvement stats
            improving_steps = 0
            improvements = []
            for tt in range(1, T):
                prev = per_adv_nearest_dist[a_local, tt - 1]
                curr = per_adv_nearest_dist[a_local, tt]
                if not np.isnan(prev) and not np.isnan(curr) and (curr - prev) < 0:
                    improving_steps += 1
                    improvements.append(abs(curr - prev))
            frac_improve = float(improving_steps / max(1, T - 1))
            avg_improve_amt = float(np.mean(improvements)) if len(improvements) > 0 else 0.0
            std_delta = float(np.nanstd(per_adv_delta[a_local, :]))

            # overall trend description
            first_d = per_adv_nearest_dist[a_local, 0]
            last_d = per_adv_nearest_dist[a_local, -1]
            if np.isnan(first_d) or np.isnan(last_d):
                delta_desc = "no data"
                delta_overall = float('nan')
            else:
                delta_overall = last_d - first_d
                if abs(delta_overall) < 1e-4:
                    delta_desc = "almost unchanged"
                elif delta_overall < 0:
                    delta_desc = f"DECREASES by {abs(delta_overall):.3f} (closer)"
                else:
                    delta_desc = f"INCREASES by {abs(delta_overall):.3f} (farther)"

            # timeline (one line per timestep)
            timeline_lines = []
            for tt in range(T):
                if mask_np is not None and float(mask_np[tt, a_idx]) < 0.5:
                    continue
                act_id = actions_all[tt][a_idx]
                a_label = action_label(act_id)
                gidx = int(per_adv_nearest_idx[a_local, tt]) if per_adv_nearest_idx[a_local, tt] >= 0 else -1
                dist_val = per_adv_nearest_dist[a_local, tt]
                rel_vec = per_adv_nearest_rel[a_local, tt, :]
                rel_str = f"rel=({rel_vec[0]:.3f},{rel_vec[1]:.3f})" if not np.isnan(rel_vec).all() else "rel=(nan,nan)"
                delta_t = per_adv_delta[a_local, tt] if tt > 0 else 0.0
                timeline_lines.append(
                    f"    - t={tt}: action={a_label}; Δd={delta_t:+.3f}; nearest_good={gidx}; distance={0.0 if np.isnan(dist_val) else dist_val:.3f}; {rel_str}"
                )

            block_lines = [
                f"Adversary {a_idx} report:",
                "  Metrics:",
                f"    Avg nearest-good distance: {avg_nearest:.3f}",
                f"    Favorite target (good agent): {fav_good} (fraction {fav_frac:.2f})",
                f"    Distance trend across segment: {delta_desc}",
                f"    Fraction of steps with distance DECREASE (good steps): {frac_improve:.2f}",
                f"    Avg improvement magnitude when improving: {avg_improve_amt:.3f}",
                f"    Δd standard deviation (smoothness): {std_delta:.3f}",
                f"    Times within close radius ({close_radius:.2f}): {int(adv_close_counts[a_local])}",
                f"    Participation in multi-adversary close events: {int(adv_coord_particip[a_local])}",
                "  Short timeline (per-step):",
            ] + timeline_lines

            adv_blocks.append("\n".join(block_lines))

        # team coordination fraction
        team_coord_frac = float(team_coord_steps / max(1, T))

        # header
        header_lines = [
            "Tactical analyst summary for a hunting scenario.",
            f"This trajectory segment has T={T} timesteps and N(team_size)={N} (per your convention).",
            f"Adversaries (learnable): {adversary_indices}",
            f"Good agents (non-learning): {good_indices}",
            f"(obs_dim asserted: expected {expected_obs_dim}, actual {obs_dim})",
            "Task: identify EXACTLY ONE adversary index (from the adversary list above) that is MOST expert-like.",
            "Output format constraint: the very last line of your RESPONSE MUST be exactly one index in the form: [k]  (k must be one of the adversary indices).",
            "",
            "Action legend:",
            "  0 = no-op",
            "  1 = move left",
            "  2 = move right",
            "  3 = move down",
            "  4 = move up",
            "",
            "Key interpretation rule (Δd = current_distance - previous_distance):",
            "  * Δd < 0  => moved CLOSER to its nearest target since previous step => POSITIVE evidence of hunting skill.",
            "  * Δd > 0  => moved AWAY since previous step => NEGATIVE evidence.",
            "",
            "Primary evidence priority (use this order to decide):",
            "  1) sustained DECREASE in distance to targets across the segment (large negative overall Δ).",
            "  2) high fraction of timesteps with Δd < 0 (consistent closing).",
            "  3) participation in multi-adversary close events (shows tactical coordination).",
            "  4) smoothness: smaller Δd variance suggests purposeful approach rather than jittery/random moves.",
            "  5) consistency in preferring the same target (favorite target fraction).",
            "",
            "If ties occur, break them by (1) higher priority above, then (2) lower Δd variance, then (3) higher participation in coordinated events.",
        ]

        if rewards_item is not None:
            total_r = float(np.sum(to_np(rewards_item)))
            header_lines.append(f"Context: team total reward over this segment ≈ {total_r:.3f}.")

        # final assembly
        prompt_parts = header_lines + [""] + adv_blocks + [
            "",
            "Team-level statistics:",
            f"  Fraction of timesteps with >=2 adversaries simultaneously very close to the same target: {team_coord_frac:.3f}",
            "",
            "FINAL INSTRUCTION: In no more than 100 words, give a brief rationale (objective, tactical) for your choice,",
            "then on the FINAL LINE output ONLY the chosen adversary index in format [k].",
        ]

        info = {
            "adversary_indices": adversary_indices,
            "good_indices": good_indices,
            "num_adversaries": num_adversaries,
            "num_good": num_good,
            "expected_obs_dim": expected_obs_dim,
            "actual_obs_dim": obs_dim,
            "team_coord_frac": team_coord_frac,
        }

        return "\n".join(prompt_parts), info


    
    def _build_llm_prompt_mpe_world(
        self,
        x_item: "torch.Tensor",
        rewards_item: "Optional[torch.Tensor]" = None,
        loss_masks_world_models_item: "Optional[torch.Tensor]" = None,
    ) -> str:
        """
        Note: currently only support for MPE world3 environment is validated.
        """

        import numpy as np
        import torch as _torch
        from collections import Counter

        def to_np(t):
            return t.detach().cpu().numpy() if isinstance(t, _torch.Tensor) else np.array(t)

        x_np = to_np(x_item)
        T = int(x_np.shape[0])
        N = int(x_np.shape[1])
        act_dim = int(self.action_dim)
        obs_dim = int(self.observation_dim) - N

        # optional mask -> [T, N]
        mask_np = None
        if loss_masks_world_models_item is not None:
            mask_np = to_np(loss_masks_world_models_item)
            if mask_np.ndim == 3 and mask_np.shape[-1] == 1:
                mask_np = mask_np[..., 0]
            assert mask_np.shape[0] == T, "mask first dim mismatch"
            if mask_np.shape != (T, N):
                mask_np = mask_np.reshape(T, N)

        # fixed counts (for MPE world3)
        num_adversaries = 3
        num_good = 3
        total_expected = num_adversaries + num_good

        adversary_indices = list(range(num_adversaries))         # [0,1,2]
        good_indices = list(range(num_adversaries, num_adversaries + num_good))  # [3,4,5]

        # obs_dim should be 32 and fully defined under these counts
        expected_obs_dim = 32
        assert obs_dim == expected_obs_dim, f"expected obs_dim {expected_obs_dim}, got {obs_dim}"

        # slicing indices according to the derived layout
        i_self_vel = 0
        i_self_pos = 2
        entity_start = 4
        entity_len = 5 * 2                # 10
        entity_end = entity_start + entity_len  # 14

        other_pos_start = entity_end      # 14
        other_pos_len = (total_expected - 1) * 2  # (6-1)*2 = 10
        other_pos_end = other_pos_start + other_pos_len  # 24

        other_vel_start = other_pos_end   # 24
        other_vel_len = num_good * 2      # 3*2 = 6
        other_vel_end = other_vel_start + other_vel_len  # 30

        in_forest_start = other_vel_end   # 30
        in_forest_len = 2
        in_forest_end = in_forest_start + in_forest_len  # 32

        # helpers (no try/except)
        def extract_action_id(a_slice):
            a_arr = np.array(a_slice)
            if a_arr.shape == () or a_arr.size == 1:
                return int(a_arr.item())
            return int(np.argmax(a_arr))

        action_legend = {
            0: "no-op",
            1: "move left",
            2: "move right",
            3: "move down",
            4: "move up",
        }

        def action_label(a_id):
            return action_legend.get(a_id, f"act_{a_id}")

        # precompute actions_all (T x N)
        actions_all = [
            [extract_action_id(x_np[t, i, :act_dim]) for i in range(N)] for t in range(T)
        ]

        # allocate per-adversary arrays
        per_adv_nearest_idx = np.full((num_adversaries, T), -1, dtype=int)
        per_adv_nearest_dist = np.full((num_adversaries, T), np.nan, dtype=float)
        per_adv_nearest_rel = np.full((num_adversaries, T, 2), np.nan, dtype=float)

        # parse obs and compute nearest-good for each adversary
        for t in range(T):
            for a_local, a_idx in enumerate(adversary_indices):
                obs_full = x_np[t, a_idx, act_dim : act_dim + obs_dim].astype(float)
                # full 32-d obs; we slice per defined layout
                # entity positions (not used for nearest-good but available)
                # other positions:
                other_flat = obs_full[other_pos_start:other_pos_end]  # length 10
                other_pos = other_flat.reshape(total_expected - 1, 2)  # shape (5,2)

                # other_vel: only non-adversary (3 goods) velocities
                other_vel_flat = obs_full[other_vel_start:other_vel_end]  # length 6
                other_vel = other_vel_flat.reshape(num_good, 2)  # (3,2)

                # find nearest among good agents (global indices 3,4,5)
                best_g = -1
                best_d = None
                best_rel = np.array([np.nan, np.nan], dtype=float)
                for g in good_indices:
                    # map global index g to index in other_pos array (skip self)
                    if g < a_idx:
                        k = g
                    else:
                        k = g - 1
                    rel = other_pos[k]
                    d = float(np.linalg.norm(rel))
                    if best_d is None or d < best_d:
                        best_d = d
                        best_g = int(g)
                        best_rel = rel.copy()

                per_adv_nearest_idx[a_local, t] = best_g
                per_adv_nearest_dist[a_local, t] = float(best_d) if best_d is not None else np.nan
                per_adv_nearest_rel[a_local, t, :] = best_rel

        # Δd computation
        per_adv_delta = np.zeros_like(per_adv_nearest_dist)
        if T > 0:
            per_adv_delta[:, 0] = 0.0
            for tt in range(1, T):
                per_adv_delta[:, tt] = per_adv_nearest_dist[:, tt] - per_adv_nearest_dist[:, tt - 1]

        # team coordination & per-adversary stats
        close_radius = 0.20
        team_coord_steps = 0
        adv_coord_particip = np.zeros((num_adversaries,), dtype=int)
        adv_close_counts = np.zeros((num_adversaries,), dtype=int)

        for t in range(T):
            counts_per_good = {g: 0 for g in good_indices}
            for a_local in range(num_adversaries):
                gidx = int(per_adv_nearest_idx[a_local, t])
                d = per_adv_nearest_dist[a_local, t]
                if gidx >= 0 and not np.isnan(d) and d <= close_radius:
                    counts_per_good[gidx] += 1

            if any(v >= 2 for v in counts_per_good.values()):
                team_coord_steps += 1

            for a_local in range(num_adversaries):
                gidx = int(per_adv_nearest_idx[a_local, t])
                d = per_adv_nearest_dist[a_local, t]
                if gidx >= 0 and not np.isnan(d) and d <= close_radius and counts_per_good[gidx] >= 2:
                    adv_coord_particip[a_local] += 1

            for a_local in range(num_adversaries):
                d = per_adv_nearest_dist[a_local, t]
                if not np.isnan(d) and d <= close_radius:
                    adv_close_counts[a_local] += 1

        # Build textual blocks for each adversary
        adv_blocks = []
        for a_local, a_idx in enumerate(adversary_indices):
            valid_mask = ~np.isnan(per_adv_nearest_dist[a_local, :])
            avg_nearest = float(np.nanmean(per_adv_nearest_dist[a_local, :])) if np.any(valid_mask) else float("nan")

            cnts = Counter([int(per_adv_nearest_idx[a_local, t]) for t in range(T) if per_adv_nearest_idx[a_local, t] >= 0])
            if len(cnts) == 0:
                fav_good = None
                fav_frac = 0.0
            else:
                fav_good, fav_times = cnts.most_common(1)[0]
                fav_frac = fav_times / max(1, sum(cnts.values()))

            improving_steps = 0
            improvements = []
            for tt in range(1, T):
                prev = per_adv_nearest_dist[a_local, tt - 1]
                curr = per_adv_nearest_dist[a_local, tt]
                if not np.isnan(prev) and not np.isnan(curr) and (curr - prev) < 0:
                    improving_steps += 1
                    improvements.append(abs(curr - prev))
            frac_improve = float(improving_steps / max(1, T - 1)) if T > 1 else 0.0
            avg_improve_amt = float(np.mean(improvements)) if len(improvements) > 0 else 0.0
            std_delta = float(np.nanstd(per_adv_delta[a_local, :]))

            first_d = per_adv_nearest_dist[a_local, 0]
            last_d = per_adv_nearest_dist[a_local, -1]
            if np.isnan(first_d) or np.isnan(last_d):
                delta_desc = "no data"
                delta_overall = float("nan")
            else:
                delta_overall = last_d - first_d
                if abs(delta_overall) < 1e-4:
                    delta_desc = "almost unchanged"
                elif delta_overall < 0:
                    delta_desc = f"DECREASES by {abs(delta_overall):.3f} (closer)"
                else:
                    delta_desc = f"INCREASES by {abs(delta_overall):.3f} (farther)"

            timeline_lines = []
            for tt in range(T):
                if mask_np is not None and float(mask_np[tt, a_idx]) < 0.5:
                    continue
                act_id = actions_all[tt][a_idx]
                a_label = action_label(act_id)
                gidx = int(per_adv_nearest_idx[a_local, tt]) if per_adv_nearest_idx[a_local, tt] >= 0 else -1
                dist_val = per_adv_nearest_dist[a_local, tt]
                rel_vec = per_adv_nearest_rel[a_local, tt, :]
                rel_str = f"rel=({rel_vec[0]:.3f},{rel_vec[1]:.3f})" if not np.isnan(rel_vec).all() else "rel=(nan,nan)"
                delta_t = per_adv_delta[a_local, tt] if tt > 0 else 0.0
                timeline_lines.append(
                    f" - t={tt}: action={a_label}; Δd={delta_t:+.3f}; nearest_good={gidx}; distance={0.0 if np.isnan(dist_val) else dist_val:.3f}; {rel_str}"
                )

            block_lines = [
                f"Adversary {a_idx} report:",
                " Metrics:",
                f" Avg nearest-good distance: {avg_nearest:.3f}",
                f" Favorite target (good agent): {fav_good} (fraction {fav_frac:.2f})",
                f" Distance trend across segment: {delta_desc}",
                f" Fraction of steps with distance DECREASE (good steps): {frac_improve:.2f}",
                f" Avg improvement magnitude when improving: {avg_improve_amt:.3f}",
                f" Δd standard deviation (smoothness): {std_delta:.3f}",
                f" Times within close radius ({close_radius:.2f}): {int(adv_close_counts[a_local])}",
                f" Participation in multi-adversary close events: {int(adv_coord_particip[a_local])}",
                " Short timeline (per-step):",
            ] + timeline_lines
            adv_blocks.append("\n".join(block_lines))

        team_coord_frac = float(team_coord_steps / max(1, T))

        # header
        header_lines = [
            "Tactical analyst summary for a hunting scenario.",
            f"This trajectory segment has T={T} timesteps and total_agents (N)={N} (3 adversaries + 3 good).",
            f"Adversaries (learnable): {adversary_indices}",
            f"Good agents (non-learning): {good_indices}",
            f"(obs_dim asserted: expected {expected_obs_dim}, actual {obs_dim}; defined obs by env = {expected_obs_dim})",
            "Task: identify EXACTLY ONE adversary index (from the adversary list above) that is MOST expert-like.",
            "Output format constraint: the very last line of your RESPONSE MUST be exactly one index in the form: [k] (k must be one of the adversary indices).",
            "",
            "Action legend:",
            " 0 = no-op",
            " 1 = move left",
            " 2 = move right",
            " 3 = move down",
            " 4 = move up",
            "",
            "Key interpretation rule (Δd = current_distance - previous_distance):",
            " * Δd < 0 => moved CLOSER to its nearest target since previous step => POSITIVE evidence of hunting skill.",
            " * Δd > 0 => moved AWAY since previous step => NEGATIVE evidence.",
            "",
            "Primary evidence priority (use this order to decide):",
            " 1) sustained DECREASE in distance to targets across the segment (large negative overall Δ).",
            " 2) high fraction of timesteps with Δd < 0 (consistent closing).",
            " 3) participation in multi-adversary close events (shows tactical coordination).",
            " 4) smoothness: smaller Δd variance suggests purposeful approach rather than jittery/random moves.",
            " 5) consistency in preferring the same target (favorite target fraction).",
            "",
            "If ties occur, break them by (1) higher priority above, then (2) lower Δd variance, then (3) higher participation in coordinated events.",
        ]

        if rewards_item is not None:
            total_r = float(np.sum(to_np(rewards_item)))
            header_lines.append(f"Context: team total reward over this segment ≈ {total_r:.3f}.")

        # final assembly
        prompt_parts = header_lines + [""] + adv_blocks + [
            "",
            "Team-level statistics:",
            f" Fraction of timesteps with >=2 adversaries simultaneously very close to the same target: {team_coord_frac:.3f}",
            "",
            "FINAL INSTRUCTION: In no more than 100 words, give a brief rationale (objective, tactical) for your choice,",
            "then on the FINAL LINE output ONLY the chosen adversary index k, in format [k].",
        ]

        info = {
            "adversary_indices": adversary_indices,
            "good_indices": good_indices,
            "num_adversaries": num_adversaries,
            "num_good": num_good,
            "expected_obs_dim": expected_obs_dim,
            "actual_obs_dim": obs_dim,
            "team_coord_frac": team_coord_frac,
        }

        return "\n".join(prompt_parts), info

        
    def _parse_llm_choice(self, text: str, n_agents: int) -> Optional[int]:
        m = re.search(r"\[(\d+)\]", text.split('\n')[-1])
        if m:
            k = int(m.group(1))
            if 0 <= k < n_agents:
                return k
        m = re.search(r"(\d+)", text)
        if m:
            k = int(m.group(1))
            if 0 <= k < n_agents:
                return k
        
        return None
    
    def compute_label_f1_score(self, prediction: torch.Tensor, target: torch.Tensor) -> float:
        with torch.no_grad():
            bs = prediction.shape[0]
            target = target.view(-1)
            prediction = prediction.view(-1)

            TP = ((prediction == 1) & (target == 1)).sum().float()
            TN = ((prediction == 0) & (target == 0)).sum().float()
            FP = ((prediction == 1) & (target == 0)).sum().float()
            FN = ((prediction == 0) & (target == 1)).sum().float()

            epsilon = 1e-8
            precision = TP / (TP + FP + epsilon)
            recall = TP / (TP + FN + epsilon)
            
            f1 = 2 * (precision * recall) / (precision + recall + epsilon)

            return torch.tensor(f1, dtype=torch.float)

    def compute_label_acc(self, prediction: torch.Tensor, target: torch.Tensor):
        # accuracy based on elementwise consistency between prediction and target
        target = target.view(-1)
        prediction = prediction.view(-1)
        return (target == prediction).float().mean()

    def eval_label_model(
        self,
        x: torch.Tensor,
        rewards: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **other_kwargs
    ):
        with torch.no_grad():
            _, label_info = self.compute_label_loss(x, labels)
        return label_info

    def compute_label_loss(
        self,
        x: torch.Tensor,
        target_labels: torch.Tensor,
    ):

        bs, T, na, dim = x.shape
        act_dim = self.action_dim
        obs_dim = self.observation_dim

        x_t = x[:, :, :, act_dim:]
        a_t = x[:, :, :, :act_dim]

        if not self.discrete_action:
            assert 0
            xa_t = x
        else:
            a_onehot_t = torch.nn.functional.one_hot(torch.squeeze(a_t, -1).to(torch.int64), num_classes=self.num_actions).type(torch.FloatTensor).to(x.device)
            xa_t = torch.cat([a_onehot_t, x_t], dim=-1)  # (bs, h, na, num_actions + obs_dim)

        model_input = xa_t

        logits = self.label_model(
            model_input,
        )

        targets = target_labels.to(logits.dtype)

        # take softmax to predict a single expert agent

        pred_bin = F.one_hot(torch.argmax(logits.squeeze(-1), dim=1), num_classes=logits.size(1)).float().unsqueeze(-1)

        info = {
            "label_model_f1_score": self.compute_label_f1_score(pred_bin, targets),
            "label_model_acc": self.compute_label_acc(pred_bin, targets)
        }

        logits = logits.squeeze(-1)
        probs = F.softmax(logits, dim=1).clamp(min=1e-8)
        targets = targets.squeeze(-1)
        label_loss = -torch.sum(targets * torch.log(probs), dim=1).mean()

        return label_loss, info

    def collect_label_batch(
        self,
        x: torch.Tensor,
        batch_idx: int = None,
        use_llm_labels: int = True,
        prompt_handler: str = '',
        task_dir: str = '',
        loss_masks_world_models: Optional[torch.Tensor] = None,
        rewards: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs,
    ):
    
        bs, T, na, _ = x.shape
        device = x.device
        llm = LLM(mode='openai')
        llm_labels = torch.zeros((bs, na, 1), device=device)


        build_llm_prompt = getattr(self, f'_build_llm_prompt_' + prompt_handler)

        true_labels = (labels > 0.5).float().max(dim=1).values  # [bs, na, 1]
        true_choices = np.full(bs, -1)
        llm_choices = np.full(bs, -1)

        for b in tqdm(range(bs)):
            prompt, prompt_info = build_llm_prompt(x[b], 
                    rewards[b] if rewards is not None else None,
                    loss_masks_world_models[b] if loss_masks_world_models is not None else None
            )

            try:
                assert use_llm_labels
                resp = llm.call_llm(prompt, big_model=False)
                choice = self._parse_llm_choice(resp, na)
                assert choice is not None
            except Exception:
                print('[WARNING] LLM failed to respond. Falling back. ')
                choice = np.random.randint(na)
                resp = f"(fake resp, choice [{choice}])"
            
            llm_choices[b] = choice
            real_choice = np.nonzero(true_labels[b].flatten())[0][0] # true_labels.flatten().nonzero()[0].item()
            true_choices[b] = real_choice

            lbl = torch.zeros(na, 1, device=device, dtype=torch.float32)
            lbl[choice, 0] = 1.0
            llm_labels[b] = lbl

        info = {}
        if labels is not None:
            info["llm_label_f1_score"] = self.compute_label_f1_score(llm_labels, true_labels)

        x_flat = x.reshape(bs, T, -1).cpu().numpy()
        rewards = rewards.cpu().numpy()
        llm_labels = llm_labels.cpu().numpy()
        true_labels = true_labels.cpu().numpy()

        oar = np.concatenate([x_flat, rewards], axis=-1)
        
        llm_labels_squeezed = np.squeeze(llm_labels, axis=-1)
        true_labels_squeezed = np.squeeze(true_labels, axis=-1)
            
        assert batch_idx != None
        batches_dir = f"diffuser/datasets/data/{task_dir}/batches"
        os.makedirs(batches_dir, exist_ok=True)
        np.save(f"{batches_dir}/oar_batch_{batch_idx}.npy", oar)
        if use_llm_labels: np.save(f"{batches_dir}/llm_labels_batch_{batch_idx}.npy", llm_labels_squeezed)
        np.save(f"{batches_dir}/true_labels_batch_{batch_idx}.npy", true_labels_squeezed)
        print(f"successfully saved batch {batch_idx}")

        label_loss = dummy_loss = torch.tensor(0.0, requires_grad=True)
        return label_loss, info

    def loss(
        self,
        x: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        # for label model training

        assert labels is not None
        bs, T, na, _ = x.shape
        device = x.device
        label_loss, label_info = self.compute_label_loss(x, labels)
        return label_loss, label_info
