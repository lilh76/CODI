from gym.envs.registration import registry, register, make, spec
from itertools import product

sizes = [15]
players = [3]
coop = [False]
foods = [3]
for s, p, f, c in product(sizes, players, foods, coop):
    register(
        id="Foraging-{0}x{0}-{1}p-{2}f{3}-v1".format(s, p, f, "-coop" if c else ""),
        entry_point="lbforaging.foraging:ForagingEnv",
        kwargs={
            "players": p,
            "max_player_level": 3,
            "field_size": (s, s),
            "max_food": f,
            "sight": s,
            "max_episode_steps": 500,
            "force_coop": c,
        },
    )
