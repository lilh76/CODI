from gym.envs.registration import register
import mpe.scenarios as scenarios

def _register(scenario_name, gymkey):
    scenario = scenarios.load(scenario_name + ".py").Scenario()
    world = scenario.make_world()
    register(
        gymkey,
        entry_point="mpe.environment:MultiAgentEnv",
        kwargs={
            "world": world,
            "reset_callback": scenario.reset_world,
            "reward_callback": scenario.reward,
            "observation_callback": scenario.observation,
            # "done_callback": scenario.done,
        },
    )

scenario_name = "simple_spread"
gymkey = "SimpleSpread-v0"
_register(scenario_name, gymkey)

scenario_name = "simple_spread_4"
gymkey = "SimpleSpread4-v0"
_register(scenario_name, gymkey)

scenario_name = "simple_spread_5"
gymkey = "SimpleSpread5-v0"
_register(scenario_name, gymkey)

scenario_name = "simple_tag_3"
gymkey = "SimpleTag3-v0"
_register(scenario_name, gymkey)

scenario_name = "simple_world_3"
gymkey = "SimpleWorld3-v0"
_register(scenario_name, gymkey)
