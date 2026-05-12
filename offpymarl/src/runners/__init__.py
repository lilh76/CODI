REGISTRY = {}

from .episode_runner import EpisodeRunner
REGISTRY["episode"] = EpisodeRunner

from .parallel_runner import ParallelRunner
REGISTRY["parallel"] = ParallelRunner

from .episode_runner_imbalance import EpisodeRunnerImbalance
REGISTRY["episode_imbalance"] = EpisodeRunnerImbalance
REGISTRY["episode"] = EpisodeRunnerImbalance