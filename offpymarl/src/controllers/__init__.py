REGISTRY = {}

from .basic_controller import BasicMAC
from .random_controller import RandomMAC
from .maddpg_controller import MADDPGMAC
from .madiff_controller import MADIFFMAC

REGISTRY["basic_mac"] = BasicMAC
REGISTRY['random_mac'] = RandomMAC
REGISTRY['maddpg_mac'] = MADDPGMAC
REGISTRY["madiff_mac"] = MADIFFMAC
