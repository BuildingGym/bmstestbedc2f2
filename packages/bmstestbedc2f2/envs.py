import functools as _functools_
from typing import Callable, Literal

import numpy as _numpy_

from controllables.core import TemporaryUnavailableError
from controllables.core.tools.gymnasium import DiscreteSpace, BoxSpace, DictSpace, BaseAgent
from controllables.core.tools.rllib import MultiAgentEnv, Env

from ray.rllib.algorithms import AlgorithmConfig
from ray.rllib.policy.policy import PolicySpec

from .systems import ProtoBMSystem, ManualBMSystem, EnergyPlusBMSystem
from .rewards import ComfortElecSavingRewardFunction, ComfortElecSavingVectorRewardFunction


def get_reward_fn(
    name_or_fn: Literal['elec-saving-diff', 'elec-saving-vector'] | Callable
):
    match name_or_fn:
        case 'elec-saving-diff':
            return ComfortElecSavingRewardFunction()
        case 'elec-saving-vector':
            return ComfortElecSavingVectorRewardFunction()
        case _:
            pass
    return name_or_fn


class MultiAgentBuildingEnv(MultiAgentEnv):
    config: MultiAgentEnv.Config = {
        'agents': {},
    }

    class RoomAgentRewardFunction:
        def __init__(self, name='elec-saving-vector'):
            self._base_fn = get_reward_fn(name)

        def __call__(self, agent: BaseAgent) -> float:
            try:
                return self._base_fn({
                    'hvac_load': agent.observation.value['load:hvac'], 
                    'office_occupancy': agent.observation.value['occupancy'], 
                    'temperature_drybulb': agent.observation.value['temperature:drybulb'], 
                    'temperature_radiant': agent.observation.value['temperature:radiant'], 
                    'humidity': agent.observation.value['humidity'],
                })
            except TemporaryUnavailableError:
                return 0.

    for zone_id in ProtoBMSystem.zone_ids:
        config['agents'][zone_id] = {
            'action_space': DictSpace({
                'setpoint:thermostat': BoxSpace(
                    low=20., high=30.,
                    dtype=_numpy_.float32,
                    shape=(),
                ).bind(
                    (zone_id, 'setpoint:thermostat')
                ),
            }),
            # NOTE using `.cast` because the stupid rllib has a bug!!!
            'observation_space': DictSpace({
                'temperature:drybulb': BoxSpace(
                    low=-_numpy_.inf, high=+_numpy_.inf,
                    dtype=_numpy_.float32,
                    shape=(),
                ).bind(
                    lambda agent, zone_id=zone_id: 
                    agent[(zone_id, 'temperature:drybulb')].cast(_numpy_.array)
                ),
                'temperature:radiant': BoxSpace(
                    low=-_numpy_.inf, high=+_numpy_.inf,
                    dtype=_numpy_.float32,
                    shape=(),
                ).bind(
                    lambda agent, zone_id=zone_id: 
                    agent[(zone_id, 'temperature:radiant')].cast(_numpy_.array)
                ),
                'humidity': BoxSpace(
                    low=-_numpy_.inf, high=+_numpy_.inf,
                    dtype=_numpy_.float32,
                    shape=(),
                ).bind(
                    lambda agent, zone_id=zone_id:
                    agent[(zone_id, 'humidity')].cast(_numpy_.array)
                ),
                'co2': BoxSpace(
                    low=-_numpy_.inf, high=+_numpy_.inf,
                    dtype=_numpy_.float32,
                    shape=(),
                ).bind(
                    lambda agent, zone_id=zone_id:
                    agent[(zone_id, 'co2')].cast(_numpy_.array)
                ),
                'load:hvac': BoxSpace(
                    low=-_numpy_.inf, high=+_numpy_.inf,
                    dtype=_numpy_.float32,
                    shape=(),
                ).bind(
                    lambda agent, zone_id=zone_id:
                    agent[(zone_id, 'load:hvac')].cast(_numpy_.array)
                ),
                'occupancy': BoxSpace(
                    low=-_numpy_.inf, high=+_numpy_.inf,
                    dtype=_numpy_.float32,
                    shape=(),
                ).bind(
                    lambda agent, zone_id=zone_id:
                    agent[(zone_id, 'occupancy')].cast(_numpy_.array)
                ),
            }),
            'reward': RoomAgentRewardFunction(),
        }

    def __init__(self, config: dict = dict()):
        self._bms_system_factory: Callable[[], ProtoBMSystem]
        match config.get('bms_system', 'energyplus'):
            case 'manual':
                self._bms_system_factory = lambda: ManualBMSystem().start()
            case 'energyplus':
                self._bms_system_factory = lambda: EnergyPlusBMSystem(repeat=True).start()
            case x:
                self._bms_system_factory = x
                
        super().__init__({
            **self.__class__.config,
            **config,
        })

    @_functools_.cached_property
    def system(self):
        return self._bms_system_factory()

    def run(self):
        self.attach(self.system).schedule_episode(errors='warn')
        self.system.wait()

    @classmethod
    def get_algo_config(cls, base_config: AlgorithmConfig, config: dict = dict(), **config_kwds):
        return (
            base_config
            .environment(cls)
            .env_runners(
                # NOTE this env (an `ExternalEnv`) does not support connectors
                enable_connectors=False,
            )
            .multi_agent(
                policies={
                    policy_id: PolicySpec(
                        action_space=agent_config['action_space'],
                        observation_space=agent_config['observation_space'],
                    )
                    for policy_id, agent_config in cls.config['agents'].items()
                },
                policy_mapping_fn=lambda agent_id, *args, **kwargs: str(agent_id),
            )
            .update_from_dict({**config, **config_kwds})
        )