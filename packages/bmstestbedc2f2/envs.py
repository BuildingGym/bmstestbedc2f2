import functools as _functools_
from typing import Callable, Literal

import numpy as _numpy_

from controllables.core import TemporaryUnavailableError
from controllables.core.tools.gymnasium import DiscreteSpace, BoxSpace, DictSpace, BaseAgent
from controllables.core.tools.rllib import MultiAgentEnv, Env

from ray.rllib.algorithms import AlgorithmConfig
from ray.rllib.policy.policy import PolicySpec

from .systems import ProtoBMSystem, ManualBMSystem, EnergyPlusBMSystem
from .rewards import ComfortElecSavingRewardFunction, ComfortElecSavingVectorRewardFunction, HeuristicComfortElecSavingRewardFunction


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
        self.attach(self.system)
        self.schedule_episode(errors='warn')
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
    

from controllables.core import BaseVariable

from .systems import _TODO_ProtoBMSystem, _TODO_ManualBMSystem, _TODO_EnergyPlusBMSystem
from .rewards import HeuristicComfortElecSavingRewardFunction


class DeltaVariable(BaseVariable):
    def __init__(
        self, 
        variable: BaseVariable, 
    ):
        self._variable = variable
        self._prev_value = None

    @property
    def value(self):
        value = self._variable.value
        delta = (
            (value - self._prev_value) 
            if self._prev_value is not None else 
            0
        )
        self._prev_value = value
        return delta
    

class _TODO_MultiAgentBuildingEnv(MultiAgentEnv):
    config: MultiAgentEnv.Config = {
        'agents': {},
    }

    class RoomAgentRewardFunction:
        def __init__(self, zone_id: str):
            self._zone_id = zone_id
            self._base_fn = HeuristicComfortElecSavingRewardFunction()

        def __call__(self, agent: BaseAgent) -> float:
            try:
                return self._base_fn({
                    'temperature': agent[(self._zone_id, 'temperature')].value,
                    'temperature_pref': agent[(self._zone_id, 'temperature:userpref')].value,
                    'hvac_load': agent[(self._zone_id, 'load:hvac')].value,
                })
            except TemporaryUnavailableError:
                return 0.

    for zone_id in _TODO_ProtoBMSystem.zone_ids:
        config['agents'][zone_id] = {
            'action_space': DictSpace({
                'temperature:thermostat': BoxSpace(
                    low=20., high=30.,
                    dtype=_numpy_.float32,
                    shape=(),
                ).bind(
                    (zone_id, 'temperature:thermostat')
                ),
            }),
            # NOTE using `.cast` because the stupid rllib has a bug!!!
            'observation_space': DictSpace({
                # TODO
                'energy': BoxSpace(
                    low=-1, high=+1,
                    dtype=_numpy_.float32,
                    shape=(),
                ).bind(
                    # TODO NOTE no vec norm here because the rllib does not support it!!!
                    lambda agent, zone_id=zone_id: 
                    (
                        (agent[(zone_id, 'temperature:thermostat')] - agent[(zone_id, 'temperature')])
                        #/ (agent[(zone_id, 'temperature')])
                    )
                    .cast(_numpy_.array)
                ),
                'comfort': BoxSpace(
                    low=-1, high=+1,
                    dtype=_numpy_.float32,
                    shape=(),
                ).bind(
                    lambda agent, zone_id=zone_id: 
                    (
                        (agent[(zone_id, 'temperature:userpref')] - agent[(zone_id, 'temperature')])
                        #/ (agent[(zone_id, 'temperature')])
                    )
                    .cast(_numpy_.array)
                ),
                # TODO !!!!
                # 'occupancy': BoxSpace(
                #     low=-1, high=+1,
                #     dtype=_numpy_.float32,
                #     shape=(),
                # ).bind(
                #     lambda agent, zone_id=zone_id:
                #     agent[(zone_id, 'temperature')]
                #     .cast(_numpy_.array)
                # ),
                'power': BoxSpace(
                    low=-1, high=+1,
                    dtype=_numpy_.float32,
                    shape=(),
                ).bind(
                    lambda agent, zone_id=zone_id:
                    agent[(zone_id, 'load:hvac')]
                    .cast(_numpy_.array)
                ),
                # 'trend:power': ...,
            }),
            'reward': RoomAgentRewardFunction(zone_id=zone_id),
        }


    def __init__(self, config: dict = dict()):
        self._bms_system_factory: Callable[[], ProtoBMSystem]
        match config.get('bms_system', 'energyplus'):
            case 'manual':
                self._bms_system_factory = lambda: _TODO_ManualBMSystem().start()
            case 'energyplus':
                self._bms_system_factory = lambda: _TODO_EnergyPlusBMSystem(repeat=True).start()
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
        self.attach(self.system)
        self.schedule_episode(errors='raise')
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