from .rewards import HeuristicComfortElecSavingRewardFunction
from .systems import _TODO_ProtoBMSystem, _TODO_ManualBMSystem, _TODO_EnergyPlusBMSystem, _TODO_NeuralBMSystem, _TODO_NeuralBMSystemV2
from controllables.core import BaseVariable
import functools as _functools_
from typing import Callable, Literal

import numpy as _numpy_

from controllables.core import TemporaryUnavailableError
from controllables.core import ComputedVariable
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
                self._bms_system_factory = lambda: EnergyPlusBMSystem(
                    repeat=True).start()
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
                policy_mapping_fn=lambda agent_id, *
                args, **kwargs: str(agent_id),
            )
            .update_from_dict({**config, **config_kwds})
        )


# TODO
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
                    # 'office_occupancy': agent[(self._zone_id, 'occupancy')].value,
                    # 'temperature': agent[(self._zone_id, 'temperature')].value,
                    'temperature_pref': agent[(self._zone_id, 'temperature:userpref')].value,
                    # 'hvac_load': agent[(self._zone_id, 'load:hvac')].value,
                    'ahu_load': agent.observation.value['load:ahu'],
                    'office_occupancy': agent.observation.value['occupancy'],
                    'temperature': agent.observation.value['temperature'],
                })
            except TemporaryUnavailableError:
                return 0.

    for zone_id in _TODO_ProtoBMSystem.zone_ids:
        config['agents'][zone_id] = {
            'action_space': DictSpace({
                # 'temperature:thermostat': DiscreteSpace(
                #     n=10, start=20
                # ).bind(
                #     (zone_id, 'temperature:thermostat')
                # ),
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
                    low=-_numpy_.inf, high=_numpy_.inf,
                    dtype=_numpy_.float32,
                    shape=(),
                ).bind(
                    # TODO NOTE no vec norm here because the rllib does not support it!!!
                    lambda agent, zone_id=zone_id:
                        ComputedVariable(
                            lambda temp_t, temp:
                                _numpy_.nan_to_num(
                                    _numpy_.float32(
                                        temp_t.value - temp.value) / temp.value
                                ),
                            temp_t=agent[(zone_id, 'temperature:thermostat')],
                            temp=agent[(zone_id, 'temperature')],
                        )
                ),
                'comfort': BoxSpace(
                    low=-_numpy_.inf, high=_numpy_.inf,
                    dtype=_numpy_.float32,
                    shape=(),
                ).bind(
                    lambda agent, zone_id=zone_id:
                        ComputedVariable(
                            lambda temp_pref, temp:
                                _numpy_.nan_to_num(
                                    _numpy_.float32(
                                        temp_pref.value - temp.value) / temp.value
                                ),
                            temp_pref=agent[(zone_id, 'temperature:userpref')],
                            temp=agent[(zone_id, 'temperature')],
                        )
                ),
                # TODO !!!!
                # 'occupancy': BoxSpace(
                #     low=-_numpy_.inf, high=_numpy_.inf,
                #     dtype=_numpy_.float32,
                #     shape=(),
                # ).bind(
                #     lambda agent, zone_id=zone_id:
                #         ComputedVariable(
                #             lambda temp_trend, temp:
                #                 _numpy_.nan_to_num(
                #                     _numpy_.float32(
                #                         temp_trend.value) / temp.value
                #                 ),
                #                 temp_trend=agent[(
                #                     zone_id, 'trend:temperature')],
                #                 temp=agent[(zone_id, 'temperature')],
                #         )
                # ),
                'load:ahu': BoxSpace(
                    low=-_numpy_.inf, high=_numpy_.inf,
                    dtype=_numpy_.float32,
                    shape=(),
                ).bind(
                    lambda agent, zone_id=zone_id:
                        ComputedVariable(
                            lambda load:
                                _numpy_.nan_to_num(load.value),
                            load=agent['load:ahu'],
                        )
                ),
                # 'trend:power': BoxSpace(
                #     low=-1, high=+1,
                #     dtype=_numpy_.float32,
                #     shape=(),
                # ).bind(
                #     lambda agent, zone_id=zone_id:
                #         ComputedVariable(
                #             lambda power_trend, power:
                #                 _numpy_.clip(
                #                     _numpy_.nan_to_num(
                #                         _numpy_.float32(power_trend.value) / (power.value+1e-6),
                #                     ),
                #                     -1, +1,
                #                 ),
                #             power_trend=agent[(zone_id, 'trend:load:hvac')],
                #             power=agent[(zone_id, 'load:hvac')]
                #         )
                # ),
                'temperature': BoxSpace(
                    low=-_numpy_.inf, high=+_numpy_.inf,
                    dtype=_numpy_.float32,
                    shape=(),
                ).bind(
                    lambda agent, zone_id=zone_id:
                    agent[(zone_id, 'temperature')].cast(_numpy_.array)
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
            'reward': RoomAgentRewardFunction(zone_id=zone_id),
        }

    def __init__(self, config: dict = dict()):
        self._bms_system = config.get('bms_system', 'energyplus')

        # self._bms_system_factory: Callable[[], ProtoBMSystem]
        # match config.get('bms_system', 'energyplus'):
        #     case 'manual':
        #         self._bms_system_factory = lambda: _TODO_ManualBMSystem()
        #     case 'energyplus':
        #         self._bms_system_factory = lambda: _TODO_EnergyPlusBMSystem(
        #             repeat=True).start()
        #     case 'neural':
        #         # TODO
        #         self._bms_system_factory = lambda: _TODO_NeuralBMSystem().start()
        #     case 'neural_v2':
        #         def neural_bms_system_factory():
        #             system = _TODO_NeuralBMSystemV2()
        #             while True:
        #                 system.start()
        #                 for _ in range(100):
        #                     system.step()
        #                 system.stop()

        #                 # TODO
        #                 print('TODO end')
        #                 for key, value in system.items():
        #                     print(f"{key}: {value.value}")

        #         self._bms_system_factory = neural_bms_system_factory
        #     case x:
        #         self._bms_system_factory = x

        super().__init__({
            **self.__class__.config,
            **config,
        })

    # @_functools_.cached_property
    # def system(self):
    #     return self._bms_system_factory()

    def run(self):
        match self._bms_system:
            case 'manual':
                self.attach(_TODO_ManualBMSystem())
                self.schedule_episode(errors='warn')
                self.system.wait()
            case 'energyplus':
                self.attach(_TODO_EnergyPlusBMSystem(
                    repeat=True
                ))
                self.schedule_episode(errors='warn')
                self.system.start()
                self.system.wait()
            case 'neural':
                raise NotImplementedError
            case 'neural_v2':
                # TODO
                self.attach(_TODO_NeuralBMSystemV2())
                self.schedule_episode(errors='warn')
                self.system.restore()
                while True:
                    self.system.start()
                    for _ in range(1_000):
                        self.system.step()
                    self.system.stop()
                    # TODO
                    # print('TODO end')
                    # for key, value in self.system.items():
                    #     print(f"{key}: {value.value}")
            case run_func:
                run_func(self)
                # self.attach(factory_func())
                # self.schedule_episode(errors='warn')
                # self.system.wait()

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
