# TODO
from controllables.core import Variable
from controllables.core.systems import ProtoProcess, SimpleProcess, BaseSystem


class ProtoBMSystem(ProtoProcess, BaseSystem):
    zone_ids = [
        'ART-01-07', 
        'ART-01-08', 
        'ART-01-09',
        'ART-01-10',
        'ART-01-11a',
        'ART-01-12',
        'ART-01-13',
        'ART-01-14',
    ]

    # TODO std
    __variable_slots__ = ['time']
    for zone_id in zone_ids:
        __variable_slots__ += [
            (zone_id, 'temperature:drybulb'),
            (zone_id, 'temperature:radiant'),
            (zone_id, 'humidity'),
            (zone_id, 'power:hvac'),
            (zone_id, 'load:hvac'),
            (zone_id, 'occupancy'),
            (zone_id, 'setpoint:thermostat'),
        ]


class ManualBMSystem(SimpleProcess, ProtoBMSystem):
    def __init__(self):
        super().__init__(slots=self.__variable_slots__)


from controllables.energyplus import (
    Actuator, 
    OutputVariable, 
    OutputMeter,
    System,
)

from ..utils import resolve_path


class EnergyPlusBMSystem(ProtoBMSystem):
    def __init__(self, **kwargs):
        super().__init__()

        self._system = System(
            building=resolve_path('model.idf'),
            weather=resolve_path('weather_sin.epw'),
            **kwargs,
        )
        # self._system.add('logging:progress')

        # TODO
        self.events = self._system.events
    
    def start(self):
        self._system.start()
        return self

    def stop(self):
        self._system.stop()
        return self

    def wait(self, timeout=None):
        self._system.wait(timeout=timeout)
        return self

    # TODO
    def __getitem__(self, slot):
        match slot:
            case (zone_id, key):
                ref_zone_id = {
                    'ART-01-07': '1FFIRSTFLOORWEST:OPENOFFICE',
                    'ART-01-08': '1FFIRSTFLOOREAST:OPENOFFICE',
                    'ART-01-09': '0FGROUNDFLOORWEST:OPENOFFICE',
                    'ART-01-10': '0FGROUNDFLOOREAST:OPENOFFICE',
                    'ART-01-11a': '1FFIRSTFLOORWEST1:OPENOFFICE',
                    'ART-01-12': '1FFIRSTFLOOREAST1:OPENOFFICE',
                    'ART-01-13': '0FGROUNDFLOORWEST1:OPENOFFICE',
                    'ART-01-14': '0FGROUNDFLOOREAST1:OPENOFFICE',
                }[zone_id]

                match key:
                    case 'temperature:drybulb':
                        return self._system[
                            OutputVariable.Ref(
                                'Zone Mean Air Temperature',
                                ref_zone_id,
                            )
                        ]
                    case 'temperature:radiant':
                        return self._system[
                            OutputVariable.Ref(
                                'Zone Mean Radiant Temperature',
                                ref_zone_id,
                            )
                        ]
                    case 'humidity':
                        return self._system[
                            OutputVariable.Ref(
                                'Zone Air Relative Humidity',
                                ref_zone_id,
                            )
                        ]
                    case 'co2':
                        return self._system[
                            OutputVariable.Ref(
                                'Zone Air CO2 Concentration',
                                ref_zone_id,
                            )
                        ]
                    case 'power:hvac':
                        return self._system[
                            OutputMeter.Ref(
                                'Electricity:HVAC',
                            )
                        ]
                    case 'load:hvac':
                        return self._system[
                            OutputMeter.Ref(
                                'Electricity:HVAC',
                            )
                        ] / 14_000_000
                    case 'load:ahu':
                        return self._system[
                             OutputVariable.Ref(
                                type='Fan Electricity Rate',
                                key='AIR LOOP AHU SUPPLY FAN',
                            )
                        ] / 1_000
                    case 'occupancy':
                        return self._system[
                            OutputVariable.Ref(
                                type='Schedule Value',
                                key='Office_OpenOff_Occ',
                            )
                        ]
                    case 'setpoint:thermostat':
                        return self._system[
                            Actuator.Ref(
                                'Schedule:Compact',
                                'Schedule Value',
                                f'{ref_zone_id} COOLING SETPOINT SCHEDULE',
                            )
                        ]
            case x:
                return self._system[x]

        raise ValueError(slot)
    
    # TODO
    def __contains__(self, slot):
        return slot in self.__variable_slots__ or slot in self._system


# TODO
class _TODO_ProtoBMSystem(ProtoProcess, BaseSystem):
    zone_ids = [
        'ART-01-07', 
        'ART-01-08', 
        'ART-01-09',
        'ART-01-10',
        'ART-01-11a',
        'ART-01-12',
        'ART-01-13',
        'ART-01-14',
    ]

    # TODO std
    __variable_slots__ = ['time']
    for zone_id in zone_ids:
        __variable_slots__ += [
            (zone_id, 'temperature'),
            (zone_id, 'temperature:userpref'),
            (zone_id, 'temperature:thermostat'),
            (zone_id, 'load:hvac'),
        ]

class _TODO_ManualBMSystem(SimpleProcess, _TODO_ProtoBMSystem):
    def __init__(self):
        super().__init__(slots=self.__variable_slots__)

class _TODO_EnergyPlusBMSystem(_TODO_ProtoBMSystem):
    def __init__(self, **kwargs):
        super().__init__()

        self._system = System(
            building=resolve_path('model.idf'),
            weather=resolve_path('weather_sin.epw'),
            **kwargs,
        )
        # self._system.add('logging:progress')

    @property
    def events(self):
        return self._system.events
    
    def start(self):
        self._system.start()
        return self

    def stop(self):
        self._system.stop()
        return self

    def wait(self, timeout=None):
        self._system.wait(timeout=timeout)
        return self

    # TODO
    def __getitem__(self, slot):
        match slot:
            case (zone_id, key):
                ref_zone_id = {
                    'ART-01-07': '1FFIRSTFLOORWEST:OPENOFFICE',
                    'ART-01-08': '1FFIRSTFLOOREAST:OPENOFFICE',
                    'ART-01-09': '0FGROUNDFLOORWEST:OPENOFFICE',
                    'ART-01-10': '0FGROUNDFLOOREAST:OPENOFFICE',
                    'ART-01-11a': '1FFIRSTFLOORWEST1:OPENOFFICE',
                    'ART-01-12': '1FFIRSTFLOOREAST1:OPENOFFICE',
                    'ART-01-13': '0FGROUNDFLOORWEST1:OPENOFFICE',
                    'ART-01-14': '0FGROUNDFLOOREAST1:OPENOFFICE',
                }[zone_id]

                match key:
                    case 'temperature':
                        return self._system[
                            OutputVariable.Ref(
                                'Zone Mean Air Temperature',
                                ref_zone_id,
                            )
                        ]
                    case 'temperature:userpref':
                        # TODO
                        return Variable(25.)
                    case 'temperature:thermostat':
                        return self._system[
                            Actuator.Ref(
                                'Schedule:Compact',
                                'Schedule Value',
                                f'{ref_zone_id} COOLING SETPOINT SCHEDULE',
                            )
                        ]
                    case 'load:hvac':
                        return self._system[
                            OutputMeter.Ref(
                                'Electricity:HVAC',
                            )
                        ] / 14_000_000                    
            case x:
                return self._system[x]

        raise ValueError(slot)
    
    # TODO
    def __contains__(self, slot):
        return slot in self.__variable_slots__ or slot in self._system


class _TODO_NeuralBMSSystem(SimpleProcess, _TODO_ProtoBMSystem):
    def __init__(self):
        super().__init__(slots=self.__variable_slots__)

    def step(self, state, /, **state_kwds):
        return super().step(state, **state_kwds)
