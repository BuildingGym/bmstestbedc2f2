
# TODO
from controllables.core import Variable
from controllables.energyplus import (
    Actuator, 
    OutputVariable, 
    OutputMeter,
    System,
)
from bmstestbedc2f2.utils import resolve_path
from bmstestbedc2f2.systems.common import ProtoBMSystem, _TODO_ProtoBMSystem


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


from controllables.core.variables import VariableManager
from bmstestbedc2f2.systems.common import _make_trend_var


class _TODO_EnergyPlusBMSystem(_TODO_ProtoBMSystem):
    def __init__(self, **kwargs):
        self._system = System(
            building=resolve_path('model.idf'),
            weather=resolve_path('weather_sin.epw'),
            **kwargs,
        )
        
        #self._system.add('logging:progress')
        super().__init__()


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
            case 'load:ahu':
                return self._system[
                    OutputVariable.Ref(
                        type='Fan Electricity Rate',
                        key='AIR LOOP AHU SUPPLY FAN',
                    )
                ] / 8_000

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
                    # case 'load:hvac':
                    #     return self._system[
                    #         OutputMeter.Ref(
                    #             'Electricity:HVAC',
                    #         )
                    #     ] / 14_000_000
                    case 'occupancy':
                        return self._system[
                            OutputVariable.Ref(
                                type='Schedule Value',
                                key='Office_OpenOff_Occ',
                            )
                        ]
                    case 'load:ahu':
                        return self._system[
                             OutputVariable.Ref(
                                type='Fan Electricity Rate',
                                key='AIR LOOP AHU SUPPLY FAN',
                            )
                        ] / 8_000

        return self._basevars[slot]
    