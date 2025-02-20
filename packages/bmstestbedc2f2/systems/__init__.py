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
            # TODO
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



from controllables.core import TemporaryUnavailableError


# TODO rm BaseSystem
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
    __variable_slots__ = ['time', 'load:ahu', 'trend:load:ahu']
    for zone_id in zone_ids:
        __variable_slots__ += [
            (zone_id, 'temperature'), # 
            (zone_id, 'temperature:userpref'), # 
            (zone_id, 'temperature:thermostat'), #
            # (zone_id, 'load:ahu'), # TODO rm
            (zone_id, 'trend:temperature'),
            # (zone_id, 'trend:load:ahu'), # TODO rm
        ]

    def __init__(self):
        self._basevars = dict()
        
        for key in ('trend:temperature', 'trend:load:ahu'):
            for zone_id in self.zone_ids:
                self._basevars[zone_id, key] = MutableVariable(0.)

        prev_values = dict()
        @self.events['step'].on
        def _(*args, **kwargs):
            for zone_id in self.zone_ids:
                for trend_key, key in [
                    [(zone_id, 'trend:temperature'), (zone_id, 'temperature')], 
                    #[(zone_id, 'trend:load:ahu'), (zone_id, 'load:ahu')],
                ]:
                    try:
                        if key in prev_values:
                            self._basevars[trend_key].value \
                                = self[key].value - prev_values[key]
                        # TODO !!!!!!!!!
                        prev_values[key] = self[key].value + 1e-6
                    except TemporaryUnavailableError:
                        pass
        
        prev_load_value = None
        self._basevars['trend:load:ahu'] = MutableVariable(0.)
        @self.events['step'].on
        def _(*args, **kwargs):
            nonlocal prev_load_value
            try:
                if prev_load_value is not None:
                    self._basevars['trend:load:ahu'].value \
                        = self['load:ahu'].value - prev_load_value
                prev_load_value = self['load:ahu'].value
            except TemporaryUnavailableError:
                pass

    def __getitem__(self, slot):
        if slot in self._basevars:
            return self._basevars[slot]
        return super().__getitem__(slot)
    
    def __contains__(self, slot):
        return slot in self.__variable_slots__ or slot in self._basevars
    


# TODO
class _TODO_ManualBMSystem(SimpleProcess, _TODO_ProtoBMSystem):
    def __init__(self):
        super().__init__()

        for slot in self.__variable_slots__:
            self[slot] = MutableVariable()


from controllables.core import MutableVariable


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

        return super().__getitem__(slot)
    

from bmstestbedc2f2_pred.utils import xarray_acc
from bmstestbedc2f2_pred.model_power import PowerModel
from bmstestbedc2f2_pred.model_temp import RoomTemperatureModel

from controllables.core.callbacks import CallbackManager
import datetime

# TODO
class _TODO_NeuralBMSSystem(dict, _TODO_ProtoBMSystem):
    def __init__(self):
        self.events = CallbackManager()

        super().__init__(self)

        _TODO_ProtoBMSystem.__init__(self)

        # TODO
        self._power_input_buffer = None
        self._power_model = PowerModel()

        self._temp_input_buffer = None
        self._temp_model = RoomTemperatureModel(num_rooms=8)

        self['time'] = MutableVariable()

        self['load:ahu'] = MutableVariable(0)
        for zone_id in self.zone_ids:
            self[zone_id, 'temperature'] = MutableVariable(25)
            self[zone_id, 'temperature:userpref'] = MutableVariable(25)
            self[zone_id, 'temperature:thermostat'] = MutableVariable(25)

        @self.events['step'].on
        def _():
            self['time'].value = datetime.datetime.now()

        @self.events['step'].on
        def _():
            self._power_input_buffer = xarray_acc(
                self._power_input_buffer,
                PowerModel.Input([[[self['load:ahu'].value]]]),
                dim='Time',
                # TODO customize max lookback
                maxlen=32,
            )
            [[next_power_data]] = self._power_model.predict(self._power_input_buffer)
            self['load:ahu'].value = next_power_data

            self._temp_input_buffer = xarray_acc(
                self._temp_input_buffer,
                RoomTemperatureModel.Input([[[
                    [
                        self[zone_id, 'temperature'].value,
                        self[zone_id, 'trend:temperature'].value,
                        self[zone_id, 'temperature'].value - self[zone_id, 'temperature:thermostat'].value,
                    ] 
                    for zone_id in self.zone_ids
                ]]]),
                dim='Time',
                # TODO customize max lookback
                maxlen=32,
            )
            [next_zone_temp_data] = self._temp_model.predict(self._temp_input_buffer)
            for zone_id, [temperature] in zip(self.zone_ids, next_zone_temp_data):
                self[zone_id, 'temperature'].value = temperature

    # TODO ...
    def step(self):
        self.events['step']()
        return self

    # TODO !!!!
    def __getitem__(self, slot):
        if slot in self._basevars:
            return self._basevars[slot]
        return super().__getitem__(slot)
    