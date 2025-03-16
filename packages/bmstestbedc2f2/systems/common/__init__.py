from controllables.core.systems import ProtoObservableProcess, BaseSystem

# TODO
from controllables.core import TemporaryUnavailableError, MutableVariable
from controllables.core.systems import ProtoProcess, BaseSystem

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


from controllables.core import BaseVariable
from controllables.core.callbacks import BaseCallback

def _make_trend_var(var: BaseVariable, on_change: BaseCallback):
    trend_var = MutableVariable(0.)
    prev_val = None
    
    @on_change.on
    def _(*args, **kwargs):
        nonlocal prev_val
        try:
            if prev_val is not None:
                trend_var.value \
                    = var.value - prev_val
            prev_val = var.value
        except TemporaryUnavailableError:
            pass
    
    return trend_var


# TODO rm BaseSystem
class _TODO_ProtoBMSystem(ProtoObservableProcess, BaseSystem):
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
            (zone_id, 'occupancy'), # TODO

            (zone_id, 'temperature'), # 
            (zone_id, 'temperature:userpref'), # 
            (zone_id, 'temperature:thermostat'), #
            # (zone_id, 'load:ahu'), # TODO rm
            (zone_id, 'trend:temperature'),
            # (zone_id, 'trend:load:ahu'), # TODO rm
        ]

    # TODO rm!!!!!
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
                        prev_values[key] = self[key].value
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
        # TODO
        return super().__getitem__(slot)
    
    def __contains__(self, slot):
        return slot in self.__variable_slots__ or slot in self._basevars
    
