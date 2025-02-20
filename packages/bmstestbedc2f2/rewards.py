from collections import deque
from typing import Optional, TypedDict
import numpy as _numpy_
import pythermalcomfort as _pytc_


class ComfortFunction:
    def __init__(self, metab_rate=1.5, clothing=.5, pmv_limit=.5):
        self._metab_rate = _numpy_.asarray(metab_rate)
        self._clothing = _numpy_.asarray(clothing)
        self._pmv_limit = _numpy_.asarray(pmv_limit)

    class Inputs(TypedDict):
        temperature_drybulb: float
        temperature_radiant: float
        humidity: float
        airspeed: Optional[float]

    def __call__(self, inputs: Inputs) -> float:
        pmv = _pytc_.models.pmv_ppd(
            tdb=inputs['temperature_drybulb'], 
            tr=inputs['temperature_radiant'], 
            # calculate relative air speed
            vr=_pytc_.utilities.v_relative(v=inputs.get('airspeed', .1), met=self._metab_rate), 
            rh=inputs['humidity'], 
            met=self._metab_rate, 
            # calculate dynamic clothing
            clo=_pytc_.utilities.clo_dynamic(clo=self._clothing, met=self._metab_rate),
            limit_inputs=False ,
        )['pmv']    
        return (self._pmv_limit - _numpy_.abs(pmv)) / self._pmv_limit


class ComfortElecSavingRewardFunction:
    def __init__(self):
        self._comfort_history, self._elec_history = deque(maxlen=2), deque(maxlen=2)
        self._comfort_function = ComfortFunction()

    class Inputs(TypedDict):
        hvac_load: float
        office_occupancy: float | bool
        temperature_drybulb: float
        temperature_radiant: float
        humidity: float
        airspeed: Optional[float]
    
    def __call__(self, inputs: Inputs) -> float:
        hvac_load = inputs['hvac_load']
        office_occupancy = inputs['office_occupancy']
        comfort = self._comfort_function({
            'temperature_drybulb': inputs['temperature_drybulb'],
            'temperature_radiant': inputs['temperature_radiant'],
            'humidity': inputs['humidity'],
            'airspeed': inputs.get('airspeed', .1),
        })

        if office_occupancy != 0:
            self._comfort_history.append(_numpy_.float32(comfort))
            self._elec_history.append(_numpy_.float32(hvac_load))

            if len(self._comfort_history) < 2 or len(self._elec_history) < 2:
                return 0.

            with _numpy_.errstate(divide='ignore', invalid='ignore'):
                comfort_diff = (self._comfort_history[1] - self._comfort_history[0]) / self._comfort_history[0]
                #elec_diff = _numpy_.array((self._elec_history[1] - self._elec_history[0]) / self._elec_history[0])
                elec_diff_saving = -(self._elec_history[0] - self._elec_history[1]) / self._elec_history[1]
                reward = comfort_diff / elec_diff_saving
                if _numpy_.isnan(reward):
                    reward = 0.
                
            return reward
        
        return 0.


class ComfortElecSavingVectorRewardFunction:
    def __init__(self):
        self._comfort_history, self._elec_history = deque(maxlen=2), deque(maxlen=2)
        self._comfort_function = ComfortFunction()

    class Inputs(TypedDict):
        hvac_load: float
        office_occupancy: float | bool
        temperature_drybulb: float
        temperature_radiant: float
        humidity: float
        airspeed: Optional[float]

    def __call__(self, inputs: Inputs) -> float:     
        hvac_load = inputs['hvac_load']
        office_occupancy = inputs['office_occupancy']
        comfort = self._comfort_function({
            'temperature_drybulb': inputs['temperature_drybulb'],
            'temperature_radiant': inputs['temperature_radiant'],
            'humidity': inputs['humidity'],
            'airspeed': inputs.get('airspeed', .1),
        })
        reward = 0
        if office_occupancy != 0:  
            self._comfort_history.append(_numpy_.float32(comfort))
            self._elec_history.append(_numpy_.float32(hvac_load))

            if len(self._comfort_history) == 2:
                with _numpy_.errstate(divide='ignore', invalid='ignore'):
                    comfort_min, comfort_max = 0, 1
                    norm_comfort_0 = (self._comfort_history[0] - comfort_min) / (comfort_max - comfort_min)
                    norm_comfort_1 = (self._comfort_history[1] - comfort_min) / (comfort_max - comfort_min)

                    # TODO
                    elec_min, elec_max = 0, 1
                    norm_elec_0 = (self._elec_history[0] - elec_min) / (elec_max - elec_min)
                    norm_elec_1 = (self._elec_history[1] - elec_min) / (elec_max - elec_min)

                    # delta_comfort =(self._comfort_history[1]-self._comfort_history[0])/self._comfort_history[0]
                    # delta_elec = (self._elec_history[1]-self._elec_history[0])/self._elec_history[0]

                    delta_comfort = norm_comfort_1 - norm_comfort_0
                    delta_elec = norm_elec_1 - norm_elec_0

                    angle = _numpy_.arctan2(delta_comfort, delta_elec)
                    distance = _numpy_.sqrt(delta_elec**2 + delta_comfort**2)
                    distance_to_goal = _numpy_.sqrt(
                        (norm_elec_1 - 0) ** 2 + (norm_comfort_1 - 1) ** 2
                    )
                    max_distance = _numpy_.sqrt(2)
                    reward_proximity = 1 - distance_to_goal / max_distance
                    reward_angle = _numpy_.cos(angle - _numpy_.pi * 3 / 4)
                    reward = reward_angle * distance

                    if _numpy_.isnan(reward):
                        reward = 0.

            # print(f'pmv: {pmv}, reward: {reward}, office_occupancy: {Office_Occupancy}' )
        return reward


class HeuristicComfortFunction:
    class Inputs(TypedDict):
        delta_temperature: float
        delta_temperature_pref: float
        tolerance_temperature_pref: Optional[float]

    def __call__(self, inputs: Inputs) -> float:
        tolerance_temperature_pref = (
            inputs['tolerance_temperature_pref'] 
            if inputs.get('tolerance_temperature_pref', None) is not None else 
            2.
        )

        return (
            (1 - _numpy_.exp(-(inputs['delta_temperature_pref'] / tolerance_temperature_pref)**2)) 
                / _numpy_.exp(-(inputs['delta_temperature_pref'] / tolerance_temperature_pref)**2)
            * (1 - _numpy_.exp(inputs['delta_temperature']**2)) 
        )


class HeuristicComfortElecSavingRewardFunction:
    def __init__(self):
        self._temperature_prev = None
        self._ahu_load_prev = None
        self._comfort_function = HeuristicComfortFunction()

    class Inputs(TypedDict):
        ahu_load: float
        temperature: float
        temperature_pref: float
        office_occupancy: float
        comfort_bias: Optional[float]
        tolerance_temperature_pref: Optional[float]

    def __call__(self, inputs: Inputs) -> float:
        comfort_bias = (
            inputs.get('comfort_bias') 
            if inputs.get('comfort_bias', None) is not None else 
            .5
        )

        reward_load = -inputs['ahu_load']
        reward_comfort = self._comfort_function({
            'delta_temperature': 
                (self._temperature_prev - inputs['temperature']) 
                if self._temperature_prev is not None else 
                0,
            'delta_temperature_pref': inputs['temperature_pref'] - inputs['temperature'],
            'tolerance_temperature_pref': inputs.get('tolerance_temperature_pref', None),
        })

        self._temperature_prev = _numpy_.float32(inputs['temperature'])
        self._ahu_load_prev = _numpy_.float32(inputs['ahu_load'])
        

        reward = 0.
        if inputs['office_occupancy'] != 0:
            reward = comfort_bias * reward_comfort + (1 - comfort_bias) * reward_load
        # TODO 
        reward = _numpy_.clip(reward, -1, 1)
        return reward


__all__ = [
    'ComfortFunction',
    'ComfortElecSavingRewardFunction',
    'ComfortElecSavingVectorRewardFunction',
    'HeuristicComfortFunction',
    'HeuristicComfortElecSavingRewardFunction',
]