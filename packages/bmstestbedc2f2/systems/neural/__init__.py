# TODO
import datetime
import os
from typing import Callable, TypedDict, Iterable

import pandas

from bmstestbedc2f2.systems.common import _TODO_ProtoBMSystem
from controllables.core import MutableVariable
from controllables.core.callbacks import CallbackManager

from .utils import xarray_acc
from .model_power import PowerModel
from .model_temp import RoomTemperatureModel


# TODO
class _TODO_NeuralBMSystem(dict, _TODO_ProtoBMSystem):
    def __init__(self):
        self.events = CallbackManager()

        dict.__init__(self)
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
            self['load:ahu'].value = float(next_power_data)

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
                self[zone_id, 'temperature'].value = float(temperature)

    # TODO ...
    def step(self):
        self.events['step']()
        return self

    # TODO !!!!
    def __getitem__(self, slot):
        if slot in self._basevars:
            return self._basevars[slot]
        return super().__getitem__(slot)
    
    class LearnResult(TypedDict):
        epoch: int
        power_model: PowerModel.TrainResult
        temp_model: RoomTemperatureModel.TrainResult

    def learn(
        self, 
        df_power: pandas.DataFrame, 
        df_temp: pandas.DataFrame,
        n_epochs: int = 1,
    ):
        def _reset_time_index(x: pandas.DataFrame):
            x = x.reset_index(drop=True)
            x.index.names = ['Time']
            return x

        df_power_batch = (
            (
                df_power
                if df_power.index.names[0] == 'Batch' else
                pandas.concat({0: df_power}, names=['Batch'])
            )
            .groupby('Batch')
            .apply(_reset_time_index)
        )

        # TODO
        df_power_batch_out = df_power_batch.groupby(level='Batch').tail(n=1)
        df_power_batch_in = df_power_batch.drop(index=df_power_batch_out.index)

        df_powermodel_in = (
            df_power_batch_in
            .melt(
                ignore_index=False, 
                value_name='value',
            )
            .set_index(['Feature'], append=True)
            ['value']
        )

        # TODO
        df_powermodel_out = (
            df_power_batch_out
            .droplevel('Time', axis='index')
            .melt(ignore_index=False, value_name='value')
            .set_index(['Feature'], append=True)
            .query('Feature == "power"')
            ['value']
        )

        df_temp_batch = (
            (
                df_temp
                if df_temp.index.names[0] == 'Batch' else
                pandas.concat({0: df_temp}, names=['Batch'])
            )
            .groupby('Batch')
            .apply(_reset_time_index)
        )

        # TODO
        df_temp_batch_out = df_temp_batch.groupby(level='Batch').tail(n=1)
        df_temp_batch_in = df_temp_batch.drop(index=df_temp_batch_out.index)

        df_tempmodel_in = pandas.DataFrame(
            index=pandas.MultiIndex.from_tuples(
                [], names=['Batch', 'Time'],
            ),
            columns=pandas.MultiIndex.from_tuples(
                [], names=['Room', 'Feature']
            )
        )

        for room_id in df_temp_batch_in.columns.unique(level='Room'):
            df_tempmodel_in[room_id, 'temperature'] = df_temp_batch_in[room_id]['temperature']
            df_tempmodel_in[room_id, 'temperature_delta'] = (
                df_temp_batch_in[room_id]['temperature'].groupby('Batch').shift(1) 
                    - df_temp_batch_in[room_id]['temperature']
            )
            df_tempmodel_in[room_id, 'temperature_error'] = \
                df_temp_batch_in[room_id].eval('temperature - temperature_sp')

        df_tempmodel_in = (
            df_tempmodel_in
            .dropna()
            .melt(
                ignore_index=False, 
                value_name='value',
            )
            .set_index(
                ['Room', 'Feature'], 
                append=True,
            )
            ['value']
        )

        # TODO
        df_tempmodel_out = (
            df_temp_batch_out
            .droplevel('Time', axis='index')
            .melt(ignore_index=False, value_name='value')
            .set_index(['Room', 'Feature'], append=True)
            .query('Feature == "temperature"')
            ['value']
        )

        powermodel_in = PowerModel.Input(df_powermodel_in.to_xarray())
        powermodel_out = PowerModel.Output(df_powermodel_out.to_xarray())

        tempmodel_in = RoomTemperatureModel.Input(df_tempmodel_in.to_xarray())
        tempmodel_out = RoomTemperatureModel.Output(df_tempmodel_out.to_xarray())

        # TODO        
        for epoch in range(n_epochs):
            # TODO
            res_power_model = self._power_model.train(
                powermodel_in, powermodel_out,
            )
            res_temp_model = self._temp_model.train(
                tempmodel_in, tempmodel_out,
            )

            yield self.LearnResult(
                epoch=epoch,
                power_model=res_power_model,
                temp_model=res_temp_model,
            )
    
    def save(self, dir_path):
        os.mkdir(dir_path)
        self._power_model.save(os.path.join(dir_path, 'power_model.pt'))
        self._temp_model.save(os.path.join(dir_path, 'temp_model.pt'))

    def restore(self, dir_path):
        if os.path.exists(dir_path):
            self._power_model.restore(os.path.join(dir_path, 'power_model.pt'))
            self._temp_model.restore(os.path.join(dir_path, 'temp_model.pt'))





from bmstestbedc2f2.utils import resolve_path
from bmstestbedc2f2.systems.neural.model_v2 import Model


class _TODO_NeuralBMSystemV2(dict, _TODO_ProtoBMSystem):
    def __init__(self, model_options: dict = dict()):
        self.events = CallbackManager(slots=('begin', 'step', 'end'))

        dict.__init__(self)
        _TODO_ProtoBMSystem.__init__(self)

        # TODO
        self._model_input_buffer = None
        self._model = Model(**model_options)

        self['time'] = MutableVariable()

        self['load:ahu'] = MutableVariable(0)

        for zone_id in self.zone_ids:
            # TODO
            self[zone_id, 'occupancy'] = MutableVariable(1.)

            self[zone_id, 'temperature'] = MutableVariable(25)
            self[zone_id, 'temperature:userpref'] = MutableVariable(25)
            self[zone_id, 'temperature:thermostat'] = MutableVariable(25)

        @self.events['step'].on
        def _():
            self['time'].value = datetime.datetime.now()

            TODO = {
                'power': self['load:ahu'].value,
            }
            for zone_id in self.zone_ids:
                TODO[rf'temperature.{zone_id}'] \
                    = self[zone_id, 'temperature'].value
                TODO[rf'temperature_delta.{zone_id}'] \
                    = self[zone_id, 'trend:temperature'].value
                TODO[rf'temperature_error.{zone_id}'] \
                    = self[zone_id, 'temperature'].value \
                        - self[zone_id, 'temperature:thermostat'].value

            self._model_input_buffer = xarray_acc(
                self._model_input_buffer,
                Model.Input([[list(TODO.values())]]),
                dim='Time',
                # TODO customize max lookback
                maxlen=32,
            )

            [[next_power_data, *next_zone_temp_data]] = self._model.predict(
                self._model_input_buffer
            )
            self['load:ahu'].value = float(next_power_data)
            for zone_id, temperature in zip(self.zone_ids, next_zone_temp_data):
                self[zone_id, 'temperature'].value = float(temperature)

    def save(self, dir_path: str | None = None):
        if dir_path is None:
            dir_path = resolve_path('weights')
        if not os.path.exists(dir_path):
            os.mkdir(dir_path)
        self._model.save(os.path.join(dir_path, 'model.pt'))

    def restore(self, dir_path: str | None = None):
        if dir_path is None:
            dir_path = resolve_path('weights')
        if os.path.exists(dir_path):
            self._model.restore(os.path.join(dir_path, 'model.pt'))

    def start(self):
        self.events['begin']()
        return self

    def stop(self):
        self.events['end']()
        return self

    # TODO ...
    def step(self):
        self.events['step']()
        return self

    # TODO !!!!
    def __getitem__(self, slot):
        if slot in self._basevars:
            return self._basevars[slot]
        return super().__getitem__(slot)
    
    class LearnResult(TypedDict):
        epoch: int
        train_result: PowerModel.TrainResult

    def learn(
        self,
        datasets: Iterable[tuple[pandas.DataFrame, pandas.DataFrame]],
        num_epochs: int = 1,
        shuffle: bool = True,
        on_result: Callable[[LearnResult], None] | None = None,
    ):
        import random

        datasets = list([
            (
                Model.Input.from_dataframe(df_batch_in),
                Model.Output.from_dataframe(df_batch_out),
            )
            for df_batch_in, df_batch_out in datasets
        ])

        for epoch in range(num_epochs):
            if shuffle:
                random.shuffle(datasets)

            train_result = None

            for xarr_batch_in, xarr_batch_out in datasets:
                train_result = self._model.train(
                    xarr_batch_in, xarr_batch_out,
                )

            if on_result is not None:
                on_result(
                    self.LearnResult(
                        epoch=epoch,
                        train_result=train_result,
                    )
                )

    # def learn(
    #     self, 
    #     df_batch_in: pandas.DataFrame, 
    #     df_batch_out: pandas.DataFrame,
    #     n_epochs: int = 1,
    # ):
    #     xarr_batch_in = Model.Input.from_dataframe(
    #         df_batch_in
    #     )

    #     xarr_batch_out = Model.Output.from_dataframe(
    #         df_batch_out
    #     )

    #     for epoch in range(n_epochs):
    #         yield self.LearnResult(
    #             epoch=epoch,
    #             train_result=self._model.train(
    #                 xarr_batch_in, xarr_batch_out,
    #             ),
    #         )
        