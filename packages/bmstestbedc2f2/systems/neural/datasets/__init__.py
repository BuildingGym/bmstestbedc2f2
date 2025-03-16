import itertools
from typing import Any, TypedDict

import pandas
import numpy
from numpy.lib.stride_tricks import sliding_window_view

from bmstestbedc2f2.utils import resolve_path


csv_paths = (
    resolve_path('TrainingDATA_Sep24.csv'),
    resolve_path('TrainingDATA_Oct24.csv'),
    resolve_path('TrainingDATA_Nov24.csv'),
)


class Dataset(TypedDict):
    power: pandas.DataFrame
    temp: pandas.DataFrame


def _remap_columns(
    df: pandas.DataFrame, 
    index_map: dict[str, Any], 
    names: list[str] | None = None,
):
    df_res = df[list(index_map.keys())].copy(deep=False)
    df_res.columns = df_res.columns.map(index_map)
    if names is not None:
        df_res.columns.names = names
    return df_res


def load_datasets_raw():
    return pandas.concat([
        pandas.read_csv(
            p, 
            index_col='Time', parse_dates=['Time'],
        )
        for p in csv_paths
    ])


def load_datasets(window_size: int = 30, batch_size: int = 128):
    df = load_datasets_raw()
    indices = sliding_window_view(df.index, window_shape=window_size)

    for batch_indices in numpy.array_split(indices, len(indices) // batch_size):
        df_chunk = pandas.concat(
            {
                i: df.loc[indices]
                for i, indices in enumerate(batch_indices)
            }, 
            names=['Batch', *df.index.names],
        )

        df_power = _remap_columns(
            df_chunk,
            {'AHU_Power': 'power'},
            names=['Feature'],
        )

        n_rooms = 8
        # TODO necesito???
        df_temp = _remap_columns(
            df_chunk,
            {
                rf'R{room_id + 1}_{feature_name}': (
                    room_id, 
                    {'Temp': 'temperature', 
                        'Setpiont': 'temperature_sp'}[feature_name],
                )
                for room_id, feature_name in itertools.product(
                    range(n_rooms),
                    ('Temp', 'Setpiont'),
                )
            },
            names=['Room', 'Feature'],
        )

        yield Dataset(power=df_power, temp=df_temp)


def load_datasets_v2(dropna: bool = True):
    df_raw = pandas.concat([
        pandas.read_csv(
            p, 
            index_col='Time', parse_dates=['Time'],
            na_values='-999',
        )
        for p in csv_paths
    ])

    df = pandas.DataFrame(
        index=pandas.Index([], name='Time'),
        columns=pandas.Index([], name='Feature'),
    )
    df['power'] = df_raw['AHU_Power']
    for room_id in range(8):
        df[rf'temperature.{room_id}'] = df_raw[rf'R{room_id + 1}_Temp']
        df[rf'temperature_delta.{room_id}'] = (
            df_raw[rf'R{room_id + 1}_Temp'].shift(1) 
                - df_raw[rf'R{room_id + 1}_Temp']
        )
        df[rf'temperature_error.{room_id}'] = \
            df_raw[rf'R{room_id + 1}_Temp'] - df_raw[rf'R{room_id + 1}_Setpiont']
        
    if dropna:
        df.dropna(inplace=True)

    return df


def load_datasets_batched_v2(
    window_size: int = 30, 
    batch_size: int = 128,
    dropna: bool = True,
):
    df = load_datasets_v2(dropna=dropna)
    indices = sliding_window_view(
        df.index, 
        window_shape=window_size,
    )

    for batch_indices in numpy.array_split(indices, len(indices) // batch_size):
        df_batch = pandas.concat(
            {
                i: df.loc[indices]
                for i, indices in enumerate(batch_indices)
            }, 
            names=['Batch', *df.index.names],
        )

        df_batch_out = df_batch.groupby(level='Batch').tail(n=1)
        df_batch_in = df_batch.drop(index=df_batch_out.index)

        yield df_batch_in, df_batch_out
