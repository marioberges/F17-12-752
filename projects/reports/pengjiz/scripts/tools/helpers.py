"""Helper functions."""

import pandas as pd  # pandas does not provide typing files
import numpy as np
import scipy as sp
from typing import Callable, Iterable, Any
from itertools import zip_longest


__all__ = ['id',
           'moving_window',
           'moving_mode',
           'segmented_map',
           'load_data',
           'conv2dt',
           'post_temperature',
           'post_occupancy',
           'post_appliance']


def id(x: Any) -> Any:
    return x


def dict_zip(dict1, dict2):
    return {k: (dict1[k], dict2[k])
            for k
            in set(dict1.keys()).intersection(set(dict2.keys()))}


def segmented_map(vec, groups: Iterable,
                  fn: [[Any], Any]) -> Any:
    return {group: fn(vec[groups == group]) for group in np.unique(groups)}


def moving_window(vec, width: int):
    return np.array([*zip_longest(*[vec[x:] for x in range(width)],
                                  fillvalue=np.nan)])


def moving_mode(vec, width: int):
    return sp.stats.mode(moving_window(vec, width),
                         axis=1, nan_policy='omit')[0].reshape(-1, )


def load_data(path: str,
              post: Callable[[Any], Any] =id,
              *args, **kwargs) -> Any:
    return (pd.read_csv(path, *args, **kwargs)
            .pipe(post))


def conv2dt(df, dt_col: str ='Time'):
    return df.assign(Time=lambda df: pd.to_datetime(df[dt_col]))


def post_temperature(df_temperature):
    return (conv2dt(df_temperature)
            .groupby([pd.Grouper(key='Time', freq='1min'), 'Location'])
            .mean()
            .unstack(1)
            .dropna())


def encode_loc(locs):
    return (locs.str.extractall(r'\'([a-zA-Z0-9]*)\'')
            .pipe(lambda df: pd.get_dummies(df,
                                            prefix='',
                                            prefix_sep='',
                                            columns=[0]))
            .reset_index('match', drop=True)
            .groupby(level=0)
            .sum())


def post_occupancy(df_occupancy):
    tmp = (conv2dt(df_occupancy)
           .set_index('Time')
           .loc[:, 'Location'])
    locs = encode_loc(tmp)
    time_idx = pd.date_range(tmp.index.min(), tmp.index.max(), freq='1s')

    results = (pd.concat([tmp, locs], axis=1)
               .drop('Location', axis=1)
               .reindex(time_idx)
               .fillna(method='ffill')
               .groupby(pd.Grouper(level=0, freq='1min'))
               .mean())

    results.columns = pd.MultiIndex.from_product([['Location'],
                                                  results.columns])
    return results


def post_appliance(df_appliance):
    return (conv2dt(df_appliance)
            .groupby(pd.Grouper(key='Time', freq='1min'))
            .mean())
