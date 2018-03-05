from abc import ABC, abstractmethod
from torch.autograd import Variable

import pandas as pd
import re
import torch


class Journal(ABC):
    @abstractmethod
    def record_step(self, data):
        pass


class DataframeJournal(Journal):
    def __init__(self, df=None, metadata=None,
            blacklist=['._[xy]_in', '._[xy]_out']):
        if metadata is None:
            metadata = {}
            metadata['created_at'] = pd.Timestamp.now()
        self.metadata = metadata

        self._buffered_data = {}
        self._df = df
        self._blacklist = blacklist
        self._blacklist_regexps = [re.compile(x) for x in blacklist]

    def is_blacklisted(self, key):
        for regex in self._blacklist_regexps:
            if regex.search(key):
                return True
        return False

    def record_step(self, data):
        data['timestamp'] = pd.Timestamp.now()
        for k, v in data.items():
            if self.is_blacklisted(k):
                continue

            if isinstance(v, Variable):
                v = v.data[0]

            if k in self._buffered_data:
                self._buffered_data[k].append(v)
            else:
                self._buffered_data[k] = [v]

    @property
    def df(self):
        if self._buffered_data:
            buffered_df = pd.DataFrame.from_dict(self._buffered_data)
            self._buffered_data = {}

            if self._df is None:
                self._df = buffered_df
            else:
                self._df = pd.concat([self._df, buffered_df], ignore_index=True)

        return self._df

    @classmethod
    def from_file(cls, filename, key):
        with pd.HDFStore(filename) as store:
            df = store[key]
            metadata = store.get_storer(key).attrs.metadata
        obj = cls(df=df, metadata=metadata)
        return obj

    def save(self, filename, key):
        with pd.HDFStore(filename) as store:
            store.put(key, self.df)
            store.get_storer(key).attrs.metadata = metadata
