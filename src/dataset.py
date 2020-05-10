from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator)

class CovidDataset:
    """
    Interface class so that all data looks the same
    """

    variables = [
        'date',
        'location',
        'new_cases',
        'new_cases_per_million',
        'total_cases',
        'total_cases_per_million',
        'new_deaths',
        'new_deaths_per_million',
        'total_deaths',
        'total_deaths_per_million',
        'new_tests',
        'new_tests',
        'new_tests_per_thousand',
        'total_tests',
        'total_tests_per_thousand',
        'tests_units'
    ]

    def __init__(self, src_df):
        """
        Initialize object from a source dataframe
        :param src_df: pd.DataFrame - src_df.columns must include:
            date,
            location,
            total_cases, total_cases_per_million,
            total_deaths, total_deaths_per_million
            total_tests, total_tests_per_thousand
            test_units.
        """

        self.df = src_df
        cols = self.df.columns
        for var in ["cases", "deaths", "tests"]:
            if f"new_{var}" not in cols:
                self._new_from_total(var)
            if var == "tests":
                if f"new_{var}_per_thousand" not in cols:
                    self._new_from_total(var, "thousand")
            else:
                if f"new_{var}_per_million" not in cols:
                    self._new_from_total(var, "million")
        return

    @property
    def current_date(self):
        return max(self.df.date)

    def _new_from_total(self, var, unit=""):
        """
        compute daily measurements from total

        :param var: str - one of cases, deaths, tests
        :param unit: str - one of empty string, thousand, million
        :return: None
        """

        # Set tot_var (source) and new_var (target)
        if unit == "":
            tot_var = f"total_{var}"
            new_var = f"new_{var}"
        else:
            tot_var = f"total_{var}_per_{unit}"
            new_var = f"new_{var}_per_{unit}"

        tot = self.df.pivot(index='date', columns='location', values=tot_var).fillna(0)
        tot.resample('D').fillna(method='ffill')
        nc = tot.diff()
        nc = nc.stack()
        nc.rename(new_var, inplace=True)
        self.df = self.df.join(nc, on=['date', 'location'], lsuffix='_old')
        if f'{new_var}_old' in self.df.columns:
            self.df.drop(f'{new_var}_old', axis='columns')
        return

    def pivot_location(self, var, *locations):
        """
        Return a DataFrame date x location -> var

        :param var: str - the variable to be used for values
        :param locations: list like - the locations to be used as columns

        :return: DataFrame
        """

        select_locations = self.df['location'].apply(lambda loc: loc in locations)
        df = self.df.loc[select_locations]
        var_pivot = df.pivot(index='date', columns='location', values=var)
        var_pivot.resample('D').fillna(method='ffill')
        return var_pivot

    def growth_rate(self, var, window, *locations):
        """
        Return a DataFrame indexed by date and with columns containing average growth rate over window

        :param var: str - the variable for which average growth is calculated
        :param window: int - number of days to include in the average calculation
        :param locations: list like - locations to be included as columns

        :return: DataFrame
        """
        growth = lambda x: np.power(x, 1 / window)

        # select_locations = self.df['location'].apply(lambda loc: loc in locations)
        # df = self.df.loc[select_locations]
        # var_pivot = df.pivot(index='date', columns='location', values=var).fillna(0)
        # var_pivot.resample('D').fillna(method='ffill')
        var_pivot = self.pivot_location(var, *locations)
        delta = var_pivot.diff(periods=window)
        return growth(1 + (delta / var_pivot)) - 1

    def cum_pos_test_rate(self, *locations):
        """
        Return a DataFrame date x locations -> positive test rate

        :param locations: list like - locations to be included as columns

        :return: DataFrame
        """

        return (self.pivot_location('total_cases', *locations)
                  / self.pivot_location('total_tests', *locations).interpolate(method='linear'))

    def cum_pos_test_growth_rate(self, window, *locations):
        """
        Return a DataFrame date x locations -> positive test rate  average growth over window

        :param locations: list like - locations to be included as columns

        :return: DataFrame
        """

        growth = lambda x: np.power(x, 1 / window)

        pt = self.cum_pos_test_rate(*locations)
        delta = pt.diff(periods=window)
        return growth(1 + (delta / pt)) - 1

    def plot(self, var, *locations, **kwargs):
        """
        plot variable for locations on single axis

        :param var: str - variable to be plotted
        :param locations: str - list like of str - the locations to be plotted

        Optional key-word arguments
        :param log_scale: boolean (default=False)
        :param date_start: str in yyyy=mm-dd format
        :param data_end: str in yyyy=mm-dd format
        :param lw: int (default=3) - linewidth
        :param colours: list - in order of locations
        :param title: str
        :param y_label: str

        :return: matplotlib.axes
        """

        var_piv = self.pivot_location(var, *locations)

        start_date = kwargs.get("date_start", var_piv.index[0])
        end_date = kwargs.get("date_end", var_piv.index[-1])

        plot_properties = dict(
            xlim=(kwargs.get("date_start", var_piv.index[0]), kwargs.get("date_end", var_piv.index[-1])),
            logy=kwargs.get("log_scale", False),
            lw=kwargs.get("lw", 3),
            title=kwargs.get("title", ""),
        )
        if "colours" in kwargs:
            plot_properties["color"] = kwargs["colours"]
        # return var_piv[start_date: end_date].plot(**plot_properties)
        return var_piv.plot(**plot_properties)


