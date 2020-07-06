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

    dbl_periods = {days: np.expm1(np.log(2) / days) for days in [2,3,4,5,6,7,14,28]}
    dbl_colour = {1: (1.0, 0, 0), 2: (1.0, 0.2, 0), 3: (1.0, 0.4, 0), 4: (1.0, 0.6, 0), 5: (1.0, 0.8, 0),
                   6: (1.0, 1.0, 0), 7: (0.8, 1.0, 0), 14: (0, 1.0, 0), 28: (0, 1.0, 0.5)}

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

    def get_location(self, location):
        return self.df.loc[self.df.location == location]

    def var_by_location(self, var, *locations, **kwargs):
        """
        Return a DataFrame date x location -> var

        :param var: str - the variable to be used for values
        :param locations: list like - the locations to be used as columns
        :param ma_window: int - if present return moving average over window

        :return: DataFrame
        """

        select_locations = self.df['location'].apply(lambda loc: loc in locations)
        df = self.df.loc[select_locations]
        var_pivot = df.pivot(index='date', columns='location', values=var)
        var_pivot.resample('D').fillna(method='ffill')

        if "ma_window" in kwargs:
            var_pivot = var_pivot.rolling(kwargs["ma_window"]).mean()
        return var_pivot

    def active_confirmed_cases(self, *locations):
        """
        Return an estimate of the active confirmed cases.

        The initial model uses a simple rolling 14-day sum of new cases calculated by taking the 14 day diff of
        total cases.  Using the total case data smooths out some of the reporting variation

        :param locations: list like - locations to be included as columns

        :return: DataFrame
        """

        return self.var_by_location("total_cases", *locations).diff(periods=14)

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
        var_pivot = self.var_by_location(var, *locations)
        delta = var_pivot.diff(periods=window)
        return growth(1 + (delta / var_pivot.shift(periods=window))) - 1

    def active_growth_rate(self, window, *locations):
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
        var_pivot = self.active_confirmed_cases(*locations)
        delta = var_pivot.diff(periods=window)
        return growth(1 + (delta / var_pivot.shift(periods=window))) - 1

    def cum_pos_test_rate(self, *locations):
        """
        Return a DataFrame date x locations -> positive test rate

        :param locations: list like - locations to be included as columns

        :return: DataFrame
        """

        return (self.var_by_location('total_cases', *locations)
                  / self.var_by_location('total_tests', *locations).interpolate(method='linear'))

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

    def plot_var(self, var, *locations, **kwargs):
        """
        plot variable for locations on single axis

        :param var: str - variable to be plotted
        :param locations: str - list like of str - the locations to be plotted

        Optional key-word arguments
        :param ma_window: int - if present plot the moving average over ma_window
        :param figsize: (int, int)
        :param log_scale: boolean (default=False)
        :param date_start: str in yyyy=mm-dd format
        :param data_end: str in yyyy=mm-dd format
        :param lw: int (default=3) - linewidth
        :param colours: dict - mapping locations to colour specifications
        :param title: str
        :param y_label: str

        :return: matplotlib.axes
        """

        if var == "cum_pos_test_rate":
            plot_data = self.cum_pos_test_rate(*locations)
        elif var == "cum_pos_test_growth_rate":
            plot_data = self.cum_pos_test_growth_rate(kwargs.get("ma_window", 1), *locations)
        elif var[-6:] == "growth":
            plot_data = self.growth_rate(var[:-7], kwargs.get("ma_window", 1), *locations)
        elif "ma_window" in kwargs:
            plot_data = self.var_by_location(var, *locations, ma_window=kwargs["ma_window"])
        else:
            plot_data = self.var_by_location(var, *locations)

        start_date = kwargs.get("date_start", plot_data.index[0])
        end_date = kwargs.get("date_end", plot_data.index[-1])

        plot_properties = dict(
            figsize=kwargs.get("figsize", (16,12)),
            xlim=(start_date, end_date),
            logy=kwargs.get("log_scale", False),
            lw=kwargs.get("lw", 3),
            title=kwargs.get("title", ""),
        )
        if "colours" in kwargs:
            plot_properties["color"] = [kwargs["colours"][loc] for loc in sorted(locations)]
        fig = plot_data.plot(**plot_properties)

        legend_labels = [f"{loc}: {var}={plot_data.to_dict()[loc][max(plot_data[:end_date].index)]:.4f}"
                         for loc in sorted(locations)]

        plt.legend(legend_labels)
        return fig

    def plot_location(self, location, **kwargs):
        """
        Plot relevant data for a single location.

        The plot foreground (right axis contains the 28, 7, and 3 day  \rolling average growth rates.  It also
        contains the cumulative deaths (log scale).  the background (left axis is the log total cases.

        :param location: string.

        Optional keyword arguments
        :param figsize: (float, float)
        :param from_date: str = yyyy-mm-dd

        :return: matplotlib.plot.figure
        """

        fig = plt.figure(figsize=kwargs.get("figsize", (16,12)))
        ax_l = fig.add_subplot(111)
        # ax_l.xaxis.set_major_locator(MultipleLocator(7))

        start = kwargs.get("from_date", 0)

        gwth_3 = self.growth_rate("total_cases", 3, location)[start:]
        gwth_7 = self.growth_rate("total_cases", 7, location)[start:]
        gwth_28 = self.growth_rate("total_cases", 28, location)[start:]

        deaths = self.var_by_location("total_deaths", location)[start:]
        cases = self.var_by_location("total_cases", location)[start:]

        dates = deaths.index
        # print([list(map(str, dates))])

        x_ticks = [dates[k] for k in range(0,len(dates), 7)]
        x_tick_labels = [str(x)[10:15] for x in x_ticks]
        ax_l.set_xticks(x_ticks)
        ax_l.set_xticklabels(x_tick_labels)
        ax_l.set_xlabel("2020")
        # ax_l.xaxis.set_major_locator(MultipleLocator(14))


        ax_l.plot(dates, gwth_3.values, label="3 day rolling average growth rate of total cases", c='g', lw=6)
        ax_l.plot(dates, gwth_7.values, label="7 day rolling average growth rate of total cases", c='b', lw=6)
        ax_l.plot(dates, gwth_28.values, label="28 day rolling average growth rate of total cases", c='r', lw=6)

        ax_r = ax_l.twinx()
        ax_r.bar(dates, cases[location].values, color=(0,0,1, 0.3))
        ax_r.plot(dates, deaths[location].values, label="Total Deaths", c='black',lw=6)

        # for dbl_days, dbl_rate in self.dbl_periods.items():
        #     if dbl_days == 28:
        #         ax_l.axhspan(0, dbl_rate, color=self.dbl_colour[dbl_days], alpha=0.2)
        #     elif dbl_days == 14:
        #         ax_l.axhspan(self.dbl_periods[28], dbl_rate, color=self.dbl_colour[dbl_days], alpha=0.2)
        #     else:
        #         ax_l.axhline(y=dbl_rate, c=self.dbl_colour[dbl_days], lw=3)
        y_ticks = ["inf",28,14,7,6,5,4,3,2,1]
        for bot_idx, top_idx in zip(y_ticks[:-1], y_ticks[1:]):
            bot = self.dbl_periods.get(bot_idx, 0)
            top = self.dbl_periods.get(top_idx, 0.5)
            ax_l.axhspan(bot, top, color=self.dbl_colour[top_idx], alpha=0.2)

        ax_l.set_yticks(list(self.dbl_periods.values()))
        ax_l.set_yticklabels(list(self.dbl_periods.keys()))
        ax_l.set_ylabel("Number of Days to Double")

        ax_r.set_ylabel('Log Total Cases')
        ax_r.set_yscale('log')
        ax_r.legend(loc='upper right')

        ax_l.annotate("Active Cases Stop Growing", (dates[0], 0.06))

        ax_l.legend(loc='upper left')
        ax_l.set_ylim(bottom=0, top=0.5)
        fig.tight_layout()

        return fig


