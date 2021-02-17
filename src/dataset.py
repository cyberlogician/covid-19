import numpy as np
import matplotlib.pyplot as plt
from .utilities import rate_to_dbl, dbl_to_rate, dbl_colour



class CovidDataset:
    """
    Interface class so that all data looks the same
    """

    variables = {
        'date': 'Date data is reported by location',
        'location': 'Report location',
        'new_cases': 'Number of new cases reported on date (confirmed and probable)',
        'new_case_rate': 'New cases as proportion of population',
        'total_cases': 'Total cases reported as of date',
        'total_cases_rate': 'Total cases as proportion of populationreported as of date',
        'new_deaths': 'New deaths reported on date',
        'new_deaths_rate': 'New deaths as proportion of population reported on date',
        'total_deaths': 'Total deaths as of date',
        'total_deaths_rate': 'Total deaths as proportion of population as of date',
        'new_tests': 'New tests reported as of date',
        'new_tests_rate': 'New tests as proportion of population reported as of date',
        'total_tests': 'Total tests as of date',
        'total_tests_rate': 'Total tests as proportion of population as of date',
        'tests_units': 'One of tests or persons'
    }


    def __init__(self, src_df):
        """
        Initialize object from a source dataframe

        :param src_df: pd.DataFrame - src_df.columns must include self.variables
        """

        self.df = src_df
        return

    @property
    def current_date(self):
        return max(self.df.date)

    def get_location(self, location):
        return self.df.loc[self.df.location == location]

    def var_by_location(self, var, *locations, **kwargs):
        """
        Return a DataFrame date x location -> var

        :param var: str - the variable to be used for values
        :param *locations: list like - the locations to be used as columns
        :keyword ma_window: int - if present return moving average over window

        :return: DataFrame
        """

        if var not in self.variables:
            return NotImplemented

        select_locations = self.df['location'].apply(lambda loc: loc in locations)
        df = self.df.loc[select_locations]
        var_pivot = df.pivot(index='date', columns='location', values=var)
        var_pivot.resample('D').fillna(method='ffill')

        if "ma_window" in kwargs:
            var_pivot = var_pivot.rolling(kwargs["ma_window"]).mean()
        return var_pivot

    def active_confirmed_cases(self, *locations, **kwargs):
        """
        Return an estimate of the active confirmed cases.

        The initial model uses a simple rolling 14-day sum of new cases calculated by taking the 14 day diff of
        total cases.  Using the total case data smooths out some of the reporting variation

        :param locations: list like - locations to be included as columns
        :keyword percapita: boolean (default=False) - if true return active confirmed cases per million population

        :return: DataFrame
        """

        if kwargs.get("percapita", False):
            return self.var_by_location("total_cases_per_million", *locations).diff(periods=14)
        else:
            return self.var_by_location("total_cases", *locations).diff(periods=14)

    def pos_test_rate(self, window,  *locations):
        """
        Return the positive test rate based on the total new cases in the window over the total tests completed
        within window

        :param window: int - the time window used to compute the positive test rate
        :param locations: list like - the locations to be used as columns

        :return: DataFrame
        """

        cases = self.var_by_location("total_cases", *locations).diff(periods=window)
        tests = self.var_by_location("total_tests", *locations).diff(periods=window)

        return cases / tests

    def incidence(self, window, *locations, **kwargs):
        """
        Calculate an estimate of the true incidence (number of new infections per million population).

        incidence = Prob(infected) = Prob(Tested) * Prob(infected | Tested) / Prob(Tested | Infected)
                  = test_rate * pos_test_rate / Prob(Tested | Infected)

        Prob(Tested | Infected) is a function of the test strategy being employed within the population.
        For example, when test capacity is limited, tests may be reserved for indivviduals who are most likely
        to be infected.  This means that many infected individuals will not be selected for testing and the
        resulting Prob(Tested | Infected) is relatively low (and the pos_test_rate is high relative to the true
        incidence).  When test criteria are relaxed the Prob(Tested | Infected) and tends to 1 when everyone is
        tested

        Now,
            test_rate * pos_test_rate = (num_tests / population) * (num_new_cases / num_tests)
                                      = num_new_cases / population

        and incidence = num_new_cases / (population *  Prob(Tested | Infected))

        So to calculate an estimate of incidence we need to estimate the function Prob(Tested | Infected)).

        Assumptions:
            Prob(Tested | Infected)) is locally constant. It is determined by the test criteria and test demand.
                The second factor relates to clarity of self-screening and perception regarding the benefit of
                being tested

        :param window: int - number od periods to use to compute moving averaged of new_cases
        :param locations: list like - the locations to be used as columns
        :keyword strategy_changes: list(time_indices) - times at which the test strategy is know to have changed
        :keyword true_incidence_observations: list(time_index, incidence) - points at which true incidence is known
        :return:
        """
        return

    def growth_rate(self, var, window, *locations):
        """
        Return a DataFrame indexed by date and with columns containing average growth rate over window

        :param var: str - the variable for which average growth is calculated
        :param window: int - number of days to include in the average calculation
        :param locations: list like - locations to be included as columns

        :return: DataFrame
        """
        growth = lambda x: np.power(x, 1 / window)

        if var in self.variables:
            var_data = self.var_by_location(var, *locations)
        elif var == "active_confirmed_cases":
            var_data = self.active_confirmed_cases(*locations)
        elif var == "pos_test_rate":
            var_data = self.pos_test_rate(window, *locations)
        else:
            return NotImplemented

        delta = var_data.diff(periods=window)
        delta_ratio = (delta / var_data.shift(periods=window)).replace([np.inf, -np.inf], np.nan)
        return growth(1 + delta_ratio) - 1

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
        :keyword ma_window: int - if present plot the moving average over ma_window
        :keyword figsize: (int, int)
        :keyword log_scale: boolean (default=False)
        :keyword date_start: str in yyyy-mm-dd format
        :keyword data_end: str in yyyy-mm-dd format
        :keyword lw: int (default=3) - linewidth
        :keyword colours: dict - mapping locations to colour specifications
        :keyword title: str
        :keyword y_label: str
        :keyword percapita: boolean (default=False) Valid only if var is active_confirmed_cases

        :return: matplotlib.figure
        """

        if var == "cum_pos_test_rate":
            plot_data = self.cum_pos_test_rate(*locations)
        elif var == "cum_pos_test_growth_rate":
            plot_data = self.cum_pos_test_growth_rate(kwargs.get("ma_window", 1), *locations)
        elif var == "active_confirmed_cases":
            plot_data = self.active_confirmed_cases(*locations, **kwargs)
        elif var[-6:] == "growth":
            plot_data = self.growth_rate(var[:-7], kwargs.get("ma_window", 1), *locations)
        elif "ma_window" in kwargs:
            if var == "pos_test_rate":
                plot_data = self.pos_test_rate(kwargs["ma_window"], *locations)
            else:
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
            ylabel=kwargs.get("y_label","")
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

        x_ticks = [k for k in range(0,len(dates), 7)]
        x_tick_labels = [str(dates[x])[10:15] for x in x_ticks]
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
            bot, top = dbl_to_rate(bot_idx), dbl_to_rate(top_idx)
            ax_l.axhspan(bot, top, color=dbl_colour[top_idx], alpha=0.2)

        ax_l.set_yticks([dbl_to_rate(tick) for tick in y_ticks])
        ax_l.set_yticklabels(y_ticks)
        ax_l.set_ylabel("Number of Days to Double")

        ax_r.set_ylabel('Log Total Cases')
        ax_r.set_yscale('log')
        ax_r.legend(loc='upper right')

        ax_l.annotate("Active Cases Stop Growing", (dates[0], 0.06))

        ax_l.legend(loc='upper left')
        ax_l.set_ylim(bottom=0, top=0.5)
        fig.tight_layout()

        return fig


