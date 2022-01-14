"""
Given a time series that is the result of a mixture of underlying geometric processes, the components
in this module find the model of the underlying processes that best explain the observed behaviour
"""
import numpy as np
import pandas as pd

class GeometricProcess:
    """
    A GeometricProcess object provides a model of the process based on an input dataset
    """
    def __init__(self, **kwargs):
        """
        Instantiate a GeometricProcess object

        :keyword est_per: int (default=14) - number of time periods to be used to estimate the value
        """

        self.est_per = kwargs.get("est_per", 14)
        return

    def fit(self, series):
        """
        Fit a model P(t), R(t) that at each point t minimizes the mean squared error
            sum( (series[t-k] - exp(P(t) + R(t)*k)**2 for k in range(est_per) )

        :param series: pandas.Series with DateTimeIndex
        """
        def _get_err(rec):
            if len(rec[1]) > 0:
                return rec[1][0]
            else:
                return 0

        logser = np.log(series)

        x = []
        fits = []
        for idx in series.index[self.est_per:]:
            s = idx - self.est_per * pd.Timedelta("1 day")
            vals = logser[s:idx].values
            lvals = np.where(np.logical_not(np.isnan(vals)))[0]
            # print(vals, lvals)
            if len(lvals) > 1:
                A = np.ones((len(lvals), 2))
                A[:,0] = lvals
                fits.append(np.linalg.lstsq(A, vals[lvals], rcond=None))
                x.append(idx)

        self.model = pd.DataFrame(
            {"LogGrowthRate": [fit[0][0] for fit in fits],
             "LogPriorProcessLoad": [fit[0][1] for fit in fits],
             "Error": [_get_err(fit) for fit in fits]},
            index=x
        )

        pred_logval = []
        for row in self.model.itertuples():
            pred_logval.append(row.LogPriorProcessLoad + (self.est_per - 1) * row.LogGrowthRate)

        self.model["PredictedLogValue"] = pred_logval
        self.model["ActualLogValue"] = logser[self.model.index]

        return




