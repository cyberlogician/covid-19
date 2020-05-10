import unittest
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator)

from src import ECDC, PHAC



class TestCovidDataset(unittest.TestCase):
    def test_pivot_location(self):
        ecdc = ECDC()
        phac = PHAC()

        ncpm_ecdc = ecdc.pivot_location("new_cases_per_million", "Canada")
        print(ncpm_ecdc[-7:])

        ncpm_phac = phac.pivot_location("new_cases_per_million", "Canada")
        print(ncpm_phac[-7:])

        self.assertTrue(np.allclose(ncpm_ecdc.values[-7:],ncpm_phac.values[-8:-1], atol=1e-3))
        return

    def test_growth_rate(self):
        ecdc = ECDC()
        ecdc_gwth = ecdc.growth_rate('total_cases', 7, 'Canada', 'United States')
        print(ecdc_gwth[-14:])
        return

    def test_cum_pos_test_rate_and_growth(self):
        ecdc = ECDC()
        print(ecdc.pivot_location('total_tests', 'Canada', 'United States')[-14:])
        ecdc_pt = ecdc.cum_pos_test_rate('Canada', 'United States')
        print(ecdc_pt[-14:])
        print(ecdc.cum_pos_test_growth_rate(7, 'Canada', 'United States')[-14:])
        return

    def test_plot(self):
        ecdc = ECDC()
        fig = plt.figure(figsize=(16,12))
        ax1 = fig.add_axes()
        ax1 = ecdc.plot("total_cases_per_million", 'Canada', 'United States',
                         colours=['red', 'blue'],
                         date_start="2020-03-01",
                         title="Total COVID-19 Cases per Million Population"
                       )
        ax2 = fig.add_axes()
        ax2 = ecdc.plot("total_cases", 'Canada', 'United States',
                        log_scale=True,
                        colours=['red', 'blue'],
                        date_start="2020-03-01",
                        lw=6,
                        title="Total COVID-19 Cases")
        plt.show()
        return





if __name__ == '__main__':
    unittest.main()
