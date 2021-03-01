import pandas as pd
import numpy as np
import datetime
import time
from src import CovidDataset
from src import population_table

class Ontario(CovidDataset):

    def __init__(self):
        # Set up constants

        # download and shape data
        col_map = {'Accurate_Episode_Date': 'date',
                   'Reporting_PHU_City': 'location',
                   'numtotal': 'total_cases',
                   'numdeaths': 'total_deaths',
                   'numtested': 'total_tests'
                   }
        dateparse = lambda x: datetime.date(*time.strptime(x, '%d-%m-%Y')[:3])
        src = pd.read_csv("https://health-infobase.canada.ca/src/data/covidLive/covid19.csv",
                          error_bad_lines=False,
                          warn_bad_lines=False,
                          parse_dates=['date'],
                          date_parser= dateparse)
        src = src[list(col_map)]
        src.rename(columns=col_map, inplace=True)
        src = src.loc[src.location != 'Repatriated travellers']

        # Compute proportions
        src['population'] = src['location'].apply(population_table.get_population)
        src['total_cases_per_million'] = 1000000 * src['total_cases'] / src['population']
        src['total_deaths_per_million'] = 1000000 * src['total_deaths'] / src['population']
        src['total_tests_per_thousand'] = 1000 * src['total_tests'] / src['population']

        super().__init__(src)
        return