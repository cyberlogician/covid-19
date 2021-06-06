import pandas as pd
import numpy as np
import datetime
import time
from src import CovidDataset
from src import population_table

class PHAC(CovidDataset):

    variables = {
        'date': 'Date data is reported by location',
        'location': 'Report location',
        'new_cases': 'Number of new cases reported on date (confirmed and probable)',
        'new_cases_rate': 'New cases as proportion of population',
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
        'tests_units': 'One of tests or persons',
        # New variables
        'total_confirmed_cases': 'total confirmed cases',
        'total_confirmed_cases_rate': '',
        'total_probable_cases': '',
        'total_probable_cases_rate': '',
        'total_recovered': '',
        'total_recovered_rate': '',
        'new_recovered': '',
        'new_recovered_rate': '',
        'active_cases': '',
        'active_cases_rate': ''
    }

    def __init__(self):
        # set up constants
        self.provinces = ['Canada',
                          'Newfoundland and Labrador', 'Prince Edward Island', 'Nova Scotia', 'New Brunswick',
                          'Quebec',
                          'Ontario',
                          'Manitoba', 'Saskatchewan', 'Alberta', 'British Columbia',
                          'Yukon', 'Northwest Territories', 'Nunavut']

        self.large_provinces = ['Canada', 'British Columbia', 'Ontario', 'Quebec']
        self.maritimes = ['Newfoundland and Labrador', 'Prince Edward Island', 'Nova Scotia', 'New Brunswick']
        self.prairies = ['Manitoba', 'Saskatchewan', 'Alberta']

        self.prov_colours = dict(zip(self.provinces,
                                ['r',
                                 'darkred', 'salmon', 'darksalmon', 'sienna',
                                 'cornflowerblue',
                                 'b',
                                 'gold', 'goldenrod', 'darkgoldenrod', 'g',
                                 'olive', 'chartreuse', 'darkseagreen']
                                )
                            )

        # download and shape data
        col_map = {'date': 'date',
                   'prname': 'location',
                   'numconf': 'total_confirmed_cases',
                   'numprob': 'total_probable_cases',
                   'numtotal': 'total_cases',  # total_confirmed_cases + total_probable
                   'ratetotal': 'total_cases_rate',
                   'numtoday': 'new_cases',
                   'numrecover': 'total_recovered',
                   'numrecoveredtoday': 'new_recovered',
                   'numactive': 'active_cases',
                   'rateactive': 'active_cases_rate',
                   'numdeaths': 'total_deaths',
                   'ratedeaths': 'total_deaths_rate',
                   'numdeathstoday': 'new_deaths',
                   'numtests': 'total_tests',
                   'numteststoday': 'new_tests'
                   }
        dateparse = lambda x: datetime.date(*time.strptime(x, '%d-%m-%Y')[:3])
        self.src_url = "https://health-infobase.canada.ca/src/data/covidLive/covid19.csv"
        src = pd.read_csv(self.src_url,
                                 error_bad_lines=False,
                                 warn_bad_lines=False,
                                 parse_dates=['date'],
                                 date_parser= dateparse)

        # Starting 2021-02-01 reporting changed from numtested to num tests
        src['numtests'] = src.apply(lambda row: row['numtested']
                                        if not np.isnan(row['numtested'])
                                        else row['numtests'], axis=1)
        src['numteststoday'] = src.apply(lambda row: row['numtestedtoday']
                                            if not np.isnan(row['numtestedtoday'])
                                            else row['numteststoday'], axis=1)

        src = src[list(col_map)]
        src.rename(columns=col_map, inplace=True)
        src = src.loc[src.location != 'Repatriated travellers']

        # Set test_units
        # print(src.columns)
        src['test_units'] = src.apply(lambda row: 'persons'
                                                   if row['date'] < datetime.date(2021,2,1)
                                                   else 'tests', axis=1)

        # Compute proportions
        # src['population'] = src['location'].apply(population_table.get_population)
        src['population'] = 100000 * src['total_cases'] / src['total_cases_rate']
        src['total_cases_rate'] = src['total_cases_rate'] / 100000.0
        src['total_deaths_rate'] = src['total_deaths_rate'] / 100000.0
        src['active_cases_rate'] = src['active_cases_rate'] / 1000000.0

        for var in ['new_cases', 'new_cases', 'new_tests', 'total_tests','new_deaths', 'total_confirmed_cases',
                    'total_probable_cases', 'total_recovered', 'new_recovered', 'active_cases']:
            src[var+"_rate"] = src[var] / src['population']

        super().__init__(src)
        return

