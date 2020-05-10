import pandas as pd
import numpy as np
import datetime
import time
from src import CovidDataset

# from src.populationdata import population_table

class ECDC(CovidDataset):
    """
    Class to manage connection to and processing covid-19 data from the European Centre for Disease Control
    """

    # Location_map maps Country names in the ECDC dataset to the corresponding country name in the Population
    # Dataset
    # location_map = {
    #     'Bolivia': 'Bolivia (Plurinational State of)',
    #     'Bonaire Sint Eustatius and Saba': 'Bonaire, Sint Eustatius and Saba',
    #     'Brunei': 'Brunei Darussalam',
    #     'Cape Verde': 'Cabo Verde',
    #     "Cote d'Ivoire": "Côte d'Ivoire",
    #     'Curacao': 'Curaçao',
    #     'Czech Republic': 'Czechia',
    #     'Democratic Republic of Congo': 'Democratic Republic of the Congo',
    #     'Faeroe Islands': 'Faroe Islands',
    #     'Falkland Islands': 'Falkland Islands (Malvinas)',
    #     'Guernsey': None,
    #     'Iran': 'Iran (Islamic Republic of)',
    #     'Jersey': None,
    #     'Kosovo': None,
    #     'Laos': "Lao People's Democratic Republic",
    #     'Macedonia': 'North Macedonia',
    #     'Moldova': 'Republic of Moldova',
    #     'Palestine': None,
    #     'Russia': 'Russian Federation',
    #     'South Korea': 'Republic of Korea',
    #     'Swaziland': None,
    #     'Syria': 'Syrian Arab Republic',
    #     'Taiwan': 'China, Taiwan Province of China',
    #     'Tanzania': 'United Republic of Tanzania',
    #     'Timor': 'Timor-Leste',
    #     'United States': 'United States of America',
    #     'Vatican': None,
    #     'Venezuela': 'Venezuela (Bolivarian Republic of)',
    #     'Vietnam': 'Viet Nam'
    # }

    # @classmethod
    # def _get_pop(cls, loc):
    #     """
    #     For locations in the ecdc dataset, return the population
    #
    #     :param loc: str - location
    #     :return: (population: float or np.nan, population_density: float or np.nan)
    #     """
    #
    #     lookup_name = cls.location_map.get(loc, loc)
    #     if lookup_name == None:
    #         return np.nan
    #     else:
    #         return population_table.get_population(lookup_name)

    # @classmethod
    # def _get_density(cls, loc):
    #     """
    #     For locations in the ecdc dataset, return the population
    #
    #     :param loc: str - location
    #     :return: (population: float or np.nan, population_density: float or np.nan)
    #     """
    #
    #     lookup_name = cls.location_map.get(loc, loc)
    #     if lookup_name == None:
    #         return np.nan
    #     else:
    #         return population_table.get_density(lookup_name)

    def __init__(self):
        dateparse = lambda x: datetime.date(*time.strptime(x, '%Y-%m-%d')[:3])
        src_df = pd.read_csv('https://covid.ourworldindata.org/data/owid-covid-data.csv',
                              error_bad_lines=False,
                              warn_bad_lines=False,
                              parse_dates=['date'],
                              date_parser= dateparse)
        super().__init__(src_df)
        return
