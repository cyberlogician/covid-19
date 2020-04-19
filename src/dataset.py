import pandas as pd
import numpy as np

# Dataset handlers
def _get_ecdc():
    # get source - note ecdc source conforms with our standard variable names
    result = pd.read_csv('https://covid.ourworldindata.org/data/ecdc/full_data.csv')

    # get population data
    pop_df = result['location'].apply(_ecdc_get_pop)


    return

def _get_phac():
    return

def _get_us():
    return

def _ecdc_get_pop(loc):
    """
    For locations in the ecdc dataset, return the population

    :param loc: str - location
    :return: (population: float or np.nan, population_density: float or np.nan)
    """

    location_map = {
        'Bolivia': 'Bolivia (Plurinational State of)',
        'Bonaire Sint Eustatius and Saba': 'Bonaire, Sint Eustatius and Saba',
        'Brunei': 'Brunei Darussalam',
        'Cape Verde': 'Cabo Verde',
        "Cote d'Ivoire": "Côte d'Ivoire",
        'Curacao': 'Curaçao',
        'Czech Republic': 'Czechia',
        'Democratic Republic of Congo': 'Democratic Republic of the Congo',
        'Faeroe Islands': 'Faroe Islands',
        'Falkland Islands': 'Falkland Islands (Malvinas)',
        'Guernsey': None,
        'Iran': 'Iran (Islamic Republic of)',
        'Jersey': None,
        'Kosovo': None,
        'Laos': "Lao People's Democratic Republic",
        'Macedonia': 'North Macedonia',
        'Moldova': 'Republic of Moldova',
        'Palestine': None,
        'Russia': 'Russian Federation',
        'South Korea': 'Republic of Korea',
        'Swaziland': None,
        'Syria': 'Syrian Arab Republic',
        'Taiwan': 'China, Taiwan Province of China',
        'Tanzania': 'United Republic of Tanzania',
        'Timor': 'Timor-Leste',
        'United States': 'United States of America',
        'Vatican': None,
        'Venezuela': 'Venezuela (Bolivarian Republic of)',
        'Vietnam': 'Viet Nam'
    }

    lookup_name = location_map.get(loc, loc)
    if lookup_name == None:
        return (np.nan, np.nan)
    else:
        return (CovidDataset.population_table.get_population(lookup_name),
                CovidDataset.population_table.get_density(lookup_name))


class CovidDataset:
    """
    Interface class so that all data looks the same
    """

    variables = [
        'date',
        'location',
        'population',
        'population_density'
        'new_cases',
        'new_deaths',
        'new_tests'
        'total_cases',
        'total_deaths',
        'total_tests',
    ]

    sources = {
        "ecdc": _get_ecdc(),
        "phac": _get_phac(),
        "us": -_get_us()
    }


    @classmethod
    def set_population_table(cls, pop_data):
        """
        Update the population_table with data from pop_data
        :param pop_data: PopulationData
        """

        cls.population_table = pop_data
        return

    def __init__(self, src):
        """
        Initialize object from a url and a variable mapping
        :param src: str - key to source dataset, each source is associated with a source data handler.
        :param population_map: dict - mapping location names in src to names in population_table
        """

        self.df = self.sources[src]
        return

