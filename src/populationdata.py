import numpy as np
import pandas as pd
from pathlib import Path

class PopulationData:
    """
    Current population of countries and subnational regions.  The primary data sources are:
     - UN: https://population.un.org/wpp
     - Statistics Canada: https://www150.statcan.gc.ca/n1/pub/91-002-x/2019004/quarterly_trimestrielles_202001_v1.xlsx
     - US Census Bureau: https://www2.census.gov/programs-surveys/popest/datasets/2010-2019/national/totals/nst-est2019-alldata.csv?#
    """
    def __init__(self, **args):
        if 'df' in args:
            self.df = args['df']
        else:
            self.df = pd.DataFrame(columns=['year', 'location', 'population', 'population_density'])
            self.update(self._get_world())
            self.update(self._get_canada())
            self.update(self._get_us())
        return

    @classmethod
    def from_csv(cls, filename):
        result = PopulationData()
        result.df = pd.read_csv(filename)
        return result

    def update(self, pop_data):
        self.df = self.df.merge(pop_data, how='outer')
        return

    def _get_world(self):
        url = "https://population.un.org/wpp/Download/Files/1_Indicators%20(Standard)/CSV_FILES/WPP2019_TotalPopulationBySex.csv"
        col_map = {
            "Location": "location",
            "PopTotal": "population",
            "PopDensity": "population_density",
            "Time": "year"
        }
        df = pd.read_csv(url)
        df.rename(columns=col_map, inplace=True)
        df['population'] = 1000 * df['population']
        df = df.loc[df.Variant == 'Medium'][list(col_map.values())]
        return df.loc[df.year == 2020]

    def _get_canada(self):
        url = "https://www150.statcan.gc.ca/n1/pub/91-002-x/2019004/quarterly_trimestrielles_202001_v1.xlsx"
        pr_mapper = {'Canada': 'Canada',
                     'N.L.': 'Newfoundland and Labrador',
                     'P.E.I.': 'Prince Edward Island',
                     'N.S.': 'Nova Scotia',
                     'N.B.': 'New Brunswick',
                     'Que.': 'Quebec',
                     'Ont.': 'Ontario',
                     'Man.': 'Manitoba',
                     'Sask.': 'Saskatchewan',
                     'Alta.': 'Alberta',
                     'B.C.': 'British Columbia',
                     'Y.T.': 'Yukon',
                     'N.W.T.': 'Northwest Territories',
                     'Nvt.': 'Nunavut'}
        areas = {
            'Canada': 9093507,
            'Newfoundland and Labrador': 373872,
            'Prince Edward Island': 5660,
            'Nova Scotia': 53338,
            'New Brunswick': 71450,
            'Quebec': 1365128,
            'Ontario': 917741,
            'Manitoba': 553556,
            'Saskatchewan': 591670,
            'Alberta': 642317,
            'British Columbia': 925186,
            'Yukon': 474391,
            'Northwest Territories': 1183085,
            'Nunavut': 1936113
        }
        df = pd.read_excel(
            url,
            sheet_name="Population",
            header=[3],
            skiprows=[4]
        )
        df.rename(columns=pr_mapper, inplace=True)
        df = df.set_index(["Year", "Month"]).drop('Level', axis='columns')
        df = df.stack(dropna=False)
        df = df.reset_index()
        df.rename(columns={
                            'Year': 'year',
                            'Month': 'month',
                            'level_2': 'location',
                             0: 'population'},
                  inplace=True
        )
        df['population'] = df['population'].apply(float)
        df["population_density"] = df["population"] / (df["location"].apply(lambda loc: areas[loc]))
        return df.loc[(df.year == 2020) & (df.month == 1)].drop('month', axis='columns')

    def _get_us(self):
        url = "https://www2.census.gov/programs-surveys/popest/datasets/2010-2019/national/totals/nst-est2019-alldata.csv?#"
        df = pd.read_csv(url)
        df = df[["NAME", "POPESTIMATE2019"]]
        df.rename(
            columns={'NAME': 'location', 'POPESTIMATE2019': 'population'}, inplace=True
        )
        df['year'] = 2019
        df['population'] = df['population'].apply(float)
        df['population_density'] = np.nan
        return df


    def get_population(self, loc):
        return self.df.loc[self.df['location'] == loc]['population'].values[0]

    def get_density(self, loc):
        return self.df.loc[self.df['location'] == loc]['population_density'].values[0]

    def to_csv(self, filename):
        """
        Save object to csv

        :param filename: str - path to file

        """
        save_path = Path(filename)
        self.df.to_csv(save_path, index=False)
        return


