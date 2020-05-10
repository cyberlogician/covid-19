import unittest
from src.ecdc import ECDC
from src.populationdata import population_table


class TestECDC(unittest.TestCase):
    def test_ECDC(self):
        ecdc = ECDC()
        print(ecdc.df.columns)
        print(ecdc.df.dtypes)
        print(ecdc.df.date.unique())
        print(ecdc.df.loc[ecdc.df.location == 'Canada'][-7:])
        self.assertEqual(True, True)


if __name__ == '__main__':
    unittest.main()
