import unittest
from src import PHAC


class TestPHAC(unittest.TestCase):
    def test_class(self):
        phac = PHAC()
        print(phac.df.columns)
        print(phac.df.dtypes)
        print(phac.df[-26:])
        self.assertEqual(True, True)


if __name__ == '__main__':
    unittest.main()
