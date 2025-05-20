import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def KWH2BTU(kwh):
    return kwh * 3412.142
def BTU2KWH(btu):
    return btu / 3412.142
def SQM2SQF(sqm):
    return sqm * 10.7639
def SQF2SQM(sqf):
    return sqf / 10.7639
def THM2BTU(thm):
    return thm * 99976.1
def BTU2THM(btu):
    return btu / 99976.1

def test_print():
    print(2)

# TODO add a fuzzy search utility
# this reques fuzzywuzzy
# def fuzzy_string_search(key: str, df: pd.DataFrame, col: str = 'label', num_results: int = -1, min_score: int = 75):
#     """
#     Perform a fuzzy string search on a DataFrame column.
    
#     Args:
#         key (str): The search string.
#         df (pd.DataFrame): The DataFrame to search.
#         col (str): The column to search in. Default is 'label'.
#         num_results (int): The number of results to return. Default is -1, which returns all results.
#         min_score (int): The minimum score for a match (0-100). Default is 75.
    
#     """
#     from fuzzywuzzy import fuzz

#     df_copy = df.copy()
#     df_copy['ratio'] = df_copy[col].apply(lambda x: fuzz.partial_ratio(key.lower(), str(x).lower()))
#     df_copy = df_copy[df_copy['ratio'] >= min_score]
#     if num_results < 0:
#         num_results = len(df_copy)
#     df_copy.sort_values(by=['ratio', col], ascending=False, inplace=True)
#     return df_copy.head(num_results).loc[:, df.columns]


class Codebook:
    

    def __init__(self, path_to_codebook):
        self.__codebook = pd.read_csv(path_to_codebook, skiprows=[0])




    def search_variable(self, keyword, section=None):
        if section is None:
            return self.__codebook[self.__codebook['Variable'].apply(lambda x: keyword.lower() in str(x).lower())]
        else:
            subbook = self.__codebook[self.__codebook['Section'] == section]
            return subbook[subbook['Variable'].apply(lambda x: keyword.lower() in str(x).lower())]
    def search_codebook(self, keyword, section=None, variable=False):
        if variable:
            return self.search_variable(keyword, section=section)
        if section is None:
            return self.__codebook[self.__codebook['Description and Labels'].apply(lambda x: keyword.lower() in str(x).lower())]
        else:
            subbook = self.__codebook[self.__codebook['Section'] == section]
            return subbook[subbook['Description and Labels'].apply(lambda x: keyword.lower() in str(x).lower())]
    def get_response_codes(self, variable):
        return self.__codebook[self.__codebook['Variable'] == variable]['Response Codes'].iloc[0]
    
    def parse_categorical_response_codes(self, response_string: str):
        entries = response_string.split('\n')
        numeric = []
        descriptive = []
        for e in entries:
            num = e.split(' ')[0]
            des = e.removeprefix(num + ' ')
            numeric.append(num)
            descriptive.append(des)
        return pd.DataFrame({'code': numeric, 'description': descriptive})
    def get_legend(self, variable):
        return self.parse_categorical_response_codes(self.get_response_codes(variable))