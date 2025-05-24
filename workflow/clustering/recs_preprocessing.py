import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utilities import parse_climate_code
from utilities import KWH2BTU, BTU2KWH, SQF2SQM, SQM2SQF, THM2BTU, BTU2THM
import warnings


warnings.filterwarnings('ignore')

path_to_recs_microdata = '../resources/RECS/recs2020_public_v6.csv'
path_to_recs_codebook = '../resources/RECS/codebook_updated.csv'#'../Data/US/RECS/RECS 2020 Codebook for Public File - v6.csv'

class Codebook:
    def __init__(self, path_to_codebook=path_to_recs_codebook):
        self.codebook = pd.read_csv(path_to_codebook, skiprows=[0])

    def sections(self):
        return self.codebook['Section'].unique()
    

    def search_by_variable(self, keyword, section=None):
        if section is None:
            return self.codebook[self.codebook['Variable'].apply(lambda x: keyword.lower() in str(x).lower())]
        else:
            subbook = self.codebook[self.codebook['Section'] == section]
            return subbook[subbook['Variable'].apply(lambda x: keyword.lower() in str(x).lower())]
    
    def search(self, keyword, section=None, variable=False):
        if variable:
            return self.search_by_variable(keyword, section=section)
        if section is None:
            return self.codebook[self.codebook['Description and Labels'].apply(lambda x: keyword.lower() in str(x).lower())]
        else:
            subbook = self.codebook[self.codebook['Section'] == section]
            return subbook[subbook['Description and Labels'].apply(lambda x: keyword.lower() in str(x).lower())]
    
    def get_response_codes(self, variable):
        return self.codebook[self.codebook['Variable'] == variable]['Response Codes'].iloc[0]
    
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
    

def parse_column(column, mapping, bypass_unmappable=True):
    def map_col(x):
        if x in mapping.keys():
            return mapping[x]
        else:
            if bypass_unmappable:
                return x
            else:
                return 0
    return column.apply(map_col)

def make_groups(w, mapping, default=0):
    if w in mapping.keys():
        return mapping[w]
    else:
        return default
    

def preprocessing(recs: pd.DataFrame, cb: Codebook):
    recs = recs.copy()
    weight_cols = cb.search('Final Analysis Weight')['Variable'].to_numpy()[1:]
    features_to_discard = [x for x in recs.columns if 
                     (x.startswith('Z')) | 
                     (x.startswith('NWEIGHT') & (x != 'NWEIGHT'))] 
    
    df = recs.copy()
    df_computed = recs.copy()

    # According to the codebook, replace the NA flags (-2, -4, 99, etc.) with the correct default value
    for _, row in cb.codebook[~np.isnan(cb.codebook['Target1'])].iterrows():
        df_computed.loc[df_computed[row['Variable']] == row['Handle1'], row['Variable']] = row['Target1']

    for i_,  row in cb.codebook[~np.isnan(cb.codebook['Target2'])].iterrows():
        df_computed.loc[df_computed[row['Variable']] == row['Handle2'], row['Variable']] = row['Target2']


    df_computed['climate_code_heat'] = df_computed['IECC_climate_code'].apply(lambda x: parse_climate_code(x)[0])

    df_computed['climate_code_humidity'] = df_computed['IECC_climate_code'].apply(lambda x: parse_climate_code(x)[1])

    df_computed = df_computed.fillna(0)

    # YEARMADERANGE
    mapping_year_made_range = {1: 1935, 2: 1955, 3: 1965, 4: 1975, 5: 1985, 6: 1995, 7: 2005, 8: 2015, 9: 2018}
    # WINDOWS
    mapping_num_windows = {1: 2, 2: 4, 3: 7, 4: 13, 5: 17, 6: 25, 7: 33}
    # TYPERFR1 TYPERFR2
    mapping_app_door_type = {1: 1, 2: 2, 3: 2, 4: 2, 5: 3, -2: 0}
    # AGERFRI1 AGERFRI2 AGEFRZR AGEDW AGECWASH AGECDRYER WHEATAGE
    mapping_app_age = {1: 1, 2: 3, 3: 7, 4: 12, 5: 17, 6: 23, -2: 0}
    # NUMMEAL
    mapping_meal_freq = {1: 22, 2: 14, 3: 7, 4: 4, 5: 1, 6: 0.5, 0: 0} # times per week
    # USECOFFEE
    mapping_coffee_freq = {1: 1, 2: 0.5, 3: 0.2, 0: 0} # use per day
    # USEEQUIPAUX
    mapping_sec_heat = {1: 1, 2: 0.3, 3: 0.1, 4: 0.05, 5: 0.02, -2: 0} # use per day (probability)
    # USEHUMID USECFAN USEDEHUM
    mapping_latent_freq = {1: 0.05, 2: 0.1, 3: 0.5, 4: 0.9, 99: 0.05, -1: 0} # use per day (probability)
    # TEMPHOME TEMPGONE TEMPNITE TEMPHOMEAC TEMPGONEAC TEMPNITEAC
    mapping_temp = {-2: 75}
    # WHEATSIZ
    mapping_tank_size = {1: 20, 2: 40, 3: 60, 4: 0}
    # LGTINLED LGTINCFL LGTINCAN
    mapping_portion = {1: 1, 2: 0.75, 3: 0.5, 4: 0.25, 5: 0}
    # LGTOUTNITE
    mapping_num_lights = {-1: 0}
    # EVCHRGHOME
    mapping_EV = {1: 1, 0: 0, -2: 0, '.': 0}
    # MONEYPY
    mapping_money = {
        1: 4000,     # Less than $5,000
        2: 6000,     # $5,000 - $7,499
        3: 8750,     # $7,500 - $9,999
        4: 11250,    # $10,000 - $12,499
        5: 13750,    # $12,500 - $14,999
        6: 17500,    # $15,000 - $19,999
        7: 22500,    # $20,000 - $24,999
        8: 27500,    # $25,000 - $29,999
        9: 32500,    # $30,000 - $34,999
        10: 37500,   # $35,000 - $39,999
        11: 45000,   # $40,000 - $49,999
        12: 55000,   # $50,000 - $59,999
        13: 67500,   # $60,000 - $74,999
        14: 87500,   # $75,000 - $99,999
        15: 125000,  # $100,000 - $149,999
        16: 175000   # $150,000 or more
    }

    # state_postal
    mapping_state = {
        'AL': 1,
        'AK': 2,
        'AZ': 4,
        'AR': 5,
        'CA': 6,
        'CO': 8,
        'CT': 9,
        'DE': 10,
        'DC': 11,
        'FL': 12,
        'GA': 13,
        'HI': 15,
        'ID': 16,
        'IL': 17,
        'IN': 18,
        'IA': 19,
        'KS': 20,
        'KY': 21,
        'LA': 22,
        'ME': 23,
        'MD': 24,
        'MA': 25,
        'MI': 26,
        'MN': 27,
        'MS': 28,
        'MO': 29,
        'MT': 30,
        'NE': 31,
        'NV': 32,
        'NH': 33,
        'NJ': 34,
        'NM': 35,
        'NY': 36,
        'NC': 37,
        'ND': 38,
        'OH': 39,
        'OK': 40,
        'OR': 41,
        'PA': 42,
        'RI': 44,
        'SC': 45,
        'SD': 46,
        'TN': 47,
        'TX': 48,
        'UT': 49,
        'VT': 50,
        'VA': 51,
        'WA': 53,
        'WV': 54,
        'WI': 55,
        'WY': 56
    }

    # Correlation Dictionary
    mapping_correlations = {
        # YEARMADERANGE
        'YEARMADERANGE': mapping_year_made_range,

        # WINDOWS
        'WINDOWS': mapping_num_windows,

        # TYPERFR1 TYPERFR2
        'TYPERFR1': mapping_app_door_type,
        'TYPERFR2': mapping_app_door_type,

        # AGERFRI1 AGERFRI2 AGEFRZR AGEDW AGECWASH AGECDRYER ACEQUIPAGE WHEATAGE
        'AGERFRI1': mapping_app_age,
        'AGERFRI2': mapping_app_age,
        'AGEFRZR': mapping_app_age,
        'AGEDW': mapping_app_age,
        'AGECWASH': mapping_app_age,
        'AGECDRYER': mapping_app_age,
        #'ACEQUIPAGE': mapping_app_age,
        'WHEATAGE': mapping_app_age,

        # NUMMEAL
        'NUMMEAL': mapping_meal_freq,

        # USECOFFEE
        'USECOFFEE': mapping_coffee_freq,

        # USEEQUIPAUX
        'USEEQUIPAUX': mapping_sec_heat,

        # USEHUMID USECFAN USEDEHUM
        'USEHUMID': mapping_latent_freq,
        'USECFAN': mapping_latent_freq,
        'USEDEHUM': mapping_latent_freq,

        # TEMPHOME TEMPGONE TEMPNITE TEMPHOMEAC TEMPGONEAC TEMPNITEAC
        'TEMPHOME': mapping_temp,
        'TEMPGONE': mapping_temp,
        'TEMPNITE': mapping_temp,
        'TEMPHOMEAC': mapping_temp,
        'TEMPGONEAC': mapping_temp,
        'TEMPNITEAC': mapping_temp,

        # WHEATSIZ
        'WHEATSIZ': mapping_tank_size,

        # LGTINLED LGTINCFL LGTINCAN
        'LGTINLED': mapping_portion,
        'LGTINCFL': mapping_portion,
        'LGTINCAN': mapping_portion,

        # LGTOUTNITE
        'LGTOUTNITE': mapping_num_lights,

        # EVCHRGHOME
        'EVCHRGHOME': mapping_EV,

        # MONEYPY
        'MONEYPY': mapping_money,

        # state_postal
        'state_postal': mapping_state
    }


    for c, mapping in mapping_correlations.items():
        df_computed[c] = parse_column(df_computed[c], mapping)



    known_variables = {}
    template_variables = {}
    known_variables['Known'] = cb.codebook[cb.codebook['Availability'] == 'Known']['Variable'].values.tolist()
    known_variables['Imputable'] = cb.codebook[cb.codebook['Availability'] == 'Imputable']['Variable'].values.tolist()
    for u in cb.codebook[~cb.codebook['Template'].isna()]['Template'].unique():
        template_variables[u] = cb.codebook[cb.codebook['Template'] == u]['Variable'].values.tolist()
    known_variables.keys(), template_variables.keys()




    df_computed['total_sqm_en'] = df['SQFTEST'].apply(SQF2SQM)

    df_computed['total_kwh_elec'] = df['KWHSPH'] + df['KWHCOL'] + df['KWHWTH'] + df['KWHRFG'] + df[
    'KWHFRZ'] + df['KWHCOK'] + df['KWHMICRO'] + df['KWHCW'] + df['KWHCDR'] + df['KWHDWH'] + df[
    'KWHLGT'] + df['KWHTVREL'] + df['KWHAHUHEAT'] + df['KWHAHUCOL'] + df['KWHCFAN'] + df[
    'KWHDHUM'] + df['KWHHUM'] + df['KWHPLPMP'] + df['KWHHTBPMP'] + df['KWHHTBHEAT'] + df[
    'KWHEVCHRG'] + df['KWHNEC'] + df['KWHNEC']
    df_computed['total_kwh'] = df['TOTALBTU'].apply(BTU2KWH)  * 1000
    df_computed['total_kwh_sph'] = df['TOTALBTUSPH'].apply(BTU2KWH) * 1000


    df_computed['ng_kwh_cooking'] = df['BTUNGCOK'].apply(BTU2KWH)  * 1000
    df_computed['ng_kwh_clothes_dryers'] = df['BTUNGCDR'].apply(BTU2KWH)  * 1000 
    df_computed['lp_kwh_cooking'] = df['BTULPCOK'].apply(BTU2KWH)  * 1000
    df_computed['lp_kwh_clothes_dryers'] = df['BTULPCDR'].apply(BTU2KWH)  * 1000

    df_computed['total_kwh_appliances'] = df['KWHRFG'] + df['KWHFRZ'] + df['KWHCOK'] + df['KWHMICRO'] + df['KWHCW'] + df['KWHCDR'] + df[
    'KWHDWH'] + df_computed['ng_kwh_cooking'] + df_computed['ng_kwh_clothes_dryers'] + df_computed[
    'lp_kwh_clothes_dryers'] + df_computed['lp_kwh_cooking']
    df_computed['total_kwh_dhw'] = df['TOTALBTUWTH'].apply(BTU2KWH)  * 1000 
    df_computed['total_kwh_lighting'] = df['KWHLGT']
    df_computed['total_kwh_electronics'] = df['KWHTVREL']
    df_computed['total_kwh_vent'] = df['KWHAHUHEAT'] + df['KWHAHUCOL'] + df['KWHCFAN'] + df['KWHDHUM'] + df['KWHHUM'] + df['KWHPLPMP'] + df['KWHHTBPMP']
    df_computed['total_kwh_col'] = df['KWHCOL']
    # combined energy categories
    df_computed['total_kwh_cooking'] = df['KWHCOK'] + df_computed['ng_kwh_cooking'] + df_computed['lp_kwh_cooking']
    df_computed['total_kwh_clothes_dryers'] = df['KWHCDR'] + df_computed['ng_kwh_clothes_dryers'] + df_computed['lp_kwh_clothes_dryers']
    # per-appliance metrics
    df_computed['total_kwh_per_refrigerator'] = df['KWHRFG'] / df['NUMFRIG']
    df_computed['total_kwh_per_freezer'] = df['KWHFRZ'] / df['NUMFREEZ']
    # ? cooktop TBD
    df_computed['total_kwh_per_microwave'] = df['KWHMICRO'] / df['MICRO']

    df_computed['total_catering'] =  df['KWHCOK'] + df['KWHMICRO'] +  df[
    'KWHDWH'] + df_computed['ng_kwh_cooking'] +  df_computed['lp_kwh_cooking']
    df_computed['total_kwh_housework'] = df['KWHCW'] + df['KWHCDR'] + df_computed['ng_kwh_clothes_dryers'] + df_computed['lp_kwh_clothes_dryers']


    # eui metrics
    df_computed['eui_kwh_sqm'] = df_computed['total_kwh']/ df_computed['total_sqm_en']
    df_computed['total_elec_eui_kwh_sqm'] = df_computed['total_kwh_elec']/ df_computed['total_sqm_en']

    df_computed['heating_eui_kwh_sqm'] = df_computed['total_kwh_sph']/ df_computed['total_sqm_en']

    df_computed['appliances_eui_kwh_sqm'] = df_computed['total_kwh_appliances']/df_computed['total_sqm_en']
    df_computed['refrigerator_eui_kwh_sqm'] = df_computed['KWHRFG']/df_computed['total_sqm_en']
    df_computed['freezer_eui_kwh_sqm'] = df_computed['KWHFRZ']/df_computed['total_sqm_en']
    df_computed['frigfreez_eui_kwh_sqm'] = (df_computed['KWHRFG'] + df_computed['KWHFRZ'])/df_computed['total_sqm_en']
    df_computed['cooking_eui_kwh_sqm'] = df_computed['total_kwh_cooking']/df_computed['total_sqm_en']
    df_computed['cooking_elec_eui_kwh_sqm'] = df_computed['KWHCOK']/df_computed['total_sqm_en']
    df_computed['cooking_fuel_eui_kwh_sqm'] = (df_computed['ng_kwh_cooking'] + df_computed['lp_kwh_cooking'])/df_computed['total_sqm_en']
    df_computed['clothes_dryers_eui_kwh_sqm'] = df_computed['total_kwh_clothes_dryers']/df_computed['total_sqm_en']
    df_computed['clothes_dryers_elec_eui_kwh_sqm'] = df['KWHCDR']/df_computed['total_sqm_en']
    df_computed['clothes_dryers_fuel_eui_kwh_sqm'] = (df_computed['ng_kwh_clothes_dryers'] + df_computed['lp_kwh_clothes_dryers'])/df_computed['total_sqm_en']
    df_computed['housework_eui_kwh_sqm'] = df_computed['total_kwh_housework']/df_computed['total_sqm_en']
    df_computed['housework_elec_eui_kwh_sqm'] = (df['KWHCW'] + df['KWHCDR'])/df_computed['total_sqm_en']
    df_computed['housework_fuel_eui_kwh_sqm'] = (df_computed['ng_kwh_clothes_dryers'] + df_computed['lp_kwh_clothes_dryers'])/df_computed['total_sqm_en']

    df_computed['dhw_eui_kwh_sqm'] = df_computed['total_kwh_dhw']/ df_computed['total_sqm_en']
    df_computed['lighting_eui_kwh_sqm'] = df_computed['total_kwh_lighting']/ df_computed['total_sqm_en']
    df_computed['electronics_eui_kwh_sqm'] = df_computed['total_kwh_electronics']/ df_computed['total_sqm_en']
    df_computed['vent_eui_kwh_sqm'] = df_computed['total_kwh_vent']/ df_computed['total_sqm_en']
    df_computed['cooling_eui_kwh_sqm'] = df_computed['total_kwh_col']/ df_computed['total_sqm_en']
    computed_columns = ['total_sqm_en', 'total_kwh', 'total_kwh_sph', 'total_kwh_appliances', 'total_kwh_dhw', 'total_kwh_lighting', 'total_kwh_electronics', 'total_kwh_vent', 'total_kwh_col',
    'eui_kwh_sqm', 'heating_eui_kwh_sqm', 'appliances_eui_kwh_sqm', 'dhw_eui_kwh_sqm', 'lighting_eui_kwh_sqm', 'electronics_eui_kwh_sqm', 'vent_eui_kwh_sqm', 'cooling_eui_kwh_sqm']


    # These are all preprocessed feature columns
    cols_discarded = cb.codebook[cb.codebook['Preserved'] <= .01]['Variable'].values.tolist()
    cols_continuous = cb.codebook[(cb.codebook['Preserved'] >= 0.99) & ((cb.codebook['Notes'] == 'Numerical') & (cb.codebook['NaiveScale'] != 1))]['Variable'].values.tolist()
    cols_scaled = cb.codebook[(cb.codebook['Preserved'] >= 0.99) & ((cb.codebook['Notes'] == 'Numerical') & (cb.codebook['NaiveScale'] == 1))]['Variable'].values.tolist() + ['climate_code_heat']
    cols_categorical = cb.codebook[(cb.codebook['Preserved'] >= 0.99) & (cb.codebook['Notes'] == 'Categorical')]['Variable'].values.tolist() + ['climate_code_humidity']

    # These are derived/computed columns
    cols_computed = ['total_sqm_en', 'total_kwh',
        'total_kwh_sph', 'total_kwh_appliances', 'total_kwh_dhw',
        'total_kwh_lighting', 'total_kwh_electronics', 'total_kwh_vent',
        'total_kwh_col', 'eui_kwh_sqm', 'heating_eui_kwh_sqm',
        'appliances_eui_kwh_sqm', 'dhw_eui_kwh_sqm', 'lighting_eui_kwh_sqm',
        'electronics_eui_kwh_sqm', 'vent_eui_kwh_sqm', 'cooling_eui_kwh_sqm']


    walltype_group = {1: 16, 2: 2, 3: 3, 4: 16, 5: 2, 6: 16, 7: 16, 99: 99}
    acequipm_pub_group = {1: 1}
    df_computed['walltype_grouped'] = df_computed['WALLTYPE'].apply(lambda x: make_groups(x, walltype_group))
    df_computed['acequipm_pub_grouped'] = df_computed['ACEQUIPM_PUB'].apply(lambda x: make_groups(x, acequipm_pub_group, default=0))
    df_computed['urban_grouped'] = df_computed['UATYP10'].apply(lambda x: 0 if x == 'R' else 1)
    df_computed['num_u65'] = df_computed['NUMCHILD'] + df_computed['NUMADULT1']
    df_computed['num_occupant'] = df_computed['NUMCHILD'] + df_computed['NUMADULT1'] + df_computed['NUMADULT2']

    df_computed['total_kwh_elec_sph'] = df['KWHSPH'] + df['KWHAHUHEAT']
    df_computed['total_kwh_elec_fr'] = df['KWHRFG'] + df['KWHFRZ']
    df_computed['total_kwh_elec_catering'] = df['KWHCOK'] + df['KWHMICRO'] 
    df_computed['total_kwh_elec_housework'] = df['KWHCDR'] + df['KWHCW']
    df_computed['total_kwh_elec_lighting'] =  df['KWHLGT']
    df_computed['total_kwh_elec_electronics'] =  df['KWHTVREL']
    df_computed['total_kwh_elec_col'] = df['KWHAHUCOL'] + df['KWHCFAN']
    df_computed['total_kwh_elec_latent'] = df['KWHDHUM'] + df['KWHHUM']
    df_computed['total_kwh_elec_ev'] = df['KWHEVCHRG']
    df_computed['total_kwh_elec_dhw'] = df['KWHHTBHEAT'] + df['KWHDWH'] 
    df_computed['total_kwh_elec_other'] = df['KWHPLPMP'] + df['KWHHTBPMP'] + df['KWHHTBHEAT'] + df['KWHNEC']

    df_computed['total_kwh_elec_thermal'] = df_computed['total_kwh_elec_sph'] + df_computed['total_kwh_elec_col'] + df_computed['total_kwh_elec_latent'] + df_computed['total_kwh_elec_dhw']
    df_computed['total_kwh_elec_constant'] = df_computed['total_kwh_elec_fr'] + df_computed['total_kwh_elec_lighting'] 
    df_computed['total_kwh_elec_activity'] = df_computed['total_kwh_elec_catering']  + df_computed['total_kwh_elec_housework'] + df_computed['total_kwh_elec_ev'] + df_computed['total_kwh_elec_electronics'] + + df_computed['total_kwh_elec_other'] 

    df_computed['RC_equipment_power'] = df_computed[
    'total_kwh_elec_fr'] + df_computed['total_kwh_elec_catering'] + df_computed['total_kwh_elec_housework'] + df_computed[
    'total_kwh_elec_electronics'] + df_computed['total_kwh_elec_other'] + df_computed[
    'total_kwh_elec_ev'] + df_computed['total_kwh_elec_latent']



    df_computed['total_btu_ng'] = df['BTUNG']
    df_computed['total_btu_ng_sph'] = df['BTUNGSPH']
    df_computed['total_btu_ng_dhw'] = df['BTUNGWTH'] + df['BTUNGPLHEAT'] + df['BTUNGHTBHEAT']
    df_computed['total_btu_ng_catering'] = df['BTUNGCOK']
    df_computed['total_btu_ng_housework'] = df['BTUNGCDR']
    df_computed['total_btu_ng_other'] = df['BTUNGNEC']

    df_computed['total_btu_ofuel'] = df['BTULP'] + df['BTUFO'] + df['BTUWD']
    df_computed['total_btu_ofuel_sph'] = df['BTULPSPH'] + df['BTUFOSPH'] + df['BTUWD']
    df_computed['total_btu_ofuel_dhw'] = df['BTULPWTH'] + df['BTUFOWTH']
    df_computed['total_btu_ofuel_catering'] = df['BTULPCOK']
    df_computed['total_btu_ofuel_housework'] = df['BTULPCDR']
    df_computed['total_btu_ofuel_other'] = df['BTULPNEC'] + df['BTUFONEC']

    df_computed['total_btu_ng_thermal'] = df_computed['total_btu_ng_sph'] 
    df_computed['total_btu_ng_activity'] = df_computed['total_btu_ng_housework'] + df_computed['total_btu_ng_dhw'] + df_computed['total_btu_ng_catering'] + df_computed['total_btu_ng_other']

    df_computed['total_btu_ofuel_thermal'] = df_computed['total_btu_ofuel_sph'] 
    df_computed['total_btu_ofuel_activity'] = df_computed['total_btu_ofuel_housework'] + df_computed['total_btu_ofuel_dhw'] + df_computed['total_btu_ofuel_catering'] + df_computed['total_btu_ofuel_other']

    df_computed['RC_gas_power'] =  df_computed['total_btu_ofuel_catering'] + df_computed[
    'total_btu_ofuel_housework'] + df_computed['total_btu_ofuel_other'] + df_computed['total_btu_ng_catering'] + df_computed['total_btu_ng_housework'] + df_computed['total_btu_ng_other']




    #----------------------------------#
    df_computed['pp_kwh_elec'] = df['KWH'] / df_computed['num_occupant'] 
    df_computed['pp_kwh_elec_sph'] = df_computed['total_kwh_elec_sph'] / df_computed['num_occupant'] 
    df_computed['pp_kwh_elec_fr'] = df_computed['total_kwh_elec_fr'] / df_computed['num_occupant'] 
    df_computed['pp_kwh_elec_catering'] = df_computed['total_kwh_elec_catering']/ df_computed['num_occupant'] 
    df_computed['pp_kwh_elec_housework'] = df_computed['total_kwh_elec_housework']/ df_computed['num_occupant'] 
    df_computed['pp_kwh_elec_lighting'] =  df_computed['total_kwh_elec_lighting']/ df_computed['num_occupant'] 
    df_computed['pp_kwh_elec_electronics'] =  df_computed['total_kwh_elec_electronics']/ df_computed['num_occupant'] 
    df_computed['pp_kwh_elec_col'] = df_computed['total_kwh_elec_col']/ df_computed['num_occupant'] 
    df_computed['pp_kwh_elec_latent'] = df_computed['total_kwh_elec_latent']/ df_computed['num_occupant'] 
    df_computed['pp_kwh_elec_ev'] = df_computed['total_kwh_elec_ev']/ df_computed['num_occupant'] 
    df_computed['pp_kwh_elec_dhw'] = df_computed['total_kwh_elec_dhw']/ df_computed['num_occupant'] 
    df_computed['pp_kwh_elec_other'] = df_computed['total_kwh_elec_other']/ df_computed['num_occupant'] 

    df_computed['pp_kwh_elec_thermal'] = df_computed['total_kwh_elec_thermal'] / df_computed['num_occupant'] 
    df_computed['pp_kwh_elec_constant'] = df_computed['total_kwh_elec_constant']/ df_computed['num_occupant'] 
    df_computed['pp_kwh_elec_activity'] = df_computed['total_kwh_elec_activity']/ df_computed['num_occupant'] 

    df_computed['pp_btu_ng'] = df['BTUNG'] / df_computed['num_occupant'] 
    df_computed['pp_btu_ng_sph'] = df_computed['total_btu_ng_sph']/ df_computed['num_occupant'] 
    df_computed['pp_btu_ng_dhw'] = df_computed['total_btu_ng_dhw']/ df_computed['num_occupant'] 
    df_computed['pp_btu_ng_catering'] = df_computed['total_btu_ng_catering']/ df_computed['num_occupant'] 
    df_computed['pp_btu_ng_housework'] = df_computed['total_btu_ng_housework']/ df_computed['num_occupant'] 
    df_computed['pp_btu_ng_other'] = df_computed['total_btu_ng_other']/ df_computed['num_occupant'] 

    df_computed['pp_btu_ng_thermal'] = df_computed['total_btu_ng_thermal']/ df_computed['num_occupant'] 
    df_computed['pp_btu_ng_activity'] = df_computed['total_btu_ng_activity']/ df_computed['num_occupant'] 

    df_computed['pp_btu_ofuel_thermal'] = df_computed['total_btu_ofuel_thermal']/ df_computed['num_occupant'] 
    df_computed['pp_btu_ofuel_activity'] = df_computed['total_btu_ofuel_activity']/ df_computed['num_occupant'] 
    #----------------------------------#
    df_computed['eui_kwh_elec'] = df['KWH'] / df_computed['total_sqm_en'] 
    df_computed['eui_kwh_elec_sph'] = df_computed['total_kwh_elec_sph'] / df_computed['total_sqm_en'] 
    df_computed['eui_kwh_elec_fr'] = df_computed['total_kwh_elec_fr'] / df_computed['total_sqm_en']
    df_computed['eui_kwh_elec_catering'] = df_computed['total_kwh_elec_catering']/ df_computed['total_sqm_en']
    df_computed['eui_kwh_elec_housework'] = df_computed['total_kwh_elec_housework']/ df_computed['total_sqm_en'] 
    df_computed['eui_kwh_elec_lighting'] =  df_computed['total_kwh_elec_lighting']/ df_computed['total_sqm_en'] 
    df_computed['eui_kwh_elec_electronics'] =  df_computed['total_kwh_elec_electronics']/ df_computed['total_sqm_en'] 
    df_computed['eui_kwh_elec_col'] = df_computed['total_kwh_elec_col']/ df_computed['total_sqm_en'] 
    df_computed['eui_kwh_elec_latent'] = df_computed['total_kwh_elec_latent']/ df_computed['total_sqm_en'] 
    df_computed['eui_kwh_elec_ev'] = df_computed['total_kwh_elec_ev']/ df_computed['total_sqm_en'] 
    df_computed['eui_kwh_elec_dhw'] = df_computed['total_kwh_elec_dhw']/ df_computed['total_sqm_en'] 
    df_computed['eui_kwh_elec_other'] = df_computed['total_kwh_elec_other']/ df_computed['total_sqm_en'] 

    df_computed['eui_kwh_elec_thermal'] = df_computed['total_kwh_elec_thermal'] / df_computed['total_sqm_en']
    df_computed['eui_kwh_elec_constant'] = df_computed['total_kwh_elec_constant']/ df_computed['total_sqm_en']
    df_computed['eui_kwh_elec_activity'] = df_computed['total_kwh_elec_activity']/ df_computed['total_sqm_en'] 

    df_computed['RC_equipment_power_density'] = df_computed['RC_equipment_power'] / df_computed['total_sqm_en'] * 1000 / 8760 # [W/m2]
    df_computed['RC_DHW_elec_power_density'] = df_computed['total_kwh_elec_dhw'] / df_computed['total_sqm_en'] * 1000 / 8760 # [W/m2]
    df_computed['RC_DHW_gas_power_density'] = (df_computed['total_btu_ofuel_dhw'] + df_computed['total_btu_ng_dhw']) / df_computed['total_sqm_en'] * 1000 / 8760 # [W/m2]
    df_computed['RC_gas_power_density'] = df_computed['RC_gas_power'] / df_computed['total_sqm_en'] * 1000 / 8760 # [W/m2]
    df_computed['RC_lighting_power_density'] = df_computed['KWHLGT'] / df_computed['total_sqm_en'] * 1000 / 8760 # [W/m2]

    df_computed['eui_btu_ng'] = df['BTUNG'] / df_computed['total_sqm_en'] 
    df_computed['eui_btu_ng_sph'] = df_computed['total_btu_ng_sph']/ df_computed['total_sqm_en'] 
    df_computed['eui_btu_ng_dhw'] = df_computed['total_btu_ng_dhw']/ df_computed['total_sqm_en']
    df_computed['eui_btu_ng_catering'] = df_computed['total_btu_ng_catering']/df_computed['total_sqm_en']
    df_computed['eui_btu_ng_housework'] = df_computed['total_btu_ng_housework']/ df_computed['total_sqm_en']
    df_computed['eui_btu_ng_other'] = df_computed['total_btu_ng_other']/ df_computed['total_sqm_en']

    df_computed['eui_btu_ng_thermal'] = df_computed['total_btu_ng_thermal']/ df_computed['total_sqm_en']  
    df_computed['eui_btu_ng_activity'] = df_computed['total_btu_ng_activity']/ df_computed['total_sqm_en']  

    df_computed['eui_btu_ofuel'] = df_computed['total_btu_ofuel'] / df_computed['total_sqm_en']
    df_computed['eui_btu_ofuel_sph'] = df_computed['total_btu_ofuel_sph'] / df_computed['total_sqm_en']
    df_computed['eui_btu_ofuel_thermal'] = df_computed['total_btu_ofuel_thermal']/ df_computed['total_sqm_en']  
    df_computed['eui_btu_ofuel_activity'] = df_computed['total_btu_ofuel_activity']/ df_computed['total_sqm_en']  

    df_computed['window_per_sqm'] = df_computed['WINDOWS'] / df_computed['total_sqm_en'] 
    df_computed['door_per_sqm'] = df_computed['DOOR1SUM'] / df_computed['total_sqm_en']


    df_computed = df_computed.copy()  # this defragments the DataFrame


    return df_computed
# 0521 checkpoint
# existing_cols = {
#     'usace_sqft': 'TOTSQFT_EN',
#     'typehuq': 'TYPEHUQ',
#     'urban': 'urban_grouped',
#     'acequipm_pub': 'acequipm_pub_grouped',
#     'fuelheat': 'FUELHEAT',
#     'equipm': 'EQUIPM',
#     'year_built': 'YEARMADERANGE',
#     'cellar': 'CELLAR',
#     'wall_type': 'WALLTYPE',
#     'basefin': 'BASEFIN',
#     'num_u65': 'num_u65',
#     'num_o65': 'NUMADULT2',
#     'el_annual': 'corrected_kwh',
#     'gs_annual': 'corrected_thmng'
# }

cb = Codebook()
recs = pd.read_csv(path_to_recs_microdata, index_col=0)
df_computed = preprocessing(recs, cb)
