import torch
import random
import numpy as np
import mdtraj as md
import os
from rdkit import Chem
from torch_geometric.data import Data, Dataset
import random
from tqdm import tqdm
from typing import Literal
import warnings
import pandas as pd

import pickle
from rdkit.Chem.rdchem import HybridizationType, BondType

# Suppress RDKit warnings about hydrogen removal
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.warning')
from .feature_utils import get_node_features
from .transforms import *
from .mistake_fixer import TrajFixer

from utils.data_filter import filter_data, get_smiles_dict
from typing import Literal, Any, Optional, Set
import pandas as pd

# import sys
# sys.path.append


########################################################
'''
UTILS
'''
########################################################


# Constants
BOND_TYPES = {t: i + 1 for i, t in enumerate(BondType.names.values())}
NUM_FRAMES = 12500
IGNORE = [
 'C_C_C_C_C_O_N_C_H_c1ccccc1_C_H_O_C_O_O_C_H_1C_C_2_O_C_H_OC_O_c3ccccc3_C_H_3_C_4_OC_C_O_CO_C_H_4C_C_H_O_C_3_C_C_O_C_H_O_C_C1C_C2_C_C_42',
 'C_C_C_C_C_O_N_C_H_c1ccccc1_C_H_O_C_O_O_C_H_1C_C_2_O_C_H_OC_O_c3ccccc3_C_H_3_C_4_OC_C_O_CO_C_H_4C_C_H_O_C_3_C_C_O_C_H_O_C_C1C_C2_C_C_41',
 'C_C_C_C_C_O_N_C_H_c1ccccc1_C_H_O_C_O_O_C_H_1C_C_2_O_C_H_OC_O_c3ccccc3_C_H_3_C_4_OC_C_O_CO_C_H_4C_C_H_O_C_3_C_C_O_C_H_O_C_C1C_C2_C_C_43',
 'C_C_C_C_C_O_N_C_H_c1ccccc1_C_H_O_C_O_O_C_H_1C_C_2_O_C_H_OC_O_c3ccccc3_C_H_3_C_4_OC_C_O_CO_C_H_4C_C_H_O_C_3_C_C_O_C_H_O_C_C1C_C2_C_C_44',
 'C_C_C_C_C_O_N_C_H_c1ccccc1_C_H_O_C_O_O_C_H_1C_C_2_O_C_H_OC_O_c3ccccc3_C_H_3_C_4_OC_C_O_CO_C_H_4C_C_H_O_C_3_C_C_O_C_H_O_C_C1C_C2_C_C_40',
 'C_C_1_C_O_O_CC_C_2_C_CC_C_3_C_C_CC_O_C_H_4_C_5_C_CC_C_H_O_C_H_6O_C_H_C_O_O-_C_H_O_C_H_O_C_H_6O_C_H_6O_C_H_C_O_O-_C_H_O_C_H_O_C_H_6O_C_C_C_C_H_5CC_C_43C_C_H_2C1_131',
 'C_C_1_C_O_O_CC_C_2_C_CC_C_3_C_C_CC_O_C_H_4_C_5_C_CC_C_H_O_C_H_6O_C_H_C_O_O-_C_H_O_C_H_O_C_H_6O_C_H_6O_C_H_C_O_O-_C_H_O_C_H_O_C_H_6O_C_C_C_C_H_5CC_C_43C_C_H_2C1_132',
 'C_C_1_C_O_O_CC_C_2_C_CC_C_3_C_C_CC_O_C_H_4_C_5_C_CC_C_H_O_C_H_6O_C_H_C_O_O-_C_H_O_C_H_O_C_H_6O_C_H_6O_C_H_C_O_O-_C_H_O_C_H_O_C_H_6O_C_C_C_C_H_5CC_C_43C_C_H_2C1_130',
 'C_C_1_C_O_O_CC_C_2_C_CC_C_3_C_C_CC_O_C_H_4_C_5_C_CC_C_H_O_C_H_6O_C_H_C_O_O-_C_H_O_C_H_O_C_H_6O_C_H_6O_C_H_C_O_O-_C_H_O_C_H_O_C_H_6O_C_C_C_C_H_5CC_C_43C_C_H_2C1_134',
 'C_C_1_C_O_O_CC_C_2_C_CC_C_3_C_C_CC_O_C_H_4_C_5_C_CC_C_H_O_C_H_6O_C_H_C_O_O-_C_H_O_C_H_O_C_H_6O_C_H_6O_C_H_C_O_O-_C_H_O_C_H_O_C_H_6O_C_C_C_C_H_5CC_C_43C_C_H_2C1_133',
 'CC_C_H_1OC_O_C_H_C_C_H_O_C_H_2C_C_C_OC_C_H_O_C_H_C_O2_C_H_C_C_H_O_C_H_2O_C_H_C_C_C_H_N_C_C_C_H_2O_C_C_O_C_C_H_C_C_H_2N_C_H_COCCOC_O_C_H_C_H_2C_C_1_C_O_48',
 'CC_C_H_1OC_O_C_H_C_C_H_O_C_H_2C_C_C_OC_C_H_O_C_H_C_O2_C_H_C_C_H_O_C_H_2O_C_H_C_C_C_H_N_C_C_C_H_2O_C_C_O_C_C_H_C_C_H_2N_C_H_COCCOC_O_C_H_C_H_2C_C_1_C_O_45',
 'CC_C_H_1OC_O_C_H_C_C_H_O_C_H_2C_C_C_OC_C_H_O_C_H_C_O2_C_H_C_C_H_O_C_H_2O_C_H_C_C_C_H_N_C_C_C_H_2O_C_C_O_C_C_H_C_C_H_2N_C_H_COCCOC_O_C_H_C_H_2C_C_1_C_O_46',
 'CC_C_H_1OC_O_C_H_C_C_H_O_C_H_2C_C_C_OC_C_H_O_C_H_C_O2_C_H_C_C_H_O_C_H_2O_C_H_C_C_C_H_N_C_C_C_H_2O_C_C_O_C_C_H_C_C_H_2N_C_H_COCCOC_O_C_H_C_H_2C_C_1_C_O_49',
 'CC_C_H_1OC_O_C_H_C_C_H_O_C_H_2C_C_C_OC_C_H_O_C_H_C_O2_C_H_C_C_H_O_C_H_2O_C_H_C_C_C_H_N_C_C_C_H_2O_C_C_O_C_C_H_C_C_H_2N_C_H_COCCOC_O_C_H_C_H_2C_C_1_C_O_47',
 'CC_C_1_O_C_C_H_2CN_CCc3c_nH_c4ccccc34_C_C_O_OC_c3cc4c_cc3OC_N_C_O_C_H_3_C_O_C_O_OC_C_H_OC_C_O_C_5_CC_C_CC_N_H_6CC_C_43_C_H_65_C2_C1_12',
 'CC_C_1_O_C_C_H_2CN_CCc3c_nH_c4ccccc34_C_C_O_OC_c3cc4c_cc3OC_N_C_O_C_H_3_C_O_C_O_OC_C_H_OC_C_O_C_5_CC_C_CC_N_H_6CC_C_43_C_H_65_C2_C1_11',
 'CC_C_1_O_C_C_H_2CN_CCc3c_nH_c4ccccc34_C_C_O_OC_c3cc4c_cc3OC_N_C_O_C_H_3_C_O_C_O_OC_C_H_OC_C_O_C_5_CC_C_CC_N_H_6CC_C_43_C_H_65_C2_C1_13',
 'CC_C_1_O_C_C_H_2CN_CCc3c_nH_c4ccccc34_C_C_O_OC_c3cc4c_cc3OC_N_C_O_C_H_3_C_O_C_O_OC_C_H_OC_C_O_C_5_CC_C_CC_N_H_6CC_C_43_C_H_65_C2_C1_14',
 'CC_C_1_O_C_C_H_2CN_CCc3c_nH_c4ccccc34_C_C_O_OC_c3cc4c_cc3OC_N_C_O_C_H_3_C_O_C_O_OC_C_H_OC_C_O_C_5_CC_C_CC_N_H_6CC_C_43_C_H_65_C2_C1_10',
 'CC_O_N_C_C_H_O_CN_C_C_O_c1c_I_c_C_O_NC_C_H_O_CO_c_I_c_C_O_NC_C_H_O_CO_c1I_c1c_I_c_C_O_NC_C_H_O_CO_c_I_c_C_O_NC_C_H_O_CO_c1I_142',
 'CC_O_N_C_C_H_O_CN_C_C_O_c1c_I_c_C_O_NC_C_H_O_CO_c_I_c_C_O_NC_C_H_O_CO_c1I_c1c_I_c_C_O_NC_C_H_O_CO_c_I_c_C_O_NC_C_H_O_CO_c1I_141',
 'CC_O_N_C_C_H_O_CN_C_C_O_c1c_I_c_C_O_NC_C_H_O_CO_c_I_c_C_O_NC_C_H_O_CO_c1I_c1c_I_c_C_O_NC_C_H_O_CO_c_I_c_C_O_NC_C_H_O_CO_c1I_143',
 'CC_O_N_C_C_H_O_CN_C_C_O_c1c_I_c_C_O_NC_C_H_O_CO_c_I_c_C_O_NC_C_H_O_CO_c1I_c1c_I_c_C_O_NC_C_H_O_CO_c_I_c_C_O_NC_C_H_O_CO_c1I_144',
 'CC_O_N_C_C_H_O_CN_C_C_O_c1c_I_c_C_O_NC_C_H_O_CO_c_I_c_C_O_NC_C_H_O_CO_c1I_c1c_I_c_C_O_NC_C_H_O_CO_c_I_c_C_O_NC_C_H_O_CO_c1I_140',
 'C_CCOCCOCCOCCNc1nc_N2CCN_C_O_C_H_C_H_C_CC_n3cc_C_H_N_CO_nn3_CC2_nc_N2CCN_C_O_C_H_C_H_C_CC_n3cc_C_H_NH3_CO_nn3_CC2_n1_83',
 'C_CCOCCOCCOCCNc1nc_N2CCN_C_O_C_H_C_H_C_CC_n3cc_C_H_N_CO_nn3_CC2_nc_N2CCN_C_O_C_H_C_H_C_CC_n3cc_C_H_NH3_CO_nn3_CC2_n1_84',
 'C_CCOCCOCCOCCNc1nc_N2CCN_C_O_C_H_C_H_C_CC_n3cc_C_H_N_CO_nn3_CC2_nc_N2CCN_C_O_C_H_C_H_C_CC_n3cc_C_H_NH3_CO_nn3_CC2_n1_80',
 'C_CCOCCOCCOCCNc1nc_N2CCN_C_O_C_H_C_H_C_CC_n3cc_C_H_N_CO_nn3_CC2_nc_N2CCN_C_O_C_H_C_H_C_CC_n3cc_C_H_NH3_CO_nn3_CC2_n1_82',
 'C_CCOCCOCCOCCNc1nc_N2CCN_C_O_C_H_C_H_C_CC_n3cc_C_H_N_CO_nn3_CC2_nc_N2CCN_C_O_C_H_C_H_C_CC_n3cc_C_H_NH3_CO_nn3_CC2_n1_81',
 'CC_C_H_1OC_O_C_C_H_O_C_H_C_C_H_O_C_H_2O_C_H_C_C_H_O_C_H_3C_C_C_O_C_H_O_C_H_C_O3_C_H_N_C_C_C_H_2O_C_H_CC_O_C_C_H_C_C_O_C_C_C_C_C_C_H_1CO_C_H_1O_C_H_C_C_H_O_C_H_OC_C_H_1OC_111',
 'CC_C_H_1OC_O_C_C_H_O_C_H_C_C_H_O_C_H_2O_C_H_C_C_H_O_C_H_3C_C_C_O_C_H_O_C_H_C_O3_C_H_N_C_C_C_H_2O_C_H_CC_O_C_C_H_C_C_O_C_C_C_C_C_C_H_1CO_C_H_1O_C_H_C_C_H_O_C_H_OC_C_H_1OC_112',
 'CC_C_H_1OC_O_C_C_H_O_C_H_C_C_H_O_C_H_2O_C_H_C_C_H_O_C_H_3C_C_C_O_C_H_O_C_H_C_O3_C_H_N_C_C_C_H_2O_C_H_CC_O_C_C_H_C_C_O_C_C_C_C_C_C_H_1CO_C_H_1O_C_H_C_C_H_O_C_H_OC_C_H_1OC_110',
 'CC_C_H_1OC_O_C_C_H_O_C_H_C_C_H_O_C_H_2O_C_H_C_C_H_O_C_H_3C_C_C_O_C_H_O_C_H_C_O3_C_H_N_C_C_C_H_2O_C_H_CC_O_C_C_H_C_C_O_C_C_C_C_C_C_H_1CO_C_H_1O_C_H_C_C_H_O_C_H_OC_C_H_1OC_113',
 'CC_C_H_1OC_O_C_C_H_O_C_H_C_C_H_O_C_H_2O_C_H_C_C_H_O_C_H_3C_C_C_O_C_H_O_C_H_C_O3_C_H_N_C_C_C_H_2O_C_H_CC_O_C_C_H_C_C_O_C_C_C_C_C_C_H_1CO_C_H_1O_C_H_C_C_H_O_C_H_OC_C_H_1OC_114',
 'C_CCOCCOCCOCCNc1nc_N2CCN_C_O_C_H_CCCCN_n3cc_C_H_NH3_C_H_C_CC_nn3_CC2_nc_N2CCN_C_O_C_H_CCCCN_n3cc_C_H_N_CO_nn3_CC2_n1_123',
 'C_CCOCCOCCOCCNc1nc_N2CCN_C_O_C_H_CCCCN_n3cc_C_H_NH3_C_H_C_CC_nn3_CC2_nc_N2CCN_C_O_C_H_CCCCN_n3cc_C_H_N_CO_nn3_CC2_n1_124',
 'C_CCOCCOCCOCCNc1nc_N2CCN_C_O_C_H_CCCCN_n3cc_C_H_NH3_C_H_C_CC_nn3_CC2_nc_N2CCN_C_O_C_H_CCCCN_n3cc_C_H_N_CO_nn3_CC2_n1_120',
 'C_CCOCCOCCOCCNc1nc_N2CCN_C_O_C_H_CCCCN_n3cc_C_H_NH3_C_H_C_CC_nn3_CC2_nc_N2CCN_C_O_C_H_CCCCN_n3cc_C_H_N_CO_nn3_CC2_n1_122',
 'C_CCOCCOCCOCCNc1nc_N2CCN_C_O_C_H_CCCCN_n3cc_C_H_NH3_C_H_C_CC_nn3_CC2_nc_N2CCN_C_O_C_H_CCCCN_n3cc_C_H_N_CO_nn3_CC2_n1_121',
 'C_CCOCCOCCOCCNc1nc_N2CCN_C_O_C_H_CCCCN_n3cc_C_H_NH3_C_H_C_CC_nn3_CC2_nc_N2CCN_C_O_C_H_Cc3cc4ccccc4_nH_3_n3cc_C_H_N_CC_C_C_nn3_CC2_n1_16',
 'C_CCOCCOCCOCCNc1nc_N2CCN_C_O_C_H_CCCCN_n3cc_C_H_NH3_C_H_C_CC_nn3_CC2_nc_N2CCN_C_O_C_H_Cc3cc4ccccc4_nH_3_n3cc_C_H_N_CC_C_C_nn3_CC2_n1_18',
 'C_CCOCCOCCOCCNc1nc_N2CCN_C_O_C_H_CCCCN_n3cc_C_H_NH3_C_H_C_CC_nn3_CC2_nc_N2CCN_C_O_C_H_Cc3cc4ccccc4_nH_3_n3cc_C_H_N_CC_C_C_nn3_CC2_n1_15',
 'C_CCOCCOCCOCCNc1nc_N2CCN_C_O_C_H_CCCCN_n3cc_C_H_NH3_C_H_C_CC_nn3_CC2_nc_N2CCN_C_O_C_H_Cc3cc4ccccc4_nH_3_n3cc_C_H_N_CC_C_C_nn3_CC2_n1_17',
 'C_CCOCCOCCOCCNc1nc_N2CCN_C_O_C_H_CCCCN_n3cc_C_H_NH3_C_H_C_CC_nn3_CC2_nc_N2CCN_C_O_C_H_Cc3cc4ccccc4_nH_3_n3cc_C_H_N_CC_C_C_nn3_CC2_n1_19',
 'C_C_H_1O_C_H_O_C_H_2_C_H_O_C_H_3CC_C_4_C_C_H_CC_C_5_C_C_H_4CC_C4_C_H_6CC_C_C_CC_C_6_C_O_O_C_H_6O_C_H_CO_C_H_7O_C_H_CO_C_H_O_C_H_8O_C_H_C_C_H_O_C_H_O_C_H_8O_C_H_O_C_H_7O_C_H_O_C_H_O_C_H_6O_CC_C_45C_C_3_C_CO_OC_C_H_O_C_H_2O_C_H_O_C_H_O_C_H_1O_34',
 'C_C_H_1O_C_H_O_C_H_2_C_H_O_C_H_3CC_C_4_C_C_H_CC_C_5_C_C_H_4CC_C4_C_H_6CC_C_C_CC_C_6_C_O_O_C_H_6O_C_H_CO_C_H_7O_C_H_CO_C_H_O_C_H_8O_C_H_C_C_H_O_C_H_O_C_H_8O_C_H_O_C_H_7O_C_H_O_C_H_O_C_H_6O_CC_C_45C_C_3_C_CO_OC_C_H_O_C_H_2O_C_H_O_C_H_O_C_H_1O_33',
 'C_C_H_1O_C_H_O_C_H_2_C_H_O_C_H_3CC_C_4_C_C_H_CC_C_5_C_C_H_4CC_C4_C_H_6CC_C_C_CC_C_6_C_O_O_C_H_6O_C_H_CO_C_H_7O_C_H_CO_C_H_O_C_H_8O_C_H_C_C_H_O_C_H_O_C_H_8O_C_H_O_C_H_7O_C_H_O_C_H_O_C_H_6O_CC_C_45C_C_3_C_CO_OC_C_H_O_C_H_2O_C_H_O_C_H_O_C_H_1O_30',
 'C_C_H_1O_C_H_O_C_H_2_C_H_O_C_H_3CC_C_4_C_C_H_CC_C_5_C_C_H_4CC_C4_C_H_6CC_C_C_CC_C_6_C_O_O_C_H_6O_C_H_CO_C_H_7O_C_H_CO_C_H_O_C_H_8O_C_H_C_C_H_O_C_H_O_C_H_8O_C_H_O_C_H_7O_C_H_O_C_H_O_C_H_6O_CC_C_45C_C_3_C_CO_OC_C_H_O_C_H_2O_C_H_O_C_H_O_C_H_1O_32',
 'C_C_H_1O_C_H_O_C_H_2_C_H_O_C_H_3CC_C_4_C_C_H_CC_C_5_C_C_H_4CC_C4_C_H_6CC_C_C_CC_C_6_C_O_O_C_H_6O_C_H_CO_C_H_7O_C_H_CO_C_H_O_C_H_8O_C_H_C_C_H_O_C_H_O_C_H_8O_C_H_O_C_H_7O_C_H_O_C_H_O_C_H_6O_CC_C_45C_C_3_C_CO_OC_C_H_O_C_H_2O_C_H_O_C_H_O_C_H_1O_31',
 'COC_O_C_1_Cc2ccc_OC_cc2_C_H_2c3cc_C_O_N4CCCC4_n_CCc4c_nH_c5ccc_O_cc45_c3C_C_H_2CN1C_O_c1ccccc1_28',
 'COC_O_C_1_Cc2ccc_OC_cc2_C_H_2c3cc_C_O_N4CCCC4_n_CCc4c_nH_c5ccc_O_cc45_c3C_C_H_2CN1C_O_c1ccccc1_25',
 'COC_O_C_1_Cc2ccc_OC_cc2_C_H_2c3cc_C_O_N4CCCC4_n_CCc4c_nH_c5ccc_O_cc45_c3C_C_H_2CN1C_O_c1ccccc1_26',
 'COC_O_C_1_Cc2ccc_OC_cc2_C_H_2c3cc_C_O_N4CCCC4_n_CCc4c_nH_c5ccc_O_cc45_c3C_C_H_2CN1C_O_c1ccccc1_29',
 'COC_O_C_1_Cc2ccc_OC_cc2_C_H_2c3cc_C_O_N4CCCC4_n_CCc4c_nH_c5ccc_O_cc45_c3C_C_H_2CN1C_O_c1ccccc1_27',
 'CCCCCC_O_N_C_H_c1ccccc1_C_H_O_C_O_O_C_H_1C_C_2_O_C_H_OC_O_c3ccccc3_C_H_3_C_4_OC_C_O_CO_C_H_4C_C_H_O_C_H_4OC_C_H_O_C_H_O_C_H_4O_C_3_C_C_O_C_H_O_C_C1C_C2_C_C_14',
 'CCCCCC_O_N_C_H_c1ccccc1_C_H_O_C_O_O_C_H_1C_C_2_O_C_H_OC_O_c3ccccc3_C_H_3_C_4_OC_C_O_CO_C_H_4C_C_H_O_C_H_4OC_C_H_O_C_H_O_C_H_4O_C_3_C_C_O_C_H_O_C_C1C_C2_C_C_13',
 'CCCCCC_O_N_C_H_c1ccccc1_C_H_O_C_O_O_C_H_1C_C_2_O_C_H_OC_O_c3ccccc3_C_H_3_C_4_OC_C_O_CO_C_H_4C_C_H_O_C_H_4OC_C_H_O_C_H_O_C_H_4O_C_3_C_C_O_C_H_O_C_C1C_C2_C_C_10',
 'CCCCCC_O_N_C_H_c1ccccc1_C_H_O_C_O_O_C_H_1C_C_2_O_C_H_OC_O_c3ccccc3_C_H_3_C_4_OC_C_O_CO_C_H_4C_C_H_O_C_H_4OC_C_H_O_C_H_O_C_H_4O_C_3_C_C_O_C_H_O_C_C1C_C2_C_C_12',
 'CCCCCC_O_N_C_H_c1ccccc1_C_H_O_C_O_O_C_H_1C_C_2_O_C_H_OC_O_c3ccccc3_C_H_3_C_4_OC_C_O_CO_C_H_4C_C_H_O_C_H_4OC_C_H_O_C_H_O_C_H_4O_C_3_C_C_O_C_H_O_C_C1C_C2_C_C_11',
 'CC_O_O_C_H_1C_C_H_O_C_H_2_C_H_O_C_C_H_O_C_H_3_C_H_O_C_C_H_O_C_H_4CC_C_5_C_C_H_CC_C_H_6_C_H_5C_C_H_O_C_5_C_C_H_C7_CC_O_OC7_CC_C_65O_C4_O_C_H_3C_O_C_H_2C_O_C_H_C_C_H_1O_C_H_1O_C_H_CO_C_H_O_C_H_O_C_H_1O_6',
 'CC_O_O_C_H_1C_C_H_O_C_H_2_C_H_O_C_C_H_O_C_H_3_C_H_O_C_C_H_O_C_H_4CC_C_5_C_C_H_CC_C_H_6_C_H_5C_C_H_O_C_5_C_C_H_C7_CC_O_OC7_CC_C_65O_C4_O_C_H_3C_O_C_H_2C_O_C_H_C_C_H_1O_C_H_1O_C_H_CO_C_H_O_C_H_O_C_H_1O_8',
 'CC_O_O_C_H_1C_C_H_O_C_H_2_C_H_O_C_C_H_O_C_H_3_C_H_O_C_C_H_O_C_H_4CC_C_5_C_C_H_CC_C_H_6_C_H_5C_C_H_O_C_5_C_C_H_C7_CC_O_OC7_CC_C_65O_C4_O_C_H_3C_O_C_H_2C_O_C_H_C_C_H_1O_C_H_1O_C_H_CO_C_H_O_C_H_O_C_H_1O_5',
 'CC_O_O_C_H_1C_C_H_O_C_H_2_C_H_O_C_C_H_O_C_H_3_C_H_O_C_C_H_O_C_H_4CC_C_5_C_C_H_CC_C_H_6_C_H_5C_C_H_O_C_5_C_C_H_C7_CC_O_OC7_CC_C_65O_C4_O_C_H_3C_O_C_H_2C_O_C_H_C_C_H_1O_C_H_1O_C_H_CO_C_H_O_C_H_O_C_H_1O_7',
 'CC_O_O_C_H_1C_C_H_O_C_H_2_C_H_O_C_C_H_O_C_H_3_C_H_O_C_C_H_O_C_H_4CC_C_5_C_C_H_CC_C_H_6_C_H_5C_C_H_O_C_5_C_C_H_C7_CC_O_OC7_CC_C_65O_C4_O_C_H_3C_O_C_H_2C_O_C_H_C_C_H_1O_C_H_1O_C_H_CO_C_H_O_C_H_O_C_H_1O_9'
]

# Common function for extracting the 2D features
def mol_2d(mol, features=None):
    # Ouputs (N_atoms,)
    atomic_number = []
    for atom in mol.GetAtoms():
        atomic_number.append(atom.GetAtomicNum())
    z = torch.tensor(atomic_number, dtype=torch.long)

    # Outputs (2, 2 * M_edges) and (2 * M_edges)
    row, col, edge_type = [], [], []
    for bond in mol.GetBonds():
        start = bond.GetBeginAtomIdx()
        end = bond.GetEndAtomIdx()
        row += [start, end]
        col += [end, start]

        bond_type = bond.GetBondType()
        bond_type_id = BOND_TYPES[bond_type]
        edge_type += [bond_type_id, bond_type_id] 
    edge_index = torch.tensor([row, col], dtype=torch.long)
    edge_type = torch.tensor(edge_type, dtype=torch.long)

    # Get the features (N_atoms, Feature_dim)
    if features:
        features = get_node_features(mol, features)

    # Return the values
    return z, edge_index, edge_type, features


def get_keep_atoms(mol, z):
    mol_no = Chem.RemoveHs(mol)
    important_atoms = list(mol.GetSubstructMatch(mol_no)) 
    one_hot_keep = torch.zeros(z.shape[0], dtype=torch.bool).numpy()
    one_hot_keep[important_atoms] = 1
    keep_idxs = [i for i, a in enumerate(one_hot_keep) if a == 1]
    assert important_atoms == keep_idxs
    return keep_idxs


########################################################
'''
TRAIN/TEST DATASETS
'''
########################################################


class ConformerDataset(Dataset):
    def __init__(
        self, pkl_path, 
        features=['aromatic', 'hybridization', 'partial_charge', 'num_bond_h', 'degree', 'formal_charge', 'ring_size'],
        transforms=["edge_order|2"],
        remove_hs=True,
        filter=True,
        subsample=None  # float (0-1) for fraction, or int for absolute number
    ):
        super().__init__()
        with open(pkl_path, 'rb') as f:
            self.data = pickle.load(f)
        if filter:
            self.data = filter_data(self.data)[0]
        
        # Apply subsampling if specified
        if subsample is not None:
            original_size = len(self.data)
            if isinstance(subsample, float):
                # Fraction of dataset (e.g., 0.1 = 10%)
                num_samples = int(len(self.data) * subsample)
            elif isinstance(subsample, int):
                # Absolute number of samples
                num_samples = min(subsample, len(self.data))
            else:
                raise ValueError(f"subsample must be float (0-1) or int, got {type(subsample)}")
            
            # Randomly subsample
            random.seed(42)  # For reproducibility
            self.data = random.sample(self.data, num_samples)
            print(f"Subsampled dataset: {original_size} -> {len(self.data)} samples ({len(self.data)/original_size:.1%})")
        
        self.features = features
        self.transforms = transforms
        self.remove_hs = remove_hs

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Get the molecule
        mol_data = self.data[idx]
        mol = mol_data['rdmol']

        # Remove Hydrogens if needed
        if self.remove_hs:
            mol = Chem.RemoveHs(mol)

        # Get the atomic numbers and features  Jiaqi: what is node_feature?
        z, edge_index, edge_type, node_features = mol_2d(mol, self.features)

        # Get the coordinates
        coords = torch.tensor(mol.GetConformer(0).GetPositions(), dtype=torch.float32).unsqueeze(-1)        

        data = Data(
            x=z, # [N,]
            pos=coords, # [N, 3, 1] (The 1 represents 1 time-step)
            edge_index=edge_index, # [2, M] 
            edge_attr=edge_type, # [M,]
        )

        if self.features:
            data.x_features = node_features # [N, F]

        # Apply the transforms
        applied_transforms = []
        for transform in self.transforms:
            transform_key = transform.split('|')[0]
            transform_arg = int(transform.split('|')[1])
            applied_transforms.append(TRANSFORMS[transform_key](transform_arg))
        all_transforms = Compose(applied_transforms)

        # Apply the transforms
        data = all_transforms(data)

        return data


class TetrapeptideConformerDataset(Dataset):
    """
    Dataset class for tetrapeptide conformer data with structure:
    {
        'train/AFDV': {  # OR just 'AFDV' if using separate train/val pickle files
            'residues': 'AFDV',
            'smiles': '...',
            'rdkit_mol': <RDKit.Mol>,
            'conformers': [
                {
                    'potential_energy_kj_mol': float,
                    'coordinates': np.array (N, 3)
                },
                ...  # 10 conformers per tetrapeptide
            ]
        },
        ...
    }
    
    Each conformer is treated as a separate training sample.
    So if a tetrapeptide has 10 conformers, it contributes 10 samples to the dataset.
    
    Supports two formats:
    1. Combined file with split prefixes (e.g., 'train/AFDV', 'val/AFDV')
    2. Separate files without prefixes (e.g., just 'AFDV' in train.pkl)
    """
    def __init__(
        self, pkl_path, 
        split='train',  # 'train', 'val', or 'test' - used only if keys have split prefixes
        features=['aromatic', 'hybridization', 'partial_charge', 'num_bond_h', 'degree', 'formal_charge', 'ring_size'],
        transforms=["edge_order|2"],
        remove_hs=True,
        filter=True
    ):
        super().__init__()
        with open(pkl_path, 'rb') as f:
            raw_data = pickle.load(f)
        
        # Check if keys have split prefixes or not
        sample_keys = list(raw_data.keys())[:5] if raw_data else []
        has_split_prefix = any('/' in key for key in sample_keys)
        
        # Filter to the desired split and expand conformers
        self.data = []
        split_prefix = f'{split}/'
        num_tetrapeptides = 0
        total_conformers = 0
        
        for key, value in raw_data.items():
            # If using combined file with split prefixes, filter by split
            # If using separate files, process all entries
            if has_split_prefix and not key.startswith(split_prefix):
                continue
            
            num_tetrapeptides += 1
            num_conformers = len(value['conformers'])
            total_conformers += num_conformers
            
            # Expand each conformer into a separate entry
            # This means each tetrapeptide contributes num_conformers samples
            for conf_idx, conformer in enumerate(value['conformers']):
                self.data.append({
                    'rdkit_mol': value['rdkit_mol'],
                    'coordinates': conformer['coordinates'],
                    'potential_energy_kj_mol': conformer.get('potential_energy_kj_mol', None),
                    'residues': value.get('residues', ''),
                    'smiles': value.get('smiles', ''),
                    'conf_idx': conf_idx,
                    'tetrapeptide_key': key  # Keep track of which tetrapeptide this came from
                })
        
        file_format = "with split prefixes" if has_split_prefix else "without split prefixes (separate file)"
        print(f"Loaded {num_tetrapeptides} tetrapeptides from {split} split ({file_format})")
        print(f"Total conformers (training samples): {total_conformers}")
        print(f"Average conformers per tetrapeptide: {total_conformers / num_tetrapeptides if num_tetrapeptides > 0 else 0:.2f}")
        
        if filter:
            # Apply filter if needed (you may need to adapt filter_data for this format)
            pass  # Skip filtering for now, or adapt filter_data function
        
        self.features = features
        self.transforms = transforms
        self.remove_hs = remove_hs

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Get the molecule data
        mol_data = self.data[idx]
        original_mol = mol_data['rdkit_mol']
        coords_array = mol_data['coordinates']  # Shape: (N, 3) for this conformer

        # Remove Hydrogens if needed
        if self.remove_hs:
            # Suppress RDKit warnings about hydrogen removal
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                mol = Chem.RemoveHs(original_mol)
            # Get atom indices to keep using the same method as TrajectoryDataset
            z_original = mol_2d(original_mol)[0]
            keep_idxs = get_keep_atoms(original_mol, z_original)
            # Filter coordinates to match non-H atoms
            coords = torch.tensor(coords_array[keep_idxs], dtype=torch.float32)
        else:
            mol = original_mol
            coords = torch.tensor(coords_array, dtype=torch.float32)

        # Get the atomic numbers and features
        z, edge_index, edge_type, node_features = mol_2d(mol, self.features)

        # Ensure coordinates match the number of atoms after processing
        assert z.shape[0] == coords.shape[0], f"Mismatch: {z.shape[0]} atoms but {coords.shape[0]} coordinates"
        
        coords = coords.unsqueeze(-1) * 10  # [N, 3, 1] - single conformer # Data is orignally in nm, convert to Å

        data = Data(
            x=z, # [N,]
            pos=coords, # [N, 3, 1] - one conformer per sample
            edge_index=edge_index, # [2, M] 
            edge_attr=edge_type, # [M,]
        )

        if self.features:
            data.x_features = node_features # [N, F]

        # Apply the transforms
        applied_transforms = []
        for transform in self.transforms:
            transform_key = transform.split('|')[0]
            transform_arg = int(transform.split('|')[1])
            applied_transforms.append(TRANSFORMS[transform_key](transform_arg))
        all_transforms = Compose(applied_transforms)

        # Apply the transforms
        data = all_transforms(data)

        return data


class WeightedConcatDataset(Dataset):
    """
    A dataset that concatenates multiple ConformerDataset instances and tracks
    which sub-dataset each sample belongs to for weighted sampling.
    Only used when dataset.type == 'both'.
    """
    def __init__(self, datasets, dataset_names, mixing_ratios):
        """
        Args:
            datasets: List of Dataset objects (e.g., ConformerDataset instances)
            dataset_names: List of strings matching the order of datasets (e.g., ["DRUGS", "QM9"])
            mixing_ratios: Dict mapping dataset names to their mixing ratios (e.g., {"DRUGS": 0.4, "QM9": 0.6})
        """
        super().__init__()
        self.datasets = datasets
        self.dataset_names = dataset_names
        self.mixing_ratios = mixing_ratios
        
        # Track cumulative sizes to map global index to sub-dataset
        self.cumulative_sizes = [0]
        for dataset in self.datasets:
            self.cumulative_sizes.append(self.cumulative_sizes[-1] + len(dataset))
        
        # Store dataset boundaries for quick lookup: (start_idx, end_idx, dataset_name)
        self.dataset_boundaries = []
        for i, dataset in enumerate(self.datasets):
            start_idx = self.cumulative_sizes[i]
            end_idx = self.cumulative_sizes[i + 1]
            self.dataset_boundaries.append((start_idx, end_idx, dataset_names[i]))
        
        # Validate mixing ratios sum to 1.0 (or normalize)
        total_ratio = sum(mixing_ratios.values())
        if abs(total_ratio - 1.0) > 1e-6:
            print(f"Warning: Mixing ratios sum to {total_ratio}, normalizing to 1.0")
            self.mixing_ratios = {k: v / total_ratio for k, v in mixing_ratios.items()}
        
        # Store mixing info as attribute for easy access
        self.has_mixing = True
    
    def __len__(self):
        return self.cumulative_sizes[-1]
    
    def __getitem__(self, idx):
        """
        Get item at global index idx.
        Maps global index to the correct sub-dataset and local index.
        Adds dataset_source attribute to track which dataset the sample came from.
        """
        if idx < 0:
            if -idx > len(self):
                raise ValueError("absolute value of index should not exceed dataset length")
            idx = len(self) + idx
        
        # Find which sub-dataset this index belongs to
        dataset_idx = 0
        local_idx = 0
        dataset_name = None
        for i, (start, end, name) in enumerate(self.dataset_boundaries):
            if start <= idx < end:
                dataset_idx = i
                local_idx = idx - start
                dataset_name = name
                break
        else:
            # Handle edge case: idx == len(self) (shouldn't happen, but be safe)
            dataset_idx = len(self.datasets) - 1
            local_idx = len(self.datasets[-1]) - 1
            dataset_name = self.dataset_names[-1]
        
        data = self.datasets[dataset_idx][local_idx]
        # Add dataset source tracking for validation metrics
        data.dataset_source = dataset_name
        return data
    
    def get_sample_weights(self):
        """
        Compute per-sample weights for WeightedRandomSampler.
        Returns a tensor of weights, one per sample in the concatenated dataset.
        Weight formula: mixing_ratio[dataset_name] / size_of_subdataset
        This ensures samples from smaller datasets get higher individual weights
        to achieve the desired overall mixing ratio.
        """
        weights = []
        for idx in range(len(self)):
            # Find which dataset this sample belongs to
            dataset_name = None
            subdataset_size = 0
            for start, end, name in self.dataset_boundaries:
                if start <= idx < end:
                    dataset_name = name
                    subdataset_size = end - start
                    break
            
            if dataset_name is None:
                # Fallback (shouldn't happen)
                dataset_name = self.dataset_names[-1]
                subdataset_size = len(self.datasets[-1])
            
            # Weight = mixing_ratio / size_of_subdataset
            weight = self.mixing_ratios[dataset_name] / subdataset_size
            weights.append(weight)
        
        # Convert to tensor for WeightedRandomSampler (more efficient and explicit)
        return torch.tensor(weights, dtype=torch.float32)


class TrajectoryDataset(Dataset):
    def __init__(
        self,
        folder_path: str,
        expected_time_dim: int,
        conditioning: Literal["none", "forward", "interpolation"],
        features: list[str] = ["aromatic", "hybridization", "partial_charge", "num_bond_h"],
        transforms: list[str] = ["edge_order|3"],
        remove_hs: bool = True,
        num_frames: int | None = None,
        frame_rate: int | None = None,
        start_frame: int | None = None,
        random_frame: bool = False,
        subsample: float | None = None
    ):
        super().__init__()
        # Collect trajectory directories
        if "DRUGS" in folder_path:
            print("Ignoring traj with over 60 atoms")
            ignore = IGNORE
        else:
            ignore = []
        ignore = set(ignore or [])
        data_dict = {}
        def remove_suffix(s):
            return s.rsplit('_', 1)[0]
        for gen in os.listdir(folder_path):
            gen_dir = os.path.join(folder_path, gen)
            if not os.path.isdir(gen_dir):
                continue
            for smile in os.listdir(gen_dir):
                smile_dir = os.path.join(gen_dir, smile)
                if os.path.isdir(smile_dir) and os.path.exists(os.path.join(smile_dir, 'system.pdb')):
                    smi_name = os.path.basename(smile_dir)
                    if smi_name not in ignore:
                        smile_key = remove_suffix(smile)
                        if smile_key not in data_dict:
                            data_dict[smile_key] = []
                        data_dict[smile_key].append(smile_dir)

        if type(subsample) is float:
            print("Subsampling the molecules based on random index")
            keys = list(data_dict.keys())
            total = len(keys)
            number = int(total * subsample)
            rng = random.Random(0)
            chosen_idxs = rng.sample(range(total), k=number)
            new_keys = [keys[i] for i in chosen_idxs]
            new_dict = {k:data_dict[k] for k in new_keys}
            print(f"New number of molecules {len(new_dict)} vs the old {len(data_dict)}")
            data_dict = new_dict
        elif type(subsample) is str:
            print("Subsampling the molecules based on given path")
            with open(subsample, 'rb') as f:
                new_keys = pickle.load(f)
            new_dict = {k:data_dict[k] for k in new_keys}
            print(f"New number of molecules {len(new_dict)} vs the old {len(data_dict)}")
            data_dict = new_dict

        print(len(data_dict))

        self.data_dirs = []
        for v in data_dict.values():
            self.data_dirs += v

        self.features = features
        self.transforms = transforms
        self.remove_hs = remove_hs
        self.num_frames = num_frames
        self.frame_rate = frame_rate
        self.start_frame = start_frame
        self.random_frame = random_frame
        self.expected_time_dim = expected_time_dim
        self.conditioning = conditioning

        # Sanity check
        if self.num_frames is not None:
            assert self.num_frames <= NUM_FRAMES, (
                f"num_frames ({self.num_frames}) must be <= total frames ({NUM_FRAMES})."
            )

    def __len__(self) -> int:
        return len(self.data_dirs)

    # @aniketh I need to make sure this hydrogen removal is not breaking anything
    def __getitem__(self, idx):
        # Get paths
        data_dir = self.data_dirs[idx]
        xtc_file = os.path.join(data_dir, 'traj.xtc')
        pdb_file = os.path.join(data_dir, 'system.pdb')
        mol_pkl = os.path.join(data_dir, 'mol.pkl')

        # Load trajectory
        traj = md.load(xtc_file, top=pdb_file)

        # Load the full RDKit molecule that corresponds to this trajectory
        mol = pickle.load(open(mol_pkl, 'rb'))

        # Remove Hs if needed
        if self.remove_hs:
            z = mol_2d(mol)[0]
            keep_idxs = get_keep_atoms(mol, z)
            traj = traj.atom_slice(keep_idxs)
            mol = Chem.RemoveHs(mol)

        # Center and align the trajectory (in memory)
        # The trajectories were already prealigned
        traj.center_coordinates()

        # Convert the traj coordinates to Angstroms because MDTraj uses Nm
        traj.xyz *= 10

        # Subsample frames evenly (e.g., every Nth frame)
        if self.random_frame:
            max_start = NUM_FRAMES - self.num_frames
            start = np.random.randint(0, max_start + 1)
        else:
            start = self.start_frame
        end = start + self.num_frames
        traj = traj[start:end:self.frame_rate]

        # Convert coordinates: traj.xyz shape is (T, N, 3)
        coords = torch.tensor(traj.xyz, dtype=torch.float32).permute(1, 2, 0)  # [N, 3, T]

        # Get the atomic numbers and features 
        z, edge_index, edge_type, features = mol_2d(mol, self.features)
        
        # Get the conditioning
        conditioning = torch.zeros(self.expected_time_dim, dtype=torch.bool) # Will contain True for conditioning frames
        if self.conditioning != 'none' and self.conditioning != 'unconditional_forward':
            conditioning[0] = True
        if self.conditioning == 'interpolation':
            conditioning[-1] = True
        denoise_coords = coords[:, :, ~conditioning]

        # Make sure sizes make sense
        assert z.shape[0] == coords.shape[0]
        assert denoise_coords.shape[-1] == self.expected_time_dim - torch.sum(conditioning)

        # Create Data object
        data = Data(
            x=z,  # [N,]
            pos=denoise_coords,  # [N, 3, T - C]
            edge_index=edge_index,  # [2, M] 
            edge_attr=edge_type,  # [M, 1]
            x_features=features,  # [N, F]
            original_frames=coords, # [N, 3, T]
        )

        # Apply the transforms
        applied_transforms = []
        for transform in self.transforms:
            key_arg = transform.split('|')
            if len(key_arg) == 1:
                transform_key = key_arg[0]
                applied_transforms.append(TRANSFORMS[transform_key]())
            else:
                transform_key = key_arg[0]
                transform_arg = int(key_arg[1])
                applied_transforms.append(TRANSFORMS[transform_key](transform_arg))
        all_transforms = Compose(applied_transforms)

        # Apply the transforms
        data = all_transforms(data)

        # Return value
        return data


class TimewarpTrajectoryDataset(Dataset):
    """
    Dataset for Timewarp tetrapeptide trajectories stored as pickles.

    Each pickle (produced by process_timewarp_iterative.py) contains:
        - `coordinates_mdtraj`: mdtraj.Trajectory of heavy-atom positions
        - `rdkit_mol`: heavy-atom RDKit molecule
        - metadata (energies, smiles, etc.)

    We slice every trajectory into fixed-length chunks (`frames_per_example`) and
    then subsample a window of `num_frames` before striding by `frame_rate` to
    obtain the `expected_time_dim` frames (matching DRUGS preprocessing).
    """
    def __init__(
        self,
        folder_path: str,
        expected_time_dim: int,
        conditioning: Literal["none", "forward", "interpolation"],
        features: list[str] = ["aromatic", "hybridization", "partial_charge", "num_bond_h"],
        transforms: list[str] = ["edge_order|3"],
        remove_hs: bool = True,
        num_frames: int | None = None,
        frame_rate: int | None = None,
        start_frame: int | None = None,
        random_frame: bool = False,
        frames_per_example: int = 10000,  # Number of frames per trajectory chunk
        subsample: float | None = None
    ):
        super().__init__()
        
        # Collect pkl files
        pkl_files = []
        for fname in sorted(os.listdir(folder_path)):
            if fname.endswith('.pkl') and not fname.endswith('_errors.pkl'):
                pkl_files.append(os.path.join(folder_path, fname))
        
        if type(subsample) is float:
            print("Subsampling the molecules based on random index")
            total = len(pkl_files)
            number = int(total * subsample)
            rng = random.Random(0)
            chosen_idxs = rng.sample(range(total), k=number)
            pkl_files = [pkl_files[i] for i in chosen_idxs]
            print(f"New number of molecules {len(pkl_files)} vs the old {total}")
        elif type(subsample) is str:
            print("Subsampling the molecules based on given path")
            with open(subsample, 'rb') as f:
                selected_names = pickle.load(f)
            # Filter pkl files to only include selected ones
            selected_set = set(selected_names)
            pkl_files = [f for f in pkl_files if os.path.basename(f).replace('.pkl', '') in selected_set]
            print(f"New number of molecules {len(pkl_files)}")
        
        print(f"Found {len(pkl_files)} pkl files")
        
        # Create example index
        self.examples: list[dict[str, Any]] = []
        for pkl_file in tqdm(pkl_files):
            num_examples = 50000 // frames_per_example
            if num_examples == 0:
                continue

            for i in range(num_examples):
                self.examples.append(
                    {
                        "pkl_file": pkl_file,
                        "chunk_idx": i,
                        "start": i * frames_per_example,
                        "end": (i + 1) * frames_per_example,
                    }
                )
        
        print(f"Created {len(self.examples)} trajectory examples from {len(pkl_files)} molecules")
        
        self.features = features
        self.transforms = transforms
        self.remove_hs = remove_hs
        self.num_frames = num_frames
        self.frame_rate = frame_rate
        self.start_frame = start_frame
        self.random_frame = random_frame
        self.expected_time_dim = expected_time_dim
        self.conditioning = conditioning
        self.frames_per_example = frames_per_example
        
        # Sanity check
        if self.num_frames is not None:
            assert self.num_frames <= frames_per_example, (
                f"num_frames ({self.num_frames}) must be <= frames_per_example ({frames_per_example})."
            )

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx):
        example_info = self.examples[idx]

        with open(example_info['pkl_file'], 'rb') as f:
            data = pickle.load(f)

        traj_md = data.get('coordinates_mdtraj')
        if traj_md is None:
            raise ValueError(f"{example_info['pkl_file']} missing `coordinates_mdtraj`")

        rdkit_mol = data['rdkit_mol']
        chunk_start = example_info['start']
        chunk_end = example_info['end']
        
        chunk_len = chunk_end - chunk_start
        if self.num_frames is None:
            window_size = chunk_len
        else:
            window_size = self.num_frames
            if window_size > chunk_len:
                raise ValueError(
                    f"Requested window ({window_size}) exceeds chunk length ({chunk_len}) "
                    f"for {example_info['pkl_file']} chunk {example_info['chunk_idx']}"
                )
        
        # Determine starting offset inside the chunk
        if self.random_frame:
            max_start = chunk_len - window_size
            start_offset = np.random.randint(0, max_start + 1)
        else:
            default_start = self.start_frame if self.start_frame is not None else 0
            start_offset = min(default_start, max(0, chunk_len - window_size))
        
        # Build absolute frame indices for this chunk
        chunk_indices = np.arange(chunk_start, chunk_end, dtype=np.int64)
        window_indices = chunk_indices[start_offset:start_offset + window_size]
        
        stride = self.frame_rate if self.frame_rate is not None and self.frame_rate > 0 else 1
        stride_indices = window_indices[::stride]
        if stride_indices.size == 0:
            raise ValueError(
                f"No frames selected after striding (stride={stride}) "
                f"for {example_info['pkl_file']} chunk {example_info['chunk_idx']}"
            )
        
        traj_segment = traj_md.slice(stride_indices, copy=True)
        traj_segment.center_coordinates()
        traj_segment.superpose(traj_segment, 0)

        xyz = traj_segment.xyz * 10.0  # nm -> Å
        coords = torch.tensor(xyz, dtype=torch.float32).permute(1, 2, 0).contiguous()

        # Get the atomic numbers and features from RDKit mol
        z, edge_index, edge_type, features = mol_2d(rdkit_mol, self.features)

        # Get the conditioning mask
        conditioning = torch.zeros(self.expected_time_dim, dtype=torch.bool)
        if self.conditioning != 'none' and self.conditioning != 'unconditional_forward':
            conditioning[0] = True
        if self.conditioning == 'interpolation':
            conditioning[-1] = True
        denoise_coords = coords[:, :, ~conditioning]

        # Sanity checks
        assert z.shape[0] == coords.shape[0], f"Atom count mismatch: z={z.shape[0]}, coords={coords.shape[0]}"
        assert denoise_coords.shape[-1] == self.expected_time_dim - torch.sum(conditioning), \
            f"Time dim mismatch: got {denoise_coords.shape[-1]}, expected {self.expected_time_dim - torch.sum(conditioning)}"

        # Extract SMILES from the data if available
        smiles = data.get('smiles', 'UNKNOWN')
        
        # Create Data object
        data_obj = Data(
            x=z,  # [N,]
            pos=denoise_coords,  # [N, 3, T - C]
            edge_index=edge_index,  # [2, M] 
            edge_attr=edge_type,  # [M, 1]
            x_features=features,  # [N, F]
            original_frames=coords,  # [N, 3, T]
            smiles=smiles,  # SMILES string for tracking
            rdmol=rdkit_mol,  # RDKit molecule for evaluation
            conf_idx=example_info['chunk_idx'],  # Chunk index as conformer ID
        )

        # Apply transforms
        applied_transforms = []
        for transform in self.transforms:
            key_arg = transform.split('|')
            if len(key_arg) == 1:
                transform_key = key_arg[0]
                applied_transforms.append(TRANSFORMS[transform_key]())
            else:
                transform_key = key_arg[0]
                transform_arg = int(key_arg[1])
                applied_transforms.append(TRANSFORMS[transform_key](transform_arg))
        all_transforms = Compose(applied_transforms)
        
        # Apply the transforms
        data_obj = all_transforms(data_obj)
        
        return data_obj


class MDGenTrajectoryDataset(Dataset):
    """
    Dataset for MDGen tetrapeptide trajectories.
    
    Structure:
    - Raw data: ${MDGEN_DATA_ROOT}/data/4AA_sims/ITKD/ITKD.xtc + ITKD.pdb
    - RDKit mols: tetrapeptide_conformers_mdgen.pkl contains rdkit_mol for each peptide
    - Splits: CSV files define train/val/test splits
    """
    def __init__(
        self,
        split_csv_path: str,  # Path to CSV (e.g., '~/mdgen/splits/4AA_train.csv')
        expected_time_dim: int,
        conditioning: Literal["none", "forward", "interpolation"],
        features: list[str] = ["aromatic", "hybridization", "partial_charge", "num_bond_h"],
        transforms: list[str] = ["edge_order|3"],
        remove_hs: bool = True,
        num_frames: int | None = None,
        frame_rate: int | None = None,
        start_frame: int | None = None,
        random_frame: bool = False,
    ):
        super().__init__()

        # Data paths
        data_path = os.path.join(os.environ.get('MDGEN_DATA_ROOT', ''), 'data/4AA_sims')
        pickle_path = 'data_gen/tetrapeptide_conformers_mdgen.pkl'
        
        # Load RDKit molecules pickle
        with open(pickle_path, 'rb') as f:
            self.tetrapeptide_confs = pickle.load(f)
        print(f"Loaded {len(self.tetrapeptide_confs)} peptide structures from pickle")

        # Load split CSV to get allowed peptides
        split_df = pd.read_csv(split_csv_path)
        allowed_peptides = set(split_df['name'].str.strip())
        print(f"Split CSV has {len(allowed_peptides)} peptides")

        # Collect valid peptide directories
        self.peptide_dirs = []
        for peptide_name in sorted(os.listdir(data_path)):
            # Check if it's in the split
            if peptide_name not in allowed_peptides:
                continue
            
            peptide_dir = os.path.join(data_path, peptide_name)
            if not os.path.isdir(peptide_dir):
                continue
            
            # Check for required files
            xtc_file = os.path.join(peptide_dir, f'{peptide_name}.xtc')
            pdb_file = os.path.join(peptide_dir, f'{peptide_name}.pdb')
            
            if os.path.exists(xtc_file) and os.path.exists(pdb_file):
                # Also check that we have the rdkit_mol for this peptide
                if peptide_name in self.tetrapeptide_confs:
                    self.peptide_dirs.append((peptide_name, xtc_file, pdb_file))
                    if 'test' in split_csv_path:
                         self.peptide_dirs += [(peptide_name, xtc_file, pdb_file)] * 1
                else:
                    print(f"Warning: {peptide_name} has trajectory but no RDKit mol in pickle")
        
        print(f"Found {len(self.peptide_dirs)} valid peptide trajectories")
        
        # Store parameters
        self.features = features
        self.transforms = transforms
        self.remove_hs = remove_hs
        self.num_frames = num_frames
        self.frame_rate = frame_rate
        self.start_frame = start_frame if start_frame is not None else 0
        self.random_frame = random_frame
        self.expected_time_dim = expected_time_dim
        self.conditioning = conditioning

    def __len__(self) -> int:
        return len(self.peptide_dirs)

    def __getitem__(self, idx):
        peptide_name, xtc_file, pdb_file = self.peptide_dirs[idx]

        # Get the RDKit molecule for this peptide
        rdkit_mol = self.tetrapeptide_confs[peptide_name]['rdkit_mol']
        
        # Get SMILES if available
        smiles = self.tetrapeptide_confs[peptide_name].get('smiles', 'UNKNOWN')

        # Load from preprocessed cache (much faster!)
        peptide_dir = os.path.dirname(xtc_file)
        stride = self.frame_rate if self.frame_rate is not None else 1
        cache_file = os.path.join(peptide_dir, f'{peptide_name}_cache_stride{stride}.npz')
        
        if not os.path.exists(cache_file):
            raise FileNotFoundError(
                f"Cache file not found: {cache_file}\n"
                f"Please run preprocessing first:\n"
                f"  python data_gen/preprocess/process_mdgen.py --stride {stride} --num_workers 32"
            )
        
        # Fast load from cache
        cached = np.load(cache_file)
        xyz_cached = cached['xyz']  # Shape: (T, N, 3) in Angstroms
        total_frames = xyz_cached.shape[0]
        
        # Update rdkit_mol if we're removing Hs (need to do this every time)
        if self.remove_hs:
            rdkit_mol = Chem.RemoveHs(rdkit_mol)

        # Subsample frames from cached data
        # Determine how many frames we want
        if self.num_frames is None:
            num_frames_to_use = total_frames
        else:
            num_frames_to_use = min(self.num_frames, total_frames)
        
        # Select starting point
        if self.random_frame:
            max_start = max(0, total_frames - num_frames_to_use)
            start = np.random.randint(0, max_start + 1)
        else:
            start = self.start_frame
            start = min(start, max(0, total_frames - num_frames_to_use))
        
        end = start + num_frames_to_use
        
        # Slice the cached coordinates
        xyz_slice = xyz_cached[start:end]  # (T, N, 3)

        # Convert coordinates to torch tensor
        coords = torch.tensor(xyz_slice, dtype=torch.float32).permute(1, 2, 0)  # [N, 3, T]

        # Get the atomic numbers and features
        z, edge_index, edge_type, features = mol_2d(rdkit_mol, self.features)

        # Get the conditioning mask
        conditioning = torch.zeros(self.expected_time_dim, dtype=torch.bool)
        if self.conditioning != 'none' and self.conditioning != 'unconditional_forward':
            conditioning[0] = True
        if self.conditioning == 'interpolation':
            conditioning[-1] = True
        denoise_coords = coords[:, :, ~conditioning]

        # Sanity checks
        assert z.shape[0] == coords.shape[0], f"Atom count mismatch: z={z.shape[0]}, coords={coords.shape[0]}"
        assert denoise_coords.shape[-1] == self.expected_time_dim - torch.sum(conditioning), \
            f"Time dim mismatch: got {denoise_coords.shape[-1]}, expected {self.expected_time_dim - torch.sum(conditioning)}"

        # Create Data object
        data = Data(
            x=z,  # [N,]
            pos=denoise_coords,  # [N, 3, T - C]
            edge_index=edge_index,  # [2, M]
            edge_attr=edge_type,  # [M, 1]
            x_features=features,  # [N, F]
            original_frames=coords,  # [N, 3, T]
            smiles=smiles,  # SMILES string for tracking
            rdmol=rdkit_mol,  # RDKit molecule for evaluation
            peptide_name=peptide_name,  # Track which peptide
        )

        # Apply the transforms
        applied_transforms = []
        for transform in self.transforms:
            key_arg = transform.split('|')
            if len(key_arg) == 1:
                transform_key = key_arg[0]
                applied_transforms.append(TRANSFORMS[transform_key]())
            else:
                transform_key = key_arg[0]
                transform_arg = int(key_arg[1])
                applied_transforms.append(TRANSFORMS[transform_key](transform_arg))
        all_transforms = Compose(applied_transforms)

        # Apply the transforms
        data = all_transforms(data)

        return data
    

# Get the dataset
def get_datasets(config):
    if config.dataset.type == 'conformer':
        print("Loading conformer training dataset")
        train_dataset = ConformerDataset(
            pkl_path=config.dataset.train_conf_path, 
            features=config.dataset.features,
            transforms=config.dataset.transforms,
            filter=config.dataset.filter_data,
            remove_hs=config.dataset.remove_hs
        )
        print("Loading conformer validation dataset")
        val_dataset = ConformerDataset(
            pkl_path=config.dataset.val_conf_path, 
            features=config.dataset.features,
            transforms=config.dataset.transforms,
            filter=config.dataset.filter_data,
            remove_hs=config.dataset.remove_hs
        )
    elif config.dataset.type == 'trajectory':
        # Check if this is TIMEWARP dataset (pkl-based), MDGEN dataset, or standard trajectory (xtc-based)
        is_timewarp = hasattr(config.dataset, 'dataset') and config.dataset.dataset == "TIMEWARP"
        is_mdgen = hasattr(config.dataset, 'dataset') and config.dataset.dataset == "MDGEN"
        
        if is_mdgen:
            print("Loading MDGEN trajectory training dataset")
            train_dataset = MDGenTrajectoryDataset(
                split_csv_path=config.dataset.train_split,
                expected_time_dim=config.dataset.expected_time_dim,
                features=config.dataset.features,
                num_frames=config.dataset.num_frames,
                transforms=config.dataset.transforms,
                frame_rate=config.dataset.frame_rate,
                start_frame=config.dataset.start_frame,
                random_frame=config.dataset.random_frame,
                remove_hs=config.dataset.remove_hs,
                conditioning=config.denoiser.conditioning
            )
            print("Loading MDGEN trajectory validation dataset")
            val_dataset = MDGenTrajectoryDataset(
                split_csv_path=config.dataset.val_split,
                expected_time_dim=config.dataset.expected_time_dim,
                features=config.dataset.features,
                num_frames=config.dataset.num_frames,
                transforms=config.dataset.transforms,
                frame_rate=config.dataset.frame_rate,
                start_frame=config.dataset.start_frame,
                random_frame=config.dataset.random_frame,
                remove_hs=config.dataset.remove_hs,
                conditioning=config.denoiser.conditioning
            )
        elif is_timewarp:
            print("Loading TIMEWARP trajectory training dataset")
            # Get frames_per_example from config, default to 10000 for val/test
            # Train might have different total frames, so we calculate based on data
            frames_per_example_train = getattr(config.dataset, 'frames_per_example_train', 10000)
            frames_per_example_val = getattr(config.dataset, 'frames_per_example_val', 10000)
            
            train_dataset = TimewarpTrajectoryDataset(
                folder_path=config.dataset.train_traj_dir,
                expected_time_dim=config.dataset.expected_time_dim,
                features=config.dataset.features,
                num_frames=config.dataset.num_frames,
                transforms=config.dataset.transforms,
                frame_rate=config.dataset.frame_rate,
                start_frame=config.dataset.start_frame,
                random_frame=config.dataset.random_frame,
                remove_hs=config.dataset.remove_hs,
                conditioning=config.denoiser.conditioning,
                frames_per_example=frames_per_example_train,
                subsample=config.dataset.subsample
            )
            print("Loading TIMEWARP trajectory validation dataset")
            val_dataset = TimewarpTrajectoryDataset(
                folder_path=config.dataset.val_traj_dir,
                expected_time_dim=config.dataset.expected_time_dim,
                features=config.dataset.features,
                num_frames=config.dataset.num_frames,
                transforms=config.dataset.transforms,
                frame_rate=config.dataset.frame_rate,
                start_frame=config.dataset.start_frame,
                random_frame=config.dataset.random_frame,
                remove_hs=config.dataset.remove_hs,
                conditioning=config.denoiser.conditioning,
                frames_per_example=frames_per_example_val
            )
        else:
            print("Loading trajectory training dataset")
            train_dataset = TrajectoryDataset(
                folder_path=config.dataset.train_traj_dir, 
                expected_time_dim=config.dataset.expected_time_dim,
                features=config.dataset.features,
                num_frames=config.dataset.num_frames,
                transforms=config.dataset.transforms,
                frame_rate=config.dataset.frame_rate,
                start_frame=config.dataset.start_frame,
                random_frame=config.dataset.random_frame,
                remove_hs=config.dataset.remove_hs,
                conditioning=config.denoiser.conditioning,
                subsample=config.dataset.subsample
            )
            print("Loading trajectory validation dataset")
            val_dataset = TrajectoryDataset(
                folder_path=config.dataset.val_traj_dir, 
                expected_time_dim=config.dataset.expected_time_dim,
                features=config.dataset.features,
                num_frames=config.dataset.num_frames,
                transforms=config.dataset.transforms,
                frame_rate=config.dataset.frame_rate,
                start_frame=config.dataset.start_frame,
                random_frame=config.dataset.random_frame,
                remove_hs=config.dataset.remove_hs,
                conditioning=config.denoiser.conditioning
            )
    elif config.dataset.type == 'both_trajectory':
        # Mix trajectory datasets (DRUGS + QM9)
        # Note: Only 'forward' and 'unconditional_forward' conditioning are supported (no interpolation)
        print("Loading 'both_trajectory' dataset type: combining multiple trajectory datasets")
        
        # Extract mixing ratios from config (default to 0.5/0.5 if not specified)
        if hasattr(config.dataset, 'mixing') and hasattr(config.dataset.mixing, 'ratio'):
            mixing_ratios = config.dataset.mixing.ratio
        else:
            print("Warning: No mixing ratios specified, using default 0.5/0.5 (DRUGS/QM9)")
            mixing_ratios = {"DRUGS": 0.5, "QM9": 0.5}
        
        # Get subsample fractions if specified
        subsample_drugs = getattr(config.dataset, 'subsample_drugs', None)
        subsample_qm9 = getattr(config.dataset, 'subsample_qm9', None)
        
        # Load DRUGS trajectory datasets
        print("Loading DRUGS trajectory training dataset")
        train_dataset_drugs = TrajectoryDataset(
            folder_path=config.dataset.train_traj_dir_drugs,
            expected_time_dim=config.dataset.expected_time_dim,
            features=config.dataset.features,
            num_frames=config.dataset.num_frames,
            transforms=config.dataset.transforms,
            frame_rate=config.dataset.frame_rate,
            start_frame=config.dataset.start_frame,
            random_frame=config.dataset.random_frame,
            remove_hs=config.dataset.remove_hs,
            conditioning=config.denoiser.conditioning,
            subsample=subsample_drugs
        )
        print("Loading DRUGS trajectory validation dataset")
        val_dataset_drugs = TrajectoryDataset(
            folder_path=config.dataset.val_traj_dir_drugs,
            expected_time_dim=config.dataset.expected_time_dim,
            features=config.dataset.features,
            num_frames=config.dataset.num_frames,
            transforms=config.dataset.transforms,
            frame_rate=config.dataset.frame_rate,
            start_frame=config.dataset.start_frame,
            random_frame=config.dataset.random_frame,
            remove_hs=config.dataset.remove_hs,
            conditioning=config.denoiser.conditioning
        )
        
        # Load QM9 trajectory datasets
        print("Loading QM9 trajectory training dataset")
        train_dataset_qm9 = TrajectoryDataset(
            folder_path=config.dataset.train_traj_dir_qm9,
            expected_time_dim=config.dataset.expected_time_dim,
            features=config.dataset.features,
            num_frames=config.dataset.num_frames,
            transforms=config.dataset.transforms,
            frame_rate=config.dataset.frame_rate,
            start_frame=config.dataset.start_frame,
            random_frame=config.dataset.random_frame,
            remove_hs=config.dataset.remove_hs,
            conditioning=config.denoiser.conditioning,
            subsample=subsample_qm9
        )
        print("Loading QM9 trajectory validation dataset")
        val_dataset_qm9 = TrajectoryDataset(
            folder_path=config.dataset.val_traj_dir_qm9,
            expected_time_dim=config.dataset.expected_time_dim,
            features=config.dataset.features,
            num_frames=config.dataset.num_frames,
            transforms=config.dataset.transforms,
            frame_rate=config.dataset.frame_rate,
            start_frame=config.dataset.start_frame,
            random_frame=config.dataset.random_frame,
            remove_hs=config.dataset.remove_hs,
            conditioning=config.denoiser.conditioning
        )
        
        # Create WeightedConcatDataset for train
        # Ensure dataset order matches config.dataset.datasets order
        dataset_order = config.dataset.datasets  # Should be ["DRUGS", "QM9"]
        train_datasets = []
        val_datasets = []
        for dataset_name in dataset_order:
            if dataset_name == "DRUGS":
                train_datasets.append(train_dataset_drugs)
                val_datasets.append(val_dataset_drugs)
            elif dataset_name == "QM9":
                train_datasets.append(train_dataset_qm9)
                val_datasets.append(val_dataset_qm9)
            else:
                raise ValueError(f"Unknown dataset name in config: {dataset_name}")
        
        train_dataset = WeightedConcatDataset(
            datasets=train_datasets,
            dataset_names=dataset_order,
            mixing_ratios=mixing_ratios
        )
        
        # Print informative messages
        print(f"\n=== Trajectory Dataset Mixing Summary ===")
        print(f"DRUGS train size: {len(train_dataset_drugs)}")
        print(f"QM9 train size: {len(train_dataset_qm9)}")
        print(f"Combined train size: {len(train_dataset)}")
        print(f"DRUGS val size: {len(val_dataset_drugs)}")
        print(f"QM9 val size: {len(val_dataset_qm9)}")
        print(f"Mixing ratios: {mixing_ratios}")
        print(f"Expected time dim: {config.dataset.expected_time_dim}")
        print(f"Note: Validation will run separately on DRUGS and QM9")
        print(f"==========================================\n")
        
        # Return train_dataset and separate validation datasets
        return train_dataset, val_dataset_drugs, val_dataset_qm9
    elif config.dataset.type == 'both':
        # Only used when dataset.type == 'both'
        print("Loading 'both' dataset type: combining multiple conformer datasets")
        
        # Extract mixing ratios from config (default to 0.4/0.6 if not specified)
        if hasattr(config.dataset, 'mixing') and hasattr(config.dataset.mixing, 'ratio'):
            mixing_ratios = config.dataset.mixing.ratio
        else:
            print("Warning: No mixing ratios specified, using default 0.4/0.6 (DRUGS/QM9)")
            mixing_ratios = {"DRUGS": 0.4, "QM9": 0.6}
        
        # Load DRUGS datasets
        print("Loading DRUGS conformer training dataset")
        train_dataset_drugs = ConformerDataset(
            pkl_path=config.dataset.train_conf_path_drugs,
            features=config.dataset.features,
            transforms=config.dataset.transforms,
            filter=config.dataset.filter_data,
            remove_hs=config.dataset.remove_hs
        )
        print("Loading DRUGS conformer validation dataset")
        val_dataset_drugs = ConformerDataset(
            pkl_path=config.dataset.val_conf_path_drugs,
            features=config.dataset.features,
            transforms=config.dataset.transforms,
            filter=config.dataset.filter_data,
            remove_hs=config.dataset.remove_hs
        )
        
        # Load QM9 datasets
        print("Loading QM9 conformer training dataset")
        train_dataset_qm9 = ConformerDataset(
            pkl_path=config.dataset.train_conf_path_qm9,
            features=config.dataset.features,
            transforms=config.dataset.transforms,
            filter=config.dataset.filter_data,
            remove_hs=config.dataset.remove_hs
        )
        print("Loading QM9 conformer validation dataset")
        val_dataset_qm9 = ConformerDataset(
            pkl_path=config.dataset.val_conf_path_qm9,
            features=config.dataset.features,
            transforms=config.dataset.transforms,
            filter=config.dataset.filter_data,
            remove_hs=config.dataset.remove_hs
        )
        
        # Create WeightedConcatDataset for train
        # Ensure dataset order matches config.dataset.datasets order
        dataset_order = config.dataset.datasets  # Should be ["DRUGS", "QM9"]
        train_datasets = []
        val_datasets = []
        for dataset_name in dataset_order:
            if dataset_name == "DRUGS":
                train_datasets.append(train_dataset_drugs)
                val_datasets.append(val_dataset_drugs)
            elif dataset_name == "QM9":
                train_datasets.append(train_dataset_qm9)
                val_datasets.append(val_dataset_qm9)
            else:
                raise ValueError(f"Unknown dataset name in config: {dataset_name}")
        
        train_dataset = WeightedConcatDataset(
            datasets=train_datasets,
            dataset_names=dataset_order,
            mixing_ratios=mixing_ratios
        )
        
        # For 'both' type, return separate validation datasets for independent tracking
        # Don't create WeightedConcatDataset for validation - we want separate dataloaders
        
        # Print informative messages
        print(f"\n=== Dataset Mixing Summary ===")
        print(f"DRUGS train size: {len(train_dataset_drugs)}")
        print(f"QM9 train size: {len(train_dataset_qm9)}")
        print(f"Combined train size: {len(train_dataset)}")
        print(f"DRUGS val size: {len(val_dataset_drugs)}")
        print(f"QM9 val size: {len(val_dataset_qm9)}")
        print(f"Mixing ratios: {mixing_ratios}")
        print(f"Note: Validation will run separately on DRUGS and QM9")
        print(f"==============================\n")
        
        # Return train_dataset and separate validation datasets
        return train_dataset, val_dataset_drugs, val_dataset_qm9
    elif config.dataset.type == 'tetrapeptide':
        # Tetrapeptide only (no mixing)
        print("Loading tetrapeptide conformer training dataset")
        train_dataset = TetrapeptideConformerDataset(
            pkl_path=config.dataset.train_conf_path,
            split='train',
            features=config.dataset.features,
            transforms=config.dataset.transforms,
            filter=config.dataset.filter_data,
            remove_hs=config.dataset.remove_hs
        )
        print("Loading tetrapeptide conformer validation dataset")
        val_dataset = TetrapeptideConformerDataset(
            pkl_path=config.dataset.val_conf_path,
            split='val',
            features=config.dataset.features,
            transforms=config.dataset.transforms,
            filter=config.dataset.filter_data,
            remove_hs=config.dataset.remove_hs
        )
        return train_dataset, val_dataset
    elif config.dataset.type == 'tetrapeptide_drugs':
        # Mix tetrapeptide and DRUGS datasets
        print("Loading 'tetrapeptide_drugs' dataset type: combining tetrapeptide and DRUGS conformer datasets")
        
        # Extract mixing ratios from config
        if hasattr(config.dataset, 'mixing') and hasattr(config.dataset.mixing, 'ratio'):
            mixing_ratios = config.dataset.mixing.ratio
        else:
            mixing_ratios = {"DRUGS": 0.3, "TETRAPEPTIDE": 0.7}
            print("Warning: No mixing ratios specified, using default 0.3/0.7 (DRUGS/tetrapeptide)")
        
        # Get DRUGS subsample fraction if specified
        drugs_subsample = None
        if hasattr(config.dataset, 'drugs_subsample'):
            drugs_subsample = config.dataset.drugs_subsample
            print(f"Will subsample DRUGS dataset to {drugs_subsample}")
        
        # Load DRUGS datasets (with optional subsampling)
        print("Loading DRUGS conformer training dataset")
        train_dataset_drugs = ConformerDataset(
            pkl_path=config.dataset.train_conf_path_drugs,
            features=config.dataset.features,
            transforms=config.dataset.transforms,
            filter=config.dataset.filter_data,
            remove_hs=config.dataset.remove_hs,
            subsample=drugs_subsample
        )
        print("Loading DRUGS conformer validation dataset")
        val_dataset_drugs = ConformerDataset(
            pkl_path=config.dataset.val_conf_path_drugs,
            features=config.dataset.features,
            transforms=config.dataset.transforms,
            filter=config.dataset.filter_data,
            remove_hs=config.dataset.remove_hs
        )
        
        # Load tetrapeptide datasets
        print("Loading tetrapeptide conformer training dataset")
        train_dataset_tetrapeptide = TetrapeptideConformerDataset(
            pkl_path=config.dataset.train_conf_path_tetrapeptide,
            split='train',
            features=config.dataset.features,
            transforms=config.dataset.transforms,
            filter=config.dataset.filter_data,
            remove_hs=config.dataset.remove_hs
        )
        print("Loading tetrapeptide conformer validation dataset")
        val_dataset_tetrapeptide = TetrapeptideConformerDataset(
            pkl_path=config.dataset.val_conf_path_tetrapeptide,
            split='val',
            features=config.dataset.features,
            transforms=config.dataset.transforms,
            filter=config.dataset.filter_data,
            remove_hs=config.dataset.remove_hs
        )
        
        # Create WeightedConcatDataset for train
        train_datasets = [train_dataset_drugs, train_dataset_tetrapeptide]
        dataset_order = ["DRUGS", "TETRAPEPTIDE"]
        
        train_dataset = WeightedConcatDataset(
            datasets=train_datasets,
            dataset_names=dataset_order,
            mixing_ratios=mixing_ratios
        )
        
        # Print summary
        print(f"\n=== Dataset Mixing Summary ===")
        print(f"DRUGS train size: {len(train_dataset_drugs)}")
        print(f"Tetrapeptide train size: {len(train_dataset_tetrapeptide)}")
        print(f"Combined train size: {len(train_dataset)}")
        print(f"DRUGS val size: {len(val_dataset_drugs)}")
        print(f"Tetrapeptide val size: {len(val_dataset_tetrapeptide)}")
        print(f"Mixing ratios: {mixing_ratios}")
        print(f"==============================\n")
        
        return train_dataset, val_dataset_drugs, val_dataset_tetrapeptide
    else:
        raise ValueError(f"Invalid dataset type: {config.dataset.type}")
    
    # For 'conformer' and 'trajectory' types, return standard (train, val) tuple
    return train_dataset, val_dataset


########################################################
'''
TEST DATASETS
'''
########################################################


class ConformerDatasetTest(Dataset):
    def __init__(
        self, pkl_path, 
        ratio=2,
        subsample=None,
        features=["aromatic", "hybridization", "partial_charge", "num_bond_h"],
        transforms=["edge_order|2"],
        remove_hs=True,
        filter=False
    ):
        super().__init__()
        with open(pkl_path, 'rb') as f:
            self.data = pickle.load(f)
        if filter:
            self.data = filter_data(self.data)[0]

        # Get all smiles and subsample if needed
        all_smiles_dict = get_smiles_dict(self.data)
        
        if subsample is not None:
            smiles_keys = {k: all_smiles_dict[k] for k in subsample}
        else:
            smiles_keys = all_smiles_dict

        # Get the smiles keys
        # Now the data should be in the ratio of the number of inferences we need.
        self.data = []
        for smiles, (item, count) in smiles_keys.items():
            for i in range(count * ratio):
                self.data.append((smiles, item, i))
        
        self.features = features
        self.transforms = transforms
        self.remove_hs = remove_hs

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Get the molecule
        data_item = self.data[idx]
        smiles, mol_data, i = data_item
        mol = mol_data['rdmol']

        # Remove Hydrogens if needed
        if self.remove_hs:
            mol = Chem.RemoveHs(mol)

        # Get the atomic numbers and features  Jiaqi: what is node_feature?
        z, edge_index, edge_type, node_features = mol_2d(mol, self.features)

        # Get the coordinates
        coords = torch.tensor(mol.GetConformer(0).GetPositions(), dtype=torch.float32).unsqueeze(-1)

        data = Data(
            x=z, # [N,]
            pos=coords, # [N, 3, 1] (The 1 represents 1 time-step)
            edge_index=edge_index, # [2, M] 
            edge_attr=edge_type, # [M,],
            smiles=smiles,
            rdmol=mol,
            conf_idx=i,
        )

        if self.features:
            data.x_features = node_features # [N, F]

        # Apply the transforms
        applied_transforms = []
        for transform in self.transforms:
            transform_key = transform.split('|')[0]
            transform_arg = int(transform.split('|')[1])
            applied_transforms.append(TRANSFORMS[transform_key](transform_arg))
        all_transforms = Compose(applied_transforms)

        # Apply the transforms
        data = all_transforms(data)

        # Return value
        return data



class TrajectoryDatasetTestInterpolation(Dataset):
    def __init__(
        self, 
        folder_path: str,
        pkl_path: str,
        expected_time_dim: int,
        subsample = None,
        features: list[str] = ["aromatic", "hybridization", "partial_charge", "num_bond_h"],
        transforms: list[str] = ["edge_order|2"],
        remove_hs: bool = True,
        ratio = 1,
    ):
        super().__init__()

        start_frame = 0
        random_frame = False

        # Collect trajectory directories
        self.data = []
        self.mols = []
        self.features = features
        self.transforms = transforms
        self.remove_hs = remove_hs
        self.expected_time_dim = expected_time_dim

        # Iterate directories lazily with scandir
        xtc_data = {}
        for batch_entry in tqdm(os.scandir(folder_path), desc='Processing batches'):
            if not batch_entry.is_dir():
                continue
            for mol_entry in tqdm(os.scandir(batch_entry.path), desc=f'Processing molecules in {batch_entry.name}', leave=False):
                if not mol_entry.is_dir():
                    continue
                base_path = mol_entry.path
                pdb_path = os.path.join(base_path, 'system.pdb')
                if not os.path.isfile(pdb_path):
                    continue

                # Read SMILES first (cheap)
                smiles_file = os.path.join(base_path, 'smiles.txt')
                try:
                    with open(smiles_file, 'r') as f:
                        smiles = f.readline().strip()
                except IOError:
                    continue

                # Load molecule and trajectory
                xtc_file = os.path.join(base_path, 'traj.xtc')
    
                # Track count per SMILES
                if smiles not in xtc_data:
                    xtc_data[smiles] = (xtc_file, pdb_path)

        # Open the frame pickle
        with open(pkl_path, 'rb') as f:
            traj_data = pickle.load(f)

        # Get only the molecules that have the frame data
        keys = [smiles for smiles in traj_data if len(traj_data[smiles]) == 3]
        print("Number of Molecules with Frame Data: ", len(keys))
        if type(subsample) is float:
            total = len(keys)
            number = int(total * subsample)
            rng = random.Random(0)
            chosen_idxs = rng.sample(range(total), k=number)
            new_keys = [keys[i] for i in chosen_idxs]
            keys = new_keys
        elif type(subsample) is str:
            with open(subsample, 'rb') as f:
                new_keys = set(pickle.load(f))
            keys = set(keys) & new_keys
            print(keys)
            keys = sorted(keys)
            total = len(keys)
            number = total
            rng = random.Random(0)
            chosen_idxs = rng.sample(range(total), k=number)
            new_keys = [keys[i] for i in chosen_idxs]
            keys = new_keys

        new_dict = {k:traj_data[k] for k in keys}
        traj_data = new_dict
        print("Number of molecules after we have sampled: ", len(traj_data))

        # Build the final index with repetition ratio
        self.data = []
        for smiles, data in traj_data.items():
            mol, start_frames, end_frames = data['rdmol'], data['start_frames'], data['end_frames']
            traj_things = xtc_data[smiles]
            assert start_frames.shape[0] == end_frames.shape[0]
            assert start_frames.shape[0] == 1000
            for i in range(int(1000 * ratio)):
                self.data.append((smiles, mol, start_frames[i], end_frames[i], traj_things, i))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Get paths
        smiles, mol, start_frame, end_frame, (xtc_file, pdb_path), i = self.data[idx]

        # Align
        traj = md.load(xtc_file, top=pdb_path)
        traj.xyz[1, ...] = start_frame.numpy() / 10 # Inject as nm
        traj.xyz[2, ...] = end_frame.numpy() / 10 # Inject as nm
        traj.superpose(traj, frame=0)

        # Remove Hs if needed
        if self.remove_hs:
            z = mol_2d(mol)[0]
            keep_idxs = get_keep_atoms(mol, z)
            traj = traj.atom_slice(keep_idxs)
            mol = Chem.RemoveHs(mol)

        # Center and align the trajectory (in memory)
        traj.center_coordinates()
        start_frame = torch.tensor(traj.xyz[1, ...] * 10)
        end_frame = torch.tensor(traj.xyz[2, ...] * 10)
       
        # Convert coordinates: traj.xyz shape is (N, 3, T)
        N = start_frame.shape[0]
        coords = torch.zeros((N, 3, self.expected_time_dim))
        coords[..., 0] = start_frame
        coords[..., -1] = end_frame

        # Handle graph featurization
        z, edge_index, edge_type, features = mol_2d(mol, self.features)

        # Get the conditioning
        conditioning = torch.zeros(self.expected_time_dim, dtype=torch.bool) # Will contain True for conditioning frames
        conditioning[0] = True
        conditioning[-1] = True
        denoise_coords = coords[:, :, ~conditioning]

        # Make sure sizes make sense
        assert z.shape[0] == coords.shape[0]
        assert denoise_coords.shape[-1] == self.expected_time_dim - torch.sum(conditioning)

        # Create Data object
        data = Data(
            x=z,  # [N,]
            pos=denoise_coords,  # [N, 3, T - C]
            edge_index=edge_index,  # [2, M] 
            edge_attr=edge_type,  # [M, 1]
            x_features=features,  # [N, F]
            original_frames=coords, # [N, 3, T]
            smiles=smiles,
            rdmol=mol,
            conf_idx=i,
        )

        # Apply the transforms
        applied_transforms = []
        for transform in self.transforms:
            key_arg = transform.split('|')
            if len(key_arg) == 1:
                transform_key = key_arg[0]
                applied_transforms.append(TRANSFORMS[transform_key]())
            else:
                transform_key = key_arg[0]
                transform_arg = int(key_arg[1])
                applied_transforms.append(TRANSFORMS[transform_key](transform_arg))
        all_transforms = Compose(applied_transforms)

        # Apply the transforms
        data = all_transforms(data)

        # Return value
        return data

class TrajectoryDatasetTestUncond(Dataset):
    """
    In-memory unconditional dataset (zero disk I/O in __getitem__):
      • Scans the folder structure once (same layout as your Forward dataset)
      • Loads mol + traj once per entry, optional H removal, centers, nm→Å
      • Slices/pads/crops to expected_time_dim and precomputes graph tensors
      • Applies transforms during init; __getitem__ just returns cached Data
    """

    def __init__(
        self,
        folder_path: str,
        expected_time_dim: int,
        features: list[str] = ["aromatic", "hybridization", "partial_charge", "num_bond_h"],
        transforms: list[str] = ["edge_order|2"],
        remove_hs: bool = True,
        num_frames: int | None = None,     # contiguous window before frame_rate stride; defaults to all
        frame_rate: int | None = None,     # stride (≥1), default 1
        start_frame: int = 0,              # deterministic start (uncond)
        random_frame: bool = False,        # keep hook; default False to mirror Forward
        subsample: float | str | None = None,  # float fraction or path to smiles list (pickle)
        replicate_by_count: bool = False,  # if True, replicate per-SMILES by file count (like 'ratio' idea)
    ):
        super().__init__()

        # ---- Config / attrs
        self.features = features
        self.transforms_cfg = transforms
        self.remove_hs = remove_hs
        self.expected_time_dim = int(expected_time_dim)
        self.num_frames = num_frames
        self.frame_rate = 1 if frame_rate is None else max(1, int(frame_rate))
        self.start_frame = max(0, int(start_frame))
        self.random_frame = bool(random_frame)
        self.replicate_by_count = bool(replicate_by_count)

        # ---- Discover raw items grouped by SMILES (collect all reps)
        # traj_map[smiles] = list of (mol.pkl, traj.xtc, system.pdb)
        traj_map: dict[str, list[tuple[str, str, str]]] = {}

        for batch_entry in tqdm(os.scandir(folder_path), desc='Processing batches'):
            if not batch_entry.is_dir():
                continue
            for mol_entry in tqdm(os.scandir(batch_entry.path),
                                  desc=f'Processing molecules in {batch_entry.name}',
                                  leave=False):
                if not mol_entry.is_dir():
                    continue
                base_path = mol_entry.path
                pdb_path = os.path.join(base_path, 'system.pdb')
                mol_file = os.path.join(base_path, 'mol.pkl')
                xtc_file = os.path.join(base_path, 'traj.xtc')
                smiles_file = os.path.join(base_path, 'smiles.txt')

                if not (os.path.isfile(pdb_path) and os.path.isfile(mol_file) and os.path.isfile(xtc_file)):
                    continue

                try:
                    with open(smiles_file, 'r') as f:
                        smiles = f.readline().strip()
                except Exception:
                    continue

                traj_map.setdefault(smiles, []).append((mol_file, xtc_file, pdb_path))

        # ---- Optional subsampling (float fraction or path to pickled list of smiles)
        if isinstance(subsample, float):
            print("Subsampling the molecules (fraction).")
            keys = list(traj_map.keys())
            total = len(keys)
            number = max(1, int(total * subsample))
            rng = random.Random(0)
            chosen = set(rng.sample(range(total), k=number))
            new_map = {k: traj_map[k] for i, k in enumerate(keys) if i in chosen}
            print(f"New number of molecules {len(new_map)} (was {total})")
            traj_map = new_map
        elif isinstance(subsample, str):
            print("Subsampling the molecules using provided path")
            with open(subsample, 'rb') as f:
                smiles_keep = set(pickle.load(f))
            old_n = len(traj_map)
            traj_map = {k: traj_map[k] for k in list(traj_map.keys()) if k in smiles_keep}
            print(f"New number of molecules {len(traj_map)} (was {old_n})")

        # ---- Prepare transform pipeline (apply once during init)
        applied_transforms = []
        for spec in self.transforms_cfg:
            parts = spec.split('|')
            if len(parts) == 1:
                applied_transforms.append(TRANSFORMS[parts[0]]())
            else:
                applied_transforms.append(TRANSFORMS[parts[0]](int(parts[1])))
        all_transforms = Compose(applied_transforms) if applied_transforms else None

        # ---- Preload & preprocess everything into memory
        self.items: list[Data] = []
        print(f"Loading trajectories into memory for {len(traj_map)} unique SMILES...")
        for smiles, entries in tqdm(traj_map.items(), desc="Preloading"):
            # Decide how many replicas to emit for this SMILES
            reps = len(entries) if self.replicate_by_count else 1
            for rep_idx in range(reps):
                # Round-robin over available entries if replicating
                mol_file, xtc_file, pdb_path = entries[rep_idx % len(entries)]

                # Load molecule and trajectory
                try:
                    with open(mol_file, 'rb') as f:
                        mol = pickle.load(f)
                except Exception as e:
                    print(f"[WARN] Failed to load mol: {mol_file} ({e})")
                    continue
                try:
                    traj = md.load(xtc_file, top=pdb_path)
                except Exception as e:
                    print(f"[WARN] Failed to load traj: {xtc_file} ({e})")
                    continue

                # Optional H removal (keep RDKit & trajectory aligned)
                if self.remove_hs:
                    try:
                        z_all = mol_2d(mol)[0]
                        keep_idxs = get_keep_atoms(mol, z_all)
                        traj = traj.atom_slice(keep_idxs)
                        mol = Chem.RemoveHs(mol)
                    except Exception as e:
                        print(f"[WARN] H-removal/slicing failed for {smiles}: {e}")
                        continue

                # Center coordinates; convert nm→Å
                traj.center_coordinates()
                traj.xyz *= 10.0  # (T, N, 3)

                # Frame selection then stride
                T = traj.n_frames
                num_frames = T if self.num_frames is None else int(self.num_frames)
                if num_frames <= 0:
                    print(f"[WARN] num_frames={num_frames} invalid; skipping {smiles}")
                    continue

                if self.random_frame:
                    max_start = max(0, T - num_frames)
                    start = int(np.random.randint(0, max_start + 1))
                else:
                    start = min(self.start_frame, max(0, T - num_frames))
                end = min(T, start + num_frames)

                traj = traj[start:end:self.frame_rate]  # stride by frame_rate
                # After stride, ensure we hit expected_time_dim by pad/crop
                if traj.n_frames != self.expected_time_dim:
                    if traj.n_frames > self.expected_time_dim:
                        traj = traj[:self.expected_time_dim]
                    else:
                        pad_needed = self.expected_time_dim - traj.n_frames
                        last = traj[-1:].xyz.copy()
                        pad_xyz = np.repeat(last, pad_needed, axis=0)
                        new_xyz = np.concatenate([traj.xyz, pad_xyz], axis=0)
                        traj = md.Trajectory(new_xyz, traj.topology)

                # Convert to torch [N,3,T]
                coords = torch.tensor(traj.xyz, dtype=torch.float32).permute(1, 2, 0).contiguous()

                # Graph featurization AFTER H removal
                try:
                    z, edge_index, edge_type, x_features = mol_2d(mol, self.features)
                except Exception as e:
                    print(f"[WARN] mol_2d failed for {smiles}: {e}")
                    continue

                # Conditioning (unconditional): all False
                cond_mask = torch.zeros(self.expected_time_dim, dtype=torch.bool)
                denoise_coords = coords[:, :, ~cond_mask]

                # Sanity checks
                if z.shape[0] != coords.shape[0]:
                    print(f"[WARN] Atom count mismatch {smiles}: z={z.shape[0]} vs coords={coords.shape[0]}")
                    continue
                if denoise_coords.shape[-1] != self.expected_time_dim:
                    print(f"[WARN] Time dim mismatch {smiles}: {denoise_coords.shape[-1]} vs {self.expected_time_dim}")
                    continue

                data = Data(
                    x=z,                          # [N]
                    pos=denoise_coords,           # [N,3,T]
                    edge_index=edge_index,        # [2,M]
                    edge_attr=edge_type,          # [M,*]
                    x_features=x_features,        # [N,F]
                    original_frames=coords,       # [N,3,T]
                    smiles=smiles,
                    rdmol=mol,
                    conf_idx=rep_idx,
                )

                # Apply transforms now (precompute), so __getitem__ is zero work
                if all_transforms is not None:
                    try:
                        data = all_transforms(data)
                    except Exception as e:
                        print(f"[WARN] Transform failed for {smiles} (rep {rep_idx}): {e}")
                        continue

                self.items.append(data)

        if len(self.items) == 0:
            print("[WARN] No items loaded into memory for TrajectoryDatasetTestUncond.")

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Data:
        # Zero I/O: return precomputed Data
        return self.items[idx]

class TrajectoryDatasetTestForward(Dataset):
    def __init__(
        self, 
        folder_path: str,
        expected_time_dim: int,
        conditioning: Literal["none", "forward", "interpolation"],
        features: list[str] = ["aromatic", "hybridization", "partial_charge", "num_bond_h"],
        transforms: list[str] = ["edge_order|2"],
        remove_hs: bool = True,
        num_frames: int | None = None,
        frame_rate: int | None = None,
        subsample = None,
        num_reps: int | None = None,
    ):
        super().__init__()

        # ---- Config / attrs
        self.features = features
        self.transforms = transforms
        self.num_frames = num_frames
        self.frame_rate = frame_rate
        self.remove_hs = remove_hs
        self.expected_time_dim = expected_time_dim
        self.conditioning = conditioning
        self.num_reps = num_reps

        # For this version we keep deterministic slicing
        start_frame = 0
        random_frame = False
        time_test = False

        # ---- Discover raw items grouped by SMILES
        traj_map: dict[str, list[tuple[str, str, str]]] = {}

        for batch_entry in tqdm(os.scandir(folder_path), desc='Processing batches'):
            if not batch_entry.is_dir():
                continue
            for mol_entry in tqdm(os.scandir(batch_entry.path),
                                  desc=f'Processing molecules in {batch_entry.name}',
                                  leave=False):
                if not mol_entry.is_dir():
                    continue
                base_path = mol_entry.path
                pdb_path = os.path.join(base_path, 'system.pdb')
                if not os.path.isfile(pdb_path):
                    continue

                smiles_file = os.path.join(base_path, 'smiles.txt')
                try:
                    with open(smiles_file, 'r') as f:
                        smiles = f.readline().strip()
                except IOError:
                    continue

                mol_file = os.path.join(base_path, 'mol.pkl')
                xtc_file = os.path.join(base_path, 'traj.xtc')

                traj_map.setdefault(smiles, []).append((mol_file, xtc_file, pdb_path))

        # ---- Optional subsampling
        if isinstance(subsample, float):
            print("Subsampling the molecules")
            keys = list(traj_map.keys())
            total = len(keys)
            number = max(1, int(total * subsample))
            rng = random.Random(0)
            chosen_idxs = rng.sample(range(total), k=number)
            new_keys = [keys[i] for i in chosen_idxs]
            traj_map = {k: traj_map[k] for k in new_keys}
            print(f"New number of molecules {len(traj_map)} (was {total})")

        elif isinstance(subsample, str):
            print("Subsampling the molecules using provided path")
            with open(subsample, 'rb') as f:
                new_keys = pickle.load(f)
            old_n = len(traj_map)
            traj_map = {k: traj_map[k] for k in new_keys if k in traj_map}
            print(f"New number of molecules {len(traj_map)} (was {old_n})")
            if 'time_test' in subsample:
                time_test = True

        # ---- Prepare transform pipeline (applied once during init)
        applied_transforms = []
        for transform in self.transforms:
            key_arg = transform.split('|')
            if len(key_arg) == 1:
                transform_key = key_arg[0]
                applied_transforms.append(TRANSFORMS[transform_key]())
            else:
                transform_key = key_arg[0]
                transform_arg = int(key_arg[1])
                applied_transforms.append(TRANSFORMS[transform_key](transform_arg))
        all_transforms = Compose(applied_transforms)

        # ---- Preload & preprocess everything into memory
        self.items: list[Data] = []
        seen_for_time_test = set()

        outer_iter = (
            (smiles, entries)
            for smiles, entries in traj_map.items()
        )

        for smiles, entries in tqdm(list(outer_iter), desc="Loading trajectories into memory"):
            # Limit number of reps per molecule if specified
            if self.num_reps is not None:
                entries = entries[:self.num_reps]
            
            for rep_idx, (mol_file, xtc_file, pdb_path) in enumerate(entries):
                # time_test: keep only one rep per SMILES
                if time_test and smiles in seen_for_time_test:
                    continue

                # Load molecule and trajectory (disk I/O happens only here)
                try:
                    with open(mol_file, 'rb') as f:
                        mol = pickle.load(f)
                except Exception as e:
                    # Skip corrupted entries
                    print(f"[WARN] Failed to load mol: {mol_file} ({e})")
                    continue

                try:
                    traj = md.load(xtc_file, top=pdb_path)
                except Exception as e:
                    print(f"[WARN] Failed to load traj: {xtc_file} ({e})")
                    continue

                # Remove Hs (keeping RDKit & trajectory aligned)
                if self.remove_hs:
                    z_all = mol_2d(mol)[0]
                    keep_idxs = get_keep_atoms(mol, z_all)
                    try:
                        traj = traj.atom_slice(keep_idxs)
                        mol = Chem.RemoveHs(mol)
                    except Exception as e:
                        print(f"[WARN] Failed H removal/slice for {smiles}: {e}")
                        continue

                # Center (they are prealigned; this is cheap)
                traj.center_coordinates()

                # Frame slicing: pick contiguous window then stride by frame_rate
                T = traj.n_frames
                if self.num_frames is None:
                    # Default: use all frames
                    num_frames = T
                else:
                    num_frames = self.num_frames

                if random_frame:
                    max_start = max(0, T - num_frames)
                    start = np.random.randint(0, max_start + 1)
                else:
                    start = start_frame

                end = min(T, start + num_frames)
                # Frame rate default: 1 (every frame)
                fr = 1 if self.frame_rate is None else max(1, int(self.frame_rate))
                traj = traj[start:end:fr]

                # Sanity check for expected_time_dim
                # expected_time_dim = total frames provided to model, including conditioning positions
                # We'll handle conditioning and then verify pos T matches expected_time_dim - C
                # Convert nm -> Å
                traj.xyz *= 10.0  # (T, N, 3)

                if traj.n_frames != self.expected_time_dim:
                    # If there's a mismatch, try to fix via simple padding/cropping
                    if traj.n_frames > self.expected_time_dim:
                        traj = traj[:self.expected_time_dim]
                    else:
                        # pad last frame to reach expected_time_dim
                        pad_needed = self.expected_time_dim - traj.n_frames
                        last = traj[-1:].xyz.copy()
                        pad_xyz = np.repeat(last, pad_needed, axis=0)
                        new_xyz = np.concatenate([traj.xyz, pad_xyz], axis=0)
                        # Build new md.Trajectory with same topology
                        traj = md.Trajectory(new_xyz, traj.topology)

                # Convert to torch tensor [N, 3, T]
                coords = torch.tensor(traj.xyz, dtype=torch.float32).permute(1, 2, 0)

                # Graph featurization
                z, edge_index, edge_type, x_features = mol_2d(mol, self.features)

                # Conditioning mask
                cond_mask = torch.zeros(self.expected_time_dim, dtype=torch.bool)
                if self.conditioning != 'none':
                    cond_mask[0] = True
                if self.conditioning == 'interpolation':
                    cond_mask[-1] = True

                denoise_coords = coords[:, :, ~cond_mask]

                # Assertions
                assert z.shape[0] == coords.shape[0], \
                    f"Atom count mismatch for {smiles}: z={z.shape[0]} vs coords={coords.shape[0]}"
                assert denoise_coords.shape[-1] == self.expected_time_dim - int(cond_mask.sum()), \
                    f"Time dim mismatch for {smiles}: {denoise_coords.shape[-1]} vs expected {self.expected_time_dim - int(cond_mask.sum())}"

                data = Data(
                    x=z,                          # [N,]
                    pos=denoise_coords,           # [N, 3, T - C]
                    edge_index=edge_index,        # [2, M]
                    edge_attr=edge_type,          # [M, 1]
                    x_features=x_features,        # [N, F]
                    original_frames=coords,       # [N, 3, T]
                    smiles=smiles,
                    rdmol=mol,
                    conf_idx=rep_idx,
                )

                # Apply transforms once (precompute)
                try:
                    data = all_transforms(data)
                except Exception as e:
                    print(f"[WARN] Transform failed for {smiles} (rep {rep_idx}): {e}")
                    continue

                self.items.append(data)
                if time_test:
                    seen_for_time_test.add(smiles)  # only first rep

        # Final sanity
        if len(self.items) == 0:
            print("[WARN] No items loaded into memory for TrajectoryDatasetTestForward.")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        # Zero I/O: return preloaded item
        return self.items[idx]


def get_test_data(config):
    if config.dataset.type == 'conformer':
        print("Loading conformer testing dataset")
        test_dataset = ConformerDatasetTest(
            pkl_path=config.dataset.test_conf_path, 
            ratio=config.test.ratio,
            subsample=config.test.subsample,
            features=config.dataset.features,
            transforms=config.dataset.transforms,
            filter=config.dataset.filter_data,
            remove_hs=config.dataset.remove_hs
        )
    elif config.dataset.type == 'trajectory' and 'forward' in config.denoiser.conditioning:
        # Check if this is TIMEWARP dataset (pkl-based), MDGEN, or standard trajectory (xtc-based)
        is_timewarp = hasattr(config.dataset, 'dataset') and config.dataset.dataset == "TIMEWARP"
        is_mdgen = hasattr(config.dataset, 'dataset') and config.dataset.dataset == "MDGEN"
        
        if is_mdgen:
            print("Loading MDGEN trajectory testing dataset for FORWARD")
            test_dataset = MDGenTrajectoryDataset(
                split_csv_path=config.dataset.test_split,
                expected_time_dim=config.dataset.expected_time_dim,
                features=config.dataset.features,
                num_frames=config.dataset.num_frames,
                transforms=config.dataset.transforms,
                frame_rate=config.dataset.frame_rate,
                start_frame=0,  # Deterministic for test
                random_frame=False,  # Deterministic for test
                remove_hs=config.dataset.remove_hs,
                conditioning=config.denoiser.conditioning
            )
        elif is_timewarp:
            print("Loading TIMEWARP trajectory testing dataset for FORWARD")
            frames_per_example_test = getattr(config.dataset, 'frames_per_example_test', 10000)
            test_dataset = TimewarpTrajectoryDataset(
                folder_path=config.dataset.test_traj_dir,
                expected_time_dim=config.dataset.expected_time_dim,
                features=config.dataset.features,
                num_frames=config.dataset.num_frames,
                transforms=config.dataset.transforms,
                frame_rate=config.dataset.frame_rate,
                start_frame=0,  # Deterministic for test
                random_frame=False,  # Deterministic for test
                remove_hs=config.dataset.remove_hs,
                conditioning=config.denoiser.conditioning,
                frames_per_example=frames_per_example_test,
                subsample=config.test.subsample
            )
        else:
            print("Loading trajectory testing dataset for FORWARD")
            test_dataset = TrajectoryDatasetTestForward(
                folder_path=config.dataset.test_traj_dir, 
                expected_time_dim=config.dataset.expected_time_dim,
                features=config.dataset.features,
                num_frames=config.dataset.num_frames,
                transforms=config.dataset.transforms,
                frame_rate=config.dataset.frame_rate,
                remove_hs=config.dataset.remove_hs,
                conditioning=config.denoiser.conditioning,
                subsample=config.test.subsample,
                num_reps=getattr(config.test, 'num_reps', None)
            )
    elif config.dataset.type == 'trajectory' and config.denoiser.conditioning =='interpolation':
        # Check if this is TIMEWARP dataset (pkl-based), MDGEN, or standard trajectory (xtc-based)
        is_timewarp = hasattr(config.dataset, 'dataset') and config.dataset.dataset == "TIMEWARP"
        is_mdgen = hasattr(config.dataset, 'dataset') and config.dataset.dataset == "MDGEN"
        
        if is_mdgen:
            print("Loading MDGEN trajectory testing dataset for INTERPOLATION")
            test_dataset = MDGenTrajectoryDataset(
                split_csv_path=config.dataset.test_split,
                expected_time_dim=config.dataset.expected_time_dim,
                features=config.dataset.features,
                num_frames=config.dataset.num_frames,
                transforms=config.dataset.transforms,
                frame_rate=config.dataset.frame_rate,
                start_frame=0,  # Deterministic for test
                random_frame=False,  # Deterministic for test
                remove_hs=config.dataset.remove_hs,
                conditioning=config.denoiser.conditioning
            )
        elif is_timewarp:
            print("Loading TIMEWARP trajectory testing dataset for INTERPOLATION")
            frames_per_example_test = getattr(config.dataset, 'frames_per_example_test', 10000)
            test_dataset = TimewarpTrajectoryDataset(
                folder_path=config.dataset.test_traj_dir,
                expected_time_dim=config.dataset.expected_time_dim,
                features=config.dataset.features,
                num_frames=config.dataset.num_frames,
                transforms=config.dataset.transforms,
                frame_rate=config.dataset.frame_rate,
                start_frame=0,  # Deterministic for test
                random_frame=False,  # Deterministic for test
                remove_hs=config.dataset.remove_hs,
                conditioning=config.denoiser.conditioning,
                frames_per_example=frames_per_example_test,
                subsample=config.test.subsample
            )
        else:
            print("Loading trajectory testing dataset for INTERPOLATION")
            test_dataset = TrajectoryDatasetTestInterpolation(
                folder_path=config.dataset.test_traj_dir,
                subsample=config.test.subsample,
                pkl_path=config.test.pkl_path, 
                expected_time_dim=config.dataset.expected_time_dim,
                features=config.dataset.features,
                transforms=config.dataset.transforms,
                remove_hs=config.dataset.remove_hs,
                ratio=config.test.ratio,
            )
    elif config.dataset.type == 'trajectory' and config.denoiser.conditioning =='none':
        print("Loading trajectory testing dataset for NO CONDITION")
        test_dataset = TrajectoryDatasetTestUncond(
            folder_path=config.dataset.test_traj_dir,
            subsample=config.test.subsample,
            num_frames=config.dataset.num_frames,
            expected_time_dim=config.dataset.expected_time_dim,
            features=config.dataset.features,
            transforms=config.dataset.transforms,
            remove_hs=config.dataset.remove_hs,
            frame_rate=config.dataset.frame_rate
        )
    elif config.dataset.type == 'both':
        # Check test_on_drugs flag to determine which test set to load
        test_on_drugs = getattr(config.dataset, 'test_on_drugs', True)  # Default to True if not specified
        if test_on_drugs:
            print("Loading DRUGS conformer testing dataset (test_on_drugs=True)")
            test_dataset = ConformerDatasetTest(
                pkl_path=config.dataset.test_conf_path_drugs,
                ratio=config.test.ratio,
                subsample=config.test.subsample,
                features=config.dataset.features,
                transforms=config.dataset.transforms,
                filter=config.dataset.filter_data,
                remove_hs=config.dataset.remove_hs
            )
        else:
            print("Loading QM9 conformer testing dataset (test_on_drugs=False)")
            test_dataset = ConformerDatasetTest(
                pkl_path=config.dataset.test_conf_path_qm9,
                ratio=config.test.ratio,
                subsample=config.test.subsample,
                features=config.dataset.features,
                transforms=config.dataset.transforms,
                filter=config.dataset.filter_data,
                remove_hs=config.dataset.remove_hs
            )
    elif config.dataset.type == 'both_trajectory':
        # Check test_on_drugs flag to determine which test set to load
        test_on_drugs = getattr(config.dataset, 'test_on_drugs', True)  # Default to True if not specified
        
        # Dataset-specific conditioning logic for both_trajectory:
        # - DRUGS (test_on_drugs=True): Use 'forward' conditioning (4 blocks with first-frame conditioning)
        # - QM9 (test_on_drugs=False): Use 'unconditional_forward' (2 blocks, no conditioning)
        if test_on_drugs:
            actual_conditioning = 'forward'
            test_dir = config.dataset.test_traj_dir_drugs
            # Use DRUGS-specific subsample if available
            subsample = getattr(config.test, 'subsample_drugs', config.test.subsample)
            print(f"Loading DRUGS trajectory testing dataset with FORWARD conditioning (test_on_drugs=True)")
            if subsample:
                print(f"Using DRUGS subsample: {subsample}")
        else:
            actual_conditioning = 'unconditional_forward'
            test_dir = config.dataset.test_traj_dir_qm9
            # Use QM9-specific subsample if available
            subsample = getattr(config.test, 'subsample_qm9', config.test.subsample)
            print(f"Loading QM9 trajectory testing dataset with UNCONDITIONAL_FORWARD conditioning (test_on_drugs=False)")
            if subsample:
                print(f"Using QM9 subsample: {subsample}")
        
        # Only 'forward' and 'unconditional_forward' are supported (no interpolation for both_trajectory)
        if 'forward' in actual_conditioning:
            test_dataset = TrajectoryDatasetTestForward(
                folder_path=test_dir,
                expected_time_dim=config.dataset.expected_time_dim,
                features=config.dataset.features,
                num_frames=config.dataset.num_frames,
                transforms=config.dataset.transforms,
                frame_rate=config.dataset.frame_rate,
                remove_hs=config.dataset.remove_hs,
                conditioning=actual_conditioning,  # Use dataset-specific conditioning
                subsample=subsample,  # Use dataset-specific subsample
                num_reps=getattr(config.test, 'num_reps', None)
            )
        else:
            raise NotImplementedError(
                f"Conditioning '{actual_conditioning}' not supported for both_trajectory. "
                f"Only 'forward' and 'unconditional_forward' are valid (no interpolation support)."
            )
    else:
        raise NotImplementedError()

    return test_dataset
