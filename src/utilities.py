import traceback
import pandas as pd
import os

REPORT_FILE_ALL_COLUMNS = \
    ['Region','Distrib','Office','FFDSL','TSE_MTDE',
    'Level_Torg_Region1','Level_Torg_Region2','Filial_Name','Filial_Ship_To','Chain_Type','Chain_Name','Chain_Id',
    'Chain_Chain_Tier_MWC','Chain_Chain_Sub_Tier_MWC','SWE_Store_Key','Store_Status','Store_Status_NOW','Outlet_Name',
    'Channel_Name_2018','Outlet_Type_2018','Trade_Structure','From_Dc','Segment_MWC_Segment_Name','Cluster_MWC',
    'Kladr_level_1','Kladr_level_2','Kladr_level_3','Kladr_level_4','Kladr_level_5',
    'LSV_WWY','LSV_CHOCO','LSV_MWC','Covering_Outlet_id','General_Duplicate','Ship_To_Visited','Filial_Visited',
    'Ship_to_Name_TO_BE','Region_loaded_RSS','MW_Ship_to_TO_BE_Name_loaded_RSS',
    'MW_Ship_to_TO_BE_loaded_RSS','CH_Ship_to_TO_BE_Name_loaded_RSS','CH_Ship_to_TO_BE_loaded_RSS',
    'WR_Ship_to_TO_BE_Name_loaded_RSS','WR_Ship_to_TO_BE_loaded_RSS','Ship_to_Code_TO_BE',
    'DC','Changed','Change_Period',
    'Region_Last_Future_Ship_to','Last_Future_ship_to_Name','Last_Future_ship_to', 'Comment']

REPORT_FILE_COLUMNS = \
    ['SWE_Store_Key','Region','Distrib','Office','FFDSL','TSE_MTDE','Level_Torg_Region1',
    'Level_Torg_Region2','Filial_Name','Filial_Ship_To','Chain_Type','Chain_Id','Chain_Chain_Tier_MWC',
    'Chain_Chain_Sub_Tier_MWC','Channel_Name_2018','Outlet_Type_2018','Trade_Structure','From_Dc',
    'Segment_MWC_Segment_Name','Cluster_MWC','Ship_To_Visited',
    'Kladr_level_1','Kladr_level_2','Kladr_level_3','Kladr_level_4','Kladr_level_5',
    'Region_Last_Future_Ship_to','Last_Future_ship_to_Name','Last_Future_ship_to']

DIGITS = ['0','1','2','3','4','5','6','7','8','9']

XGB_PARAMS = {'colsample_bytree': 0.7,
				'gamma': 0.7,
				'learning_rate': 0.06,
				'max_depth': 3,
				'min_child_weight': 11.0,
				'n_estimators': 197,
				'reg_alpha': 0.7,
				'reg_lambda': 1.25,
				'subsample': 0.6,
				'transformer_nominal': 'JamesSteinEncoder',
				'transformer_ordinal': 'OrdinalEncoder',
				'under_predict_weight': 2.5}

def create_sample_files(coord_file, report_file, frac=0.1, random_state=42):
	""" Creates sample input files for dev/test purposes """
	# Remove dupes and non-active outlets, then save a fraction of data in 10%
	df_terr = pd.read_excel(report_file, skiprows=1)
	df_terr = df_terr[df_terr['Основная / Дубликат']!='Дубликат']
	df_terr = df_terr[df_terr['Store Status NOW']!='Неактивная']
	df_terr_sample = df_terr.sample(frac=frac, random_state=random_state)
	sample_report_file = os.path.splitext(report_file)[0] + ' sample.xlsx'
	with pd.ExcelWriter(sample_report_file) as writer: # pylint: disable=abstract-class-instantiated
		df_terr_sample.to_excel(writer, sheet_name='Sheet1', index=False, startrow=1)
	# Leave only used outlets in coord file 
	df_coor = pd.read_excel(coord_file)
	df_coor_sample = df_coor[df_coor['OL_ID'].isin(df_terr_sample['SWE Store Key'])]
	sample_coord_file = os.path.splitext(coord_file)[0] + ' sample.xlsx'
	with pd.ExcelWriter(sample_coord_file) as writer: # pylint: disable=abstract-class-instantiated
		df_coor_sample.to_excel(writer, sheet_name='Sheet1', index=False)

def create_sample_files_top200(coord_file, report_file, non_top200_frac = 0.1, top200_frac=1, random_state=42):
	""" Creates sample input files for dev/test purposes """
	# Remove dupes and non-active outlets, then save a fraction of data
	df_terr = pd.read_excel(report_file, skiprows=1)
	df_terr = df_terr[df_terr['Основная / Дубликат']!='Дубликат']
	df_terr = df_terr[df_terr['Store Status NOW']!='Неактивная']
	df_terr1 = df_terr[df_terr['Trade Structure']!='TOP200'].sample(frac=non_top200_frac, random_state=random_state)
	df_terr2 = df_terr[df_terr['Trade Structure']=='TOP200'].sample(frac=top200_frac, random_state=random_state)
	df_terr_sample = pd.concat([df_terr1, df_terr2], axis=1)
	sample_report_file = os.path.splitext(report_file)[0] + ' sample.xlsx'
	with pd.ExcelWriter(sample_report_file) as writer: # pylint: disable=abstract-class-instantiated
		df_terr_sample.to_excel(writer, sheet_name='Sheet1', index=False, startrow=1)
	# Leave only used outlets in coord file 
	df_coor = pd.read_excel(coord_file)
	df_coor_sample = df_coor[df_coor['OL_ID'].isin(df_terr_sample['SWE Store Key'])]
	sample_coord_file = os.path.splitext(coord_file)[0] + ' sample.xlsx'
	with pd.ExcelWriter(sample_coord_file) as writer: # pylint: disable=abstract-class-instantiated
		df_coor_sample.to_excel(writer, sheet_name='Sheet1', index=False)

def who_am_i():
   stack = traceback.extract_stack()
   filename, codeline, funcName, text = stack[-2]

   return funcName