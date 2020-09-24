import traceback

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

def who_am_i():
   stack = traceback.extract_stack()
   filename, codeline, funcName, text = stack[-2]

   return funcName