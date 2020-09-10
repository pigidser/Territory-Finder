from utilities import *

import os, sys
from time import time
import numpy as np
import pandas as pd

from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import balanced_accuracy_score
from sklearn.ensemble import RandomForestClassifier
import openpyxl
from openpyxl.utils.dataframe import dataframe_to_rows
from openpyxl.styles import PatternFill


class TerritoryFinder(object):

    def __init__(self, coord_file, report_file, output_file):
        """ Class initialization, logging set-up, checking input files """
        # input and output files
        self.coord_file, self.report_file, self.output_file = coord_file, report_file, output_file
        # Выборка не сбалансированная, используем class_weight='balanced', n_estimators=40
        self.model = RandomForestClassifier(class_weight='balanced', n_estimators=40, random_state=42, n_jobs=-1, warm_start=False)
        self.df = pd.DataFrame()
        self.__X_enc_train, self.__y_enc_train, self.__X_enc_pred = None, None, None
        self.log = set_up_logging()
        self.__check_env()
        self.log.debug("TerritoryFinder class initialized")
    
    def __check_env(self):
        if not os.path.isfile(self.coord_file):
            self.log.error(f"File '{self.coord_file}' not found. Please place it in a folder with this program")
            raise Exception
        if not os.path.isfile(self.report_file):
            self.log.error(f"File '{self.report_file}' not found. Please place it in a folder with this program")
            raise Exception
        self.log.debug(f"Input files were found")

    def __restore_coordinates(self):
        """ Find coordinates for an outlet by its neighbors """

        self.df['Latitude'].replace(0, np.NaN, inplace=True)
        self.df['Longitude'].replace(0, np.NaN, inplace=True)
        self.df['isCoord'] = ~( (self.df['Latitude'].isna()) | (self.df['Longitude'].isna()) )

        kladr_lat_grouped = self.df[self.df['isCoord']==1].groupby(['Kladr_level_1','Kladr_level_2','Kladr_level_3','Kladr_level_4']).Latitude.mean()
        kladr_lon_grouped = self.df[self.df['isCoord']==1].groupby(['Kladr_level_1','Kladr_level_2','Kladr_level_3','Kladr_level_4']).Longitude.mean()

        def get_avg_coordinate(row, kladr_grouped):
            """
            Вернуть среднюю координату населенного пункта, области, региона или страны. Используем функции
            multiindex.isin().any(), чтобы проверить, что в Series имеется индекс для всех 4-х уровней
            и вернуть значение. В случае отсутствия индекса, отрубить последний уровень в индексе и проверить
            индекс для 3-х уровней и т.д.

            Parameters:
            row (Series): ['Kladr_level_1','Kladr_level_2','Kladr_level_3','Kladr_level_4'] для которых нужно
                получить координату
            kladr_grouped (Series): с мультииндексом (['Kladr_level_1','Kladr_level_2','Kladr_level_3','Kladr_level_4'])
                который содержит значения координаты для 4-х уровней из адресного классификатора
                
            Returns:
            float: Координата

            """
            try:
                if kladr_grouped.index \
                        .isin([(row['Kladr_level_1'],row['Kladr_level_2'],row['Kladr_level_3'],row['Kladr_level_4'])]).any():
                    return kladr_grouped[row['Kladr_level_1'],row['Kladr_level_2'],row['Kladr_level_3'],row['Kladr_level_4']]
                elif kladr_grouped.index.droplevel(['Kladr_level_4']) \
                        .isin([(row['Kladr_level_1'],row['Kladr_level_2'],row['Kladr_level_3'])]).any():
                    return kladr_grouped[row['Kladr_level_1'],row['Kladr_level_2'],row['Kladr_level_3']].mean()
                elif kladr_grouped.index.droplevel(['Kladr_level_3','Kladr_level_4']) \
                        .isin([(row['Kladr_level_1'],row['Kladr_level_2'])]).any():
                    return kladr_grouped[row['Kladr_level_1'],row['Kladr_level_2']].mean()
                elif kladr_grouped.index.droplevel(['Kladr_level_2','Kladr_level_3','Kladr_level_4']) \
                        .isin([(row['Kladr_level_1'])]).any():
                    return kladr_grouped[row['Kladr_level_1']].mean()
                else:
                    return 0
            except:
                print(row['Kladr_level_1'],row['Kladr_level_2'],row['Kladr_level_3'],row['Kladr_level_4'])
                raise KeyError

        self.df.loc[self.df['isCoord']==0,'Latitude'] = \
            self.df.loc[self.df['isCoord']==0][['SWE_Store_Key','Kladr_level_1','Kladr_level_2','Kladr_level_3','Kladr_level_4']].apply( \
                get_avg_coordinate, args=(kladr_lat_grouped,), axis=1)

        self.df.loc[self.df['isCoord']==0,'Longitude'] = \
            self.df.loc[self.df['isCoord']==0][['SWE_Store_Key','Kladr_level_1','Kladr_level_2','Kladr_level_3','Kladr_level_4']].apply( \
                get_avg_coordinate, args=(kladr_lon_grouped,), axis=1)

        self.df.drop(['Kladr_level_1','Kladr_level_2','Kladr_level_3','Kladr_level_4','Kladr_level_5'], axis=1, inplace=True)

    def load_data(self):          
        """ Load and transform data """

        self.log.info(f"Loading coordinates...")
        df_coor = pd.read_excel(self.coord_file, nrows=1000)
        self.log.debug(f"Rows in {self.coord_file}: {df_coor.shape[0]}")
        df_coor.columns = ['SWE_Store_Key','Latitude','Longitude']
        # cleansing from invalid coordinates
        df_coor = df_coor[df_coor['Latitude']!=0]
        df_coor = df_coor[(df_coor['Latitude']>40)&(df_coor['Latitude']<82)]
        df_coor = df_coor[((df_coor['Longitude']>=10)&(df_coor['Longitude']<180)) | \
            ((df_coor['Longitude']>=-180)&(df_coor['Longitude']<-160))]

        # check if outlets are duplicated
        if df_coor.SWE_Store_Key.value_counts().values[0] > 1:
            self.log.error(f"Found duplicated codes of outlets in '{self.coord_file}!")
            raise Exception

        self.log.info(f"Loading report file...")
        df_terr = pd.read_excel(self.report_file, skiprows=1, nrows=1000)
        self.log.debug(f"Rows in {self.report_file}: {df_terr.shape[0]}")
        # rename fields
        df_terr.columns = ['Region','Distrib','Office','FFDSL','TSE_MTDE',
            'Level_Torg_Region1','Level_Torg_Region2','Filial_Name',
            'Filial_Ship_To','Chain_Type','Chain_Name','Chain_Id',
            'Chain_Chain_Tier_MWC','Chain_Chain_Sub_Tier_MWC','SWE_Store_Key',
            'Store_Status','Store_Status_NOW','Outlet_Name','Channel_Name_2018',
            'Outlet_Type_2018','Trade_Structure','From_Dc',
            'Segment_MWC_Segment_Name','Cluster_MWC','Kladr_level_1',
            'Kladr_level_2','Kladr_level_3','Kladr_level_4','Kladr_level_5',
            'LSV_WWY','LSV_CHOCO','LSV_MWC','Covering_Outlet_id',
            'General_Duplicate','Ship_To_Visited','Filial_Visited',
            'Ship_to_Name_TO_BE','Region_loaded_RSS',
            'MW_Ship_to_TO_BE_Name_loaded_RSS',
            'MW_Ship_to_TO_BE_loaded_RSS',
            'CH_Ship_to_TO_BE_Name_loaded_RSS',
            'CH_Ship_to_TO_BE_loaded_RSS',
            'WR_Ship_to_TO_BE_Name_loaded_RSS',
            'WR_Ship_to_TO_BE_loaded_RSS','Ship_to_Code_TO_BE',
            'DC','Changed',
            'Change_Period','Region_Last_Future_Ship_to',
            'Last_Future_ship_to_Name', 'Last_Future_ship_to', 'Comment']

        df_codes = pd.DataFrame(data=df_terr['SWE_Store_Key'],columns=['SWE_Store_Key'])

        # do not take unused columns
        df_terr = df_terr[['SWE_Store_Key','Region','Distrib','Office','FFDSL','TSE_MTDE','Level_Torg_Region1',
            'Level_Torg_Region2','Filial_Name','Filial_Ship_To','Chain_Type','Chain_Id','Chain_Chain_Tier_MWC',
            'Chain_Chain_Sub_Tier_MWC','Channel_Name_2018','Outlet_Type_2018','Trade_Structure','From_Dc',
            'Segment_MWC_Segment_Name','Cluster_MWC','Covering_Outlet_id','General_Duplicate','Ship_To_Visited',
            'Kladr_level_1','Kladr_level_2','Kladr_level_3','Kladr_level_4','Kladr_level_5',
            'Region_Last_Future_Ship_to','Last_Future_ship_to_Name','Last_Future_ship_to']]

        # Remove outlet-duplicates and associated fields
        df_terr = df_terr[df_terr['General_Duplicate']!='Дубликат']
        df_terr.drop(['Covering_Outlet_id','General_Duplicate'], axis=1, inplace=True)

        self.log.info("Merging territories with coordinates and start preprocessing...")
        self.df = pd.merge(df_terr, df_coor, on='SWE_Store_Key',how='left')
        del df_terr
        del df_coor

        self.log.info("Restore coordinates...")
        self.__restore_coordinates()

        self.df['isTrain'] = ~ self.df['Last_Future_ship_to'].isna()

        # Last_Future_ship_to убрать внешние пробелы и преобразовать к типу str
        # self.df['Last_Future_ship_to'] = self.df['Last_Future_ship_to'].astype(str)
        self.df.loc[self.df['isTrain']==True,'Last_Future_ship_to'] = \
            self.df.loc[self.df['isTrain']==True]['Last_Future_ship_to'].apply(align_value)

        self.df['From_Dc'] = self.df['From_Dc'].astype(int)
        self.df['Chain_Id'] = self.df['Chain_Id'].astype(str)
        # Установить поле как индекс, тем самым исключив его из списка признаков
        self.df.set_index('SWE_Store_Key',inplace=True)

    def get_ships_to_exclude(self, threshold=2):
        """
        Вернуть список классов с количеством сэмплов меньшим threshold
    
        """
        if self.df.empty:
            self.log.error(f"Envoke the load_data method first!")
            raise Exception
        if threshold < 2:
            threshold = 2
        ship_counts = self.df[~self.df['Last_Future_ship_to'].isna()].groupby('Last_Future_ship_to').size().to_frame()
        ship_counts.reset_index(inplace=True)
        ship_counts.columns = ['Last_Future_ship_to','Counts']
        
        return [str(item) for item in list(ship_counts['Last_Future_ship_to'][ship_counts['Counts']<threshold].values)]

    def __get_encoded(self):
        """ Ordinal encoding implementation """

        # fill NaN values, not touching target variable
        cat_features = self.df.select_dtypes(include=['object']).columns  # Categorical
        num_features = self.df.select_dtypes(exclude=['object']).columns  # Numeric
        self.df['Last_Future_ship_to'].replace(np.NaN, 0, inplace=True)
        for name in cat_features:
            self.df[name].fillna('missing', inplace=True)
        for name in num_features:
            self.df[name].fillna(0, inplace=True)
        self.df['Last_Future_ship_to'].replace(0, np.NaN, inplace=True)

        target = ['Last_Future_ship_to']         # Целевая переменная
        aux_target = ['Region_Last_Future_Ship_to','Last_Future_ship_to_Name']
        service = ['isTrain','isCoord']  # Сервисные признаки, которые отбросить
        # Из полного списка признаков или из переданного списка признаков исключить target, aux_target & service
        features = [column for column in self.df.columns \
                    if column not in target + aux_target + service]
        # Классы для исключения
        samples_threshold = 2
        ships_to_exclude = self.get_ships_to_exclude(samples_threshold)

        # Обучить кодировщик на полном наборе признаков
        X = self.df[(~self.df['Last_Future_ship_to'].isin(ships_to_exclude))][features]
        cat_features = X.select_dtypes(include=['object']).columns  # Categorical
        num_features = X.select_dtypes(exclude=['object']).columns  # Numeric
        self.__encoder_x = OrdinalEncoder()
        self.__encoder_x.fit(X[cat_features])

        X_train = self.df[(self.df['isTrain']==True)&(~self.df['Last_Future_ship_to'].isin(ships_to_exclude))][features]
        X_cat = self.__encoder_x.transform(X_train[cat_features])       # Transform cats features
        X_num = X_train[num_features]                                   # Not transform nums features
        self.__X_enc_train = np.hstack([X_cat, X_num])                  # Join

        X_pred = self.df[(self.df['isTrain']==False)&(~self.df['Last_Future_ship_to'].isin(ships_to_exclude))][features]
        X_cat = self.__encoder_x.transform(X_pred[cat_features])
        X_num = X_pred[num_features]
        self.__X_enc_pred = np.hstack([X_cat, X_num])
        self.log.debug(f"Shapes: X {X.shape}, X_enc_train {self.__X_enc_train.shape}, X_enc_pred {self.__X_enc_pred.shape}")
        
        # Transform y
        y = self.df[(self.df['isTrain']==True)&(~self.df['Last_Future_ship_to'].isin(ships_to_exclude))][target]
        self.__encoder_y = LabelEncoder()
        # y is a DataFrame, converting to 1D array
        self.__y_enc_train = self.__encoder_y.fit_transform(y.values.ravel())
        self.log.debug(f"Shape: y_enc_train {self.__y_enc_train.shape}")
        
    def validate(self):
        """ Split the dataset into the training and the validation parts to training and validation """
        self.__get_encoded()
        # Training-Validation split
        X_train, X_valid, y_train, y_valid = train_test_split(self.__X_enc_train, self.__y_enc_train,
            test_size=0.3, random_state=42, stratify=self.__y_enc_train)
        # Training, validation, Cross-Validation
        t0 = time()
        self.model.fit(X_train, y_train)
        self.log.debug(f"Training finished in {time() - t0:.3f} sec.")
        t0 = time()
        y_pred = self.model.predict(X_valid)
        self.val_score = balanced_accuracy_score(y_valid, y_pred)
        self.log.info(f"Balanced accuracy score: {self.val_score:.3f}")
        self.log.debug(f"Validation finished in {time() - t0:.1f} sec.")
        t0 = time()
        val_cv_score = cross_val_score(self.model, self.__X_enc_train, self.__y_enc_train, cv=3, scoring='balanced_accuracy')
        self.val_cv_score = np.array([round(item, 5) for item in val_cv_score])
        self.log.info(f"Cross-validation average score: {self.val_cv_score.mean():.3f}")
        self.log.debug(f"Cross-validation finished in {time() - t0:.3f} sec.")
        # print statistics
        self.__get_statistics(X_valid)

    def __get_statistics(self, X_valid):
        """ Print statistics """
        self.__find_top_3(X_valid)
        # # y_valid закодирован, поэтому инвертируем как было
        # self.proba['y_valid'] = self.__encoder_y.inverse_transform(y_valid)
        # self.proba['correct_1'] = self.proba.apply(lambda x: int(x.top_1_class==x.y_valid),axis=1)
        # self.proba['correct_2'] = self.proba.apply(lambda x: int(x.top_2_class==x.y_valid),axis=1)
        # self.proba['correct_3'] = self.proba.apply(lambda x: int(x.top_3_class==x.y_valid),axis=1)

        # # Оставим только новые столбцы с информацией по 3-м классам с наивысшей уверенностью
        # self.proba = self.proba.loc[:,'top_1_class':]

    def __find_top_3(self, X_valid):
        """ Define top 3 classes for each outlet without an answer """
        y_pred_proba = self.model.predict_proba(X_valid)
        self.proba = pd.DataFrame(data=y_pred_proba, columns=self.model.classes_)
        self.log.debug(f"proba.shape {self.proba.shape}")

        def get_max_3_classes(row):
            """
            Получает серию из предсказаний размерностью n-классов и возвращает три класса с максимальной вероятностью
            и значания вероятности для этих классов.
            
            """
            ser = pd.Series(data=row.values, index=self.model.classes_)
            ser.sort_values(inplace=True, ascending=False)
            return ser[0:3].index[0],ser[0:3].values[0], \
                ser[0:3].index[1],ser[0:3].values[1], \
                ser[0:3].index[2],ser[0:3].values[2]

        self.proba['top_1_class'], self.proba['top_1_proba'], \
            self.proba['top_2_class'], self.proba['top_2_proba'], \
            self.proba['top_3_class'], self.proba['top_3_proba'] = zip(*self.proba.apply(get_max_3_classes, axis=1))

    def fit(self):
        """ Training on full data set """
        self.log.info("Final training...")
        self.__get_encoded()
        t0 = time()
        self.model.fit(self.__X_enc_train, self.__y_enc_train)
        self.log.debug(f"Final training finished in {time() - t0:.3f} sec.")

    def get_report(self):
        """ Prepare a new report """
        t0 = time()
        self.log.info("Preparing report...")
        self.__find_top_3(self.__X_enc_pred)

        # X_pred.reset_index(inplace=True)
        # df_concat = pd.concat([X_pred['SWE_Store_Key'], df_proba.loc[:,'top_1_class':]], axis=1,join='inner')

        # df_info = df_codes.merge(right=df_concat,how='left',on='SWE_Store_Key')

        # del df_codes
        # del df_concat

        # self.workbook = openpyxl.load_workbook(report_file)
        # worksheet = self.workbook['Sheet1']

        # rows = dataframe_to_rows(df_info, index=False, header=True)

        # start_row = 2
        # start_col = 46
        # for r_idx, row in enumerate(rows, start_row):
        #     for c_idx, value in enumerate(row, start_col):
        #         worksheet.cell(row=r_idx, column=c_idx, value=value)
        #     if type(row[2])==float and not pd.isnull(row[2]):
        #         proba = float(row[2])
        #         if proba >= 0.9:
        #             worksheet.cell(row=r_idx, column=39, value=row[1])
        #             worksheet.cell(row=r_idx, column=39).fill = PatternFill("solid", fgColor="00D328")
        #         elif proba >= 0.8:
        #             worksheet.cell(row=r_idx, column=39, value=row[1])
        #             worksheet.cell(row=r_idx, column=39).fill = PatternFill("solid", fgColor="F9F405")
        #         else:
        #             worksheet.cell(row=r_idx, column=39, value=row[1])
        #             worksheet.cell(row=r_idx, column=39).fill = PatternFill("solid", fgColor="FCB09F")

        self.log.debug(f"Report prepared in {time() - t0:.3f} sec.")


    def save_report(self):
        self.log.info("Saving output file...")
        t0 = time()
        self.workbook.save(self.output_file)
        self.log.debug(f"Saved in {time() - t0:.3f} sec.")
        self.info(f"The new report saved as {output_file}")
