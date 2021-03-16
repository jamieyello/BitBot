from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from keras.models import Model
from keras.models import Sequential
from keras.models import load_model
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import Dropout
from keras.callbacks import CSVLogger
import keras.optimizers as optimizers
from tqdm import tqdm_notebook as tqdm
from Data.DatabaseInterface import DatabaseInterface
from TradingWorkers.WorkerObjects.BitBotSettings import BitBotSettings
#from AI.Callbacks.DebugPrintCallback import DebugPrintCallback
from AI.TrainingData import TrainingData
from AI.AISettings import AISettings
import numpy as np
from numpy import array
import pandas as pd
import copy
import os
import time
from pprint import pprint

class PricePredictAI:
    ai_s = AISettings()
    __databases = {}
    __models = {}
    __td = {}

    def __init__(self, settings: BitBotSettings, df: pd.DataFrame, name: str):
        self.settings = settings
        self.__csv_columns = df.columns.values.tolist()
        self.logs_path = settings.save_folder + '\\Logs\\AI'
        self.models_path = settings.save_folder + '\\Models'
        self.__td = TrainingData()
        self.__database = df
        self.name = name

    def __PrepTrainingData(self):
        df_train, df_test = train_test_split(self.__database, train_size=0.8, test_size=0.2, shuffle=False)
        print("Train and Test size", len(df_train), len(df_test))
        # prep data, belongs in another method
        # scale the feature MinMax, build array
        x = df_train.loc[:,self.__csv_columns].values
        self.min_max_scaler = MinMaxScaler()
        
        self.x_train = self.min_max_scaler.fit_transform(x)
        self.x_test = self.min_max_scaler.transform(df_test.loc[:,self.__csv_columns])
        
        self.x_t, self.y_t = self.__BuildTimeSeries(self.x_train, 3)
        self.x_t = self.__TruncDataSet(self.x_t, self.ai_s.BATCH_SIZE)
        self.y_t = self.__TruncDataSet(self.y_t, self.ai_s.BATCH_SIZE)
        x_temp, y_temp = self.__BuildTimeSeries(self.x_test, 3)
        self.x_val, self.x_test_t = np.split(self.__TruncDataSet(x_temp, self.ai_s.BATCH_SIZE),2) # pylint: disable=unbalanced-tuple-unpacking
        self.y_val, self.y_test_t = np.split(self.__TruncDataSet(y_temp, self.ai_s.BATCH_SIZE),2) # pylint: disable=unbalanced-tuple-unpacking

    def CreateModel(self):
        self.__PrepTrainingData()
        self.__model = self.__CreateLSTM(self.x_t)

    def TrainModel(self, initial_epoch: int = 0):
        output_path = os.path.join(self.logs_path, ('ACD_training.log'))
        csv_logger = CSVLogger(output_path, append=True)
        if not os.path.exists(os.path.dirname(output_path)):
            os.makedirs(os.path.dirname(output_path))

        self.__model.fit(self.x_t, self.y_t, epochs=self.ai_s.EPOCHS, verbose=2, batch_size=self.ai_s.BATCH_SIZE,
                    shuffle=True, validation_data=(self.__TruncDataSet(self.x_val, self.ai_s.BATCH_SIZE),
                    self.__TruncDataSet(self.y_val, self.ai_s.BATCH_SIZE)), callbacks=[csv_logger],
                    initial_epoch=initial_epoch)

    def Evaluate(self):
        print('starting to predict')
        y_pred = self.__model.predict(self.__TruncDataSet(self.x_test_t, self.ai_s.BATCH_SIZE), batch_size=self.ai_s.BATCH_SIZE)
        print('done predicting')
        y_pred = y_pred.flatten()
        self.y_test_t = self.__TruncDataSet(self.y_test_t, self.ai_s.BATCH_SIZE)
        error = mean_squared_error(self.y_test_t, y_pred)
        print("Error is", error, y_pred.shape, self.y_test_t.shape)
        print(y_pred[0:15])
        print(self.y_test_t[0:15])

        # convert the predicted value to range of real data
        y_pred_org = (y_pred * self.min_max_scaler.data_range_[3]) + self.min_max_scaler.data_min_[3]
        # min_max_scaler.inverse_transform(y_pred)
        y_test_t_org = (self.y_test_t * self.min_max_scaler.data_range_[3]) + self.min_max_scaler.data_min_[3]
        # min_max_scaler.inverse_transform(y_test_t)
        print(y_pred_org[0:15])
        print(y_test_t_org[0:15])



    def Predict(self, ct: str, time_index = None):
        if time_index != None:
            df_test = self.__database
        else:
            df_test = self.__database[:time_index]

        x_test = self.min_max_scaler.transform(df_test.loc[:,self.__csv_columns])
        x_temp, y_temp = self.__BuildTimeSeries(x_test, 3)
        x_test_t = self.__TrimDataSet(x_temp, self.ai_s.BATCH_SIZE)
        y_test_t = self.__TrimDataSet(y_temp, self.ai_s.BATCH_SIZE)
        
        print(f'starting to predict {ct} at {time.localtime()}')
        y_pred = self.__model.predict(self.__TrimDataSet(x_test_t, self.ai_s.BATCH_SIZE), batch_size=self.ai_s.BATCH_SIZE)
        
        #this shit is all done
        print('done predicting')
        y_pred = y_pred.flatten()
        y_test_t = self.__TrimDataSet(y_test_t, self.ai_s.BATCH_SIZE)
        error = mean_squared_error(y_test_t, y_pred)
        print("Error is", error, y_pred.shape, y_test_t.shape)
        print(y_pred[0:15])
        print(y_test_t[0:15])
        
        # convert the predicted value to range of real data
        y_pred_org = (y_pred * self.min_max_scaler.data_range_[3]) + self.min_max_scaler.data_min_[3]
        # min_max_scaler.inverse_transform(y_pred)
        y_test_t_org = (y_test_t * self.min_max_scaler.data_range_[3]) + self.min_max_scaler.data_min_[3]
        # min_max_scaler.inverse_transform(y_test_t)
        print(y_pred_org[0:15])
        pprint(y_pred_org)
        print(y_test_t_org[0:15])
        return y_test_t_org
        
    def __CreateLSTM(self, x_t) -> Sequential:
        lstm_model = Sequential()
        # (batch_size, timesteps, data_dim)
        lstm_model.add(LSTM(100, batch_input_shape=(self.ai_s.BATCH_SIZE, self.ai_s.TIME_STEPS, x_t.shape[2]),
                            dropout=0.0, recurrent_dropout=0.0, stateful=True, return_sequences=True,
                            kernel_initializer='random_uniform'))
        lstm_model.add(Dropout(0.4))
        lstm_model.add(LSTM(60, dropout=0.0))
        lstm_model.add(Dropout(0.4))
        lstm_model.add(Dense(20,activation='relu'))
        lstm_model.add(Dense(1,activation='sigmoid'))
        optimizer = optimizers.RMSprop(lr=self.ai_s.LEARNING_RATE)
        lstm_model.compile(loss='mean_squared_error', optimizer=optimizer)
        return lstm_model
                
    def SaveModel(self):
        wc = ''.join(self.__csv_columns)
        output_path = os.path.join(self.models_path, (self.name + '_' + wc + '_model.h5'))
        
        if not os.path.exists(self.models_path):
            os.makedirs(self.models_path)
        
        self.__model.save(output_path, overwrite=True)

    def GetModel(self) -> Model:
        return self.__model

    def LoadModel(self) -> bool:
        wc = ''.join(self.__csv_columns)
        output_path = os.path.join(self.models_path, (self.name + '_' + wc + '_model.h5'))
        if os.path.exists(output_path):
            self.__model = load_model(output_path)
            if self.__models != None:
                self.__PrepTrainingData()
                return True
        return False

    def __BuildTimeSeries(self, mat, y_col_index):
        # y_col_index is the index of column that would act as output column
        # total number of time-series samples would be len(mat) - TIME_STEPS
        dim_0 = mat.shape[0] - self.ai_s.TIME_STEPS
        dim_1 = mat.shape[1]
        x = np.zeros((dim_0, self.ai_s.TIME_STEPS, dim_1))
        y = np.zeros((dim_0,))
        
        for i in tqdm(range(dim_0)):
            x[i] = mat[i:self.ai_s.TIME_STEPS+i]
            y[i] = mat[self.ai_s.TIME_STEPS+i, y_col_index]
        #print("length of time-series i/o",x.shape,y.shape)
        return x, y

    def __TruncDataSet(self, mat, batch_size):
        """
        truncates dataset to a size that's divisible by BATCH_SIZE
        """
        no_of_rows_drop = mat.shape[0]%batch_size
        if(no_of_rows_drop > 0):
            return mat[:-no_of_rows_drop]
        else:
            return mat

    def __TrimDataSet(self, mat, batch_size):
        """
        trims dataset to a size that's divisible by BATCH_SIZE
        """
        no_of_rows_drop = mat.shape[0]%batch_size
        if(no_of_rows_drop > 0):
            return mat[no_of_rows_drop:]
        else:
            return mat
