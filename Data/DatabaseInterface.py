import os
import ccxt
import csv
import datetime
from math import floor
from datetime import timedelta
from decimal import Decimal
import copy
from pprint import pprint
import pandas as pd
import numpy as np
from pandas import array

from TradingWorkers.WorkerObjects.BitBotSettings import BitBotSettings

class DatabaseInterface:
    # future data structure
    csv_columns = ['date', 'open' , 'high', 'low' , 'close' , 'volume']
    csv_columns_dif = ['open' , 'high', 'low' , 'close' , 'volume']
    
    def __init__(self, settings: BitBotSettings, skip_load: bool = False):
        self.save_folder = os.path.expanduser('~\\Documents') + '\\BitBot'
        self.databases_folder = self.save_folder + '\\Databases\\Cryptocurrency History'
        self.settings = settings
        self.download_exchange = ccxt.bitfinex({
            'verbose': False,
            'enableRateLimit': True
        })
        self.start_date_8601 = self.download_exchange.parse8601(self.settings.data_start_date)
        if (self.start_date_8601 == None):
            raise Exception(f'Start date "{self.settings.data_start_date}" is invalid.')
        self.interval_8601 = self.__IntervalToExchange8601(self.settings.data_time_interval)
        self.databases = {}
        # Contains OHLCV gains and losses by percentage
        self.dif_databases = {}
        # Contains close gains and losses for each crypto type (database used to train and predict)
        self.acd_database = None
        if not skip_load:
            self.LoadDatabases()
            
    def LoadDatabases(self):
        print('Loading databases...')
        for ct in self.settings.working_currencies:
            if os.path.exists(self.GetCSVPath(ct)):
                self.databases[ct] = pd.read_csv(self.GetCSVPath(ct), engine='python')
            if os.path.exists(self.GetCSVDifferencesPath(ct)):
                self.dif_databases[ct] = pd.read_csv(self.GetCSVDifferencesPath(ct), engine='python')
        if os.path.exists(self.GetACDPath()):
            self.acd_database = pd.read_csv(self.GetACDPath(), engine='python')

    def SaveDataBases(self):
        print('Saving databases...')
        for ct in self.settings.working_currencies:
            if ct in self.databases:
                file_name = self.GetCSVPath(ct)
                if not os.path.exists(os.path.dirname(file_name)):
                    os.makedirs(os.path.dirname(file_name))
                self.databases[ct].to_csv(file_name, index=False)
            if ct in self.dif_databases:
                file_name = self.GetCSVDifferencesPath(ct)
                if not os.path.exists(os.path.dirname(file_name)):
                    os.makedirs(os.path.dirname(file_name))
                self.dif_databases[ct].to_csv(file_name, index=False)
        file_name = self.GetACDPath()
        if not os.path.exists(os.path.dirname(file_name)):
            os.makedirs(os.path.dirname(file_name))
        self.acd_database.to_csv(file_name, index=False)

        print('Done saving.')

    def __UpdateDifferencesDatabases(self):
        for ct in self.settings.working_currencies:
            if not ct in self.dif_databases:
                self.dif_databases[ct] = pd.DataFrame(columns=self.csv_columns_dif)
                self.dif_databases[ct].loc[0] = pd.to_numeric(pd.Series([1] * len(self.csv_columns_dif), index = self.csv_columns_dif))
            i = self.dif_databases[ct].shape[0]
            db_length = self.databases[ct].shape[0]
            while i < db_length:
                row = [self.databases[ct]['open'][i] / self.databases[ct]['open'][i - 1],
                    self.databases[ct]['high'][i] / self.databases[ct]['high'][i - 1],
                    self.databases[ct]['low'][i] / self.databases[ct]['low'][i - 1],
                    self.databases[ct]['close'][i] / self.databases[ct]['close'][i - 1],
                    self.databases[ct]['volume'][i] / self.databases[ct]['volume'][i - 1]]
                
                self.dif_databases[ct].loc[self.dif_databases[ct].shape[0]] = pd.Series(row, index = self.csv_columns_dif)
                i += 1
                if (i % 1000) == 0:
                    print(ct)
                    print(i)
            pprint(self.dif_databases[ct].loc[:10])

    def __UpdateACDDatabase(self):
        if self.acd_database == None:
            self.acd_database = pd.DataFrame(columns=self.settings.working_currencies)
            self.acd_database.loc[0] = pd.to_numeric(pd.Series([1] * len(self.settings.working_currencies), index = self.settings.working_currencies))
        i = self.acd_database.shape[0]
        db_length = self.GetAllLength()
        while i < db_length:
            row = []
            for ct in self.settings.working_currencies:
                row.append((self.databases[ct]['close'][i] / self.databases[ct]['close'][i - 1] - 4/5) * 5)
            
            self.acd_database.loc[self.acd_database[ct].shape[0]] = pd.Series(row, index = self.settings.working_currencies)
            i += 1
            if (i % 1000) == 0:
                print(f'Updating ACD: {i}/{db_length}')

    def UpdateDatabases(self):
        self.__UpdateDatabases()
        self.__ValidateDatabases()
        self.__UpdateACDDatabase()
        #self.__UpdateDifferencesDatabases()
        

    def __UpdateDatabases(self):
        '''Downloads all new entries for the OHLC databases.'''
        print('Updating databases...')
        for crypto_type in self.settings.working_currencies:
            last_missing_data_time = -1

            # if database is empty
            if self.databases[crypto_type].shape[0] == 0:
                print(f'Database for {crypto_type} is empty, downloading full database.')
                last_missing_data_time = self.start_date_8601
            else:
                # if database is not complete
                if self.__GetCurrentTimeIndex() != self.databases[crypto_type].shape[0] - 1:
                    print(f'Database for {crypto_type} is out of date, downloading missing entries.')
                    last_missing_data_time = self.__GetLastEntryDate(crypto_type) + self.interval_8601
            
            # if database is complete
            if last_missing_data_time == -1:
                print(f'Database for {crypto_type} is up to date, no action taken.')
                continue

            #try:
            self.__DownloadRange(crypto_type, last_missing_data_time, self.__CurrentTimeAsEntry())
            #except:
            #    print(f'Failed to download {crypto_type} OHLCV data')
             
    def __DownloadRange(self, crypto_type: str, start_time: int, end_time: int):
            print(f'downloading {start_time} to {end_time}')
            interval = self.__IntervalToExchange8601(self.settings.data_time_interval)
            total_count = (end_time - start_time) / interval
            if total_count == 0:
                return
            if total_count % 1 > 0:
                raise Exception('Error downloading range, formatting broken. total_count = (end_time - start_time) / interval: interval should be an integer. ({end_time} - {start_time}) / {interval} = {total_count}')

            # Data must be downloaded multiple times if range is too much, the following will loop
            # through the range until all data is downloaded.
            

            batches_downloaded = 0
            addition_count = 0
            last_batch = False
            ohlc = []
            download_start = start_time
            first_download = True
            while not last_batch:
                print(f'downloading batch #{batches_downloaded}')
                
                # fetching data from download_exchange
                download_attempt_count = 0
                download = []
                while True:
                    if download_start > self.__CurrentTimeAsEntry():
                        break
                    download = self.download_exchange.fetch_ohlcv(
                        crypto_type + '/' + self.settings.data_comparison_currency,
                        self.settings.data_time_interval,
                        download_start
                        )
                    download_attempt_count += 1

                    if first_download:
                        break

                    # retry failed downloads
                    if len(download) != 0:
                        break

                    print(f'Retrying download, attempt #{download_attempt_count}')

                    if download_attempt_count >= self.settings.retry_data_download_count:
                        raise Exception('Failed to download data.')
                print(f'Entries downloaded = {len(download)}')

                ohlc += download

                # sloppy bitfinex specific code
                download_start += self.__IntervalToExchange8601(self.settings.data_time_interval) * 99
                
                if download_start >= end_time: 
                    last_batch = True

                batches_downloaded += 1
                first_download = False

            # add downloaded data to database
            for data in ohlc:
                if data[0] > self.__GetLastEntryDate(crypto_type):
                    self.databases[crypto_type].loc[self.databases[crypto_type].shape[0]] = pd.Series([data[0], data[1], data[2], data[3], data[4], data[5]], index = self.csv_columns)
                    addition_count += 1
            
            print(f'downloaded {batches_downloaded} batches')
            print(f'Added {addition_count} entries')
    
    def GetCSVPath(self, crypto_type: str) -> str:
        return f'{self.databases_folder}\\ohlc_{self.settings.data_comparison_currency}{crypto_type}_{self.settings.data_time_interval}.csv'

    def GetCSVDifferencesPath(self, crypto_type: str) -> str:
        return f'{self.databases_folder}\\ohlc_{self.settings.data_comparison_currency}{crypto_type}_{self.settings.data_time_interval}_differences.csv'

    def GetACDPath(self):
        wc = ''.join(self.settings.working_currencies)
        return f'{self.databases_folder}\\acd_{wc}_{self.settings.data_time_interval}_differences.csv'

    def __CurrentTimeAsEntry(self) -> int:
        passed_time = self.download_exchange.parse8601((datetime.datetime.now() + timedelta(hours=4)).isoformat()) - self.start_date_8601
        return floor(passed_time / self.interval_8601) * self.interval_8601 + self.start_date_8601

    def __GetCurrentTimeIndex(self) -> int:
        '''What should always be the proper length of the databases.'''
        return self.__ExchangeEntryToIndex(self.__CurrentTimeAsEntry() - self.interval_8601)

    def __ExchangeEntryToIndex(self, entry: int) -> int:
        return floor((entry - self.start_date_8601) / self.interval_8601)

    def __IntervalToExchange8601(self, interval: str) -> int:
        seconds_per_unit = {"s": 1, "m": 60, "h": 3600, "d": 86400, "w": 604800}
        return int(interval[:-1]) * seconds_per_unit[interval[-1]] * 1000

    def __GetLastEntryDate(self, crypto_type: str) -> int:
        if self.databases[crypto_type].shape[0] == 0:
            return -1
        
        return int(self.databases[crypto_type]['date'][self.databases[crypto_type].shape[0] - 1])

    #def GetLastEntry(self, crypto_type: str) -> dict:
    #    if self.databases[crypto_type].shape[0] <= 0:
    #        return -1
        
    #    return int(self.databases[crypto_type][len(self.databases[crypto_type]) - 1])

    #delete this shit
    def __IndexOf(self, crypto_type: str, entry: int) -> int:
        '''Optimized to parse database in reverse.'''
        i = self.databases[crypto_type].shape[0] - 1

        while i >= 0:
            if self.databases[crypto_type]['date'][i] == str(entry):
                return i
            if int(self.databases[crypto_type]['date'][i]) < entry:
                return -1 
            i -= 1

        return False
    
    def __RemoveEntry(self, crypto_type, i: int):
        self.databases[crypto_type].drop([i])

    def __InsertEntry(self, ct: str, row, index: int):
        pd.Series(row, index = self.csv_columns)
        line = pd.DataFrame({"date": row[0], "open": row[1], "high": row[2], "low": row[3], "close": row[4], "volume": row[5]}, index=[index])
        self.databases[ct] = pd.concat([self.databases[ct].iloc[:index], line, self.databases[ct].iloc[index:]]).reset_index(drop=True)

    def __ValidateDatabases(self, attempt_fixes: bool = True, print_debug: bool = True) -> int:
        '''Checks databases for errors. Returns number of problems found. Attempts to fix issues, if enabled.'''
        if print_debug:
            print('Validating databases...')

        # Bad, should get start date from data_start_date
        expected_start_date = self.databases['BTC']['date'][0]
        problem_count = 0
        fix_count = 0

        for crypto_type in self.settings.working_currencies:
            # check for empty databases
            if (self.databases[crypto_type].shape[0] == 0):
                if print_debug:
                    print(f'WARNING: {crypto_type} database is empty')
                problem_count += 1
                continue

            # check for databases starting too soon
            start_date = self.databases[crypto_type]['date'][0]
            if start_date != expected_start_date:
                if print_debug:
                    print(f"WARNING: {crypto_type} start date too recent, start date = {start_date}")
                problem_count += 1

            # check for duplicate entries
            entry = ''
            previous_entry = ''
            duplicate_entries = []
            for i in range(0, self.databases[crypto_type].shape[0] - 1):
                entry = self.databases[crypto_type]['date'][i]
                if i > 0:
                    previous_entry = self.databases[crypto_type]['date'][i - 1]
                else:
                    previous_entry = ''
                if previous_entry == entry:
                    if print_debug:
                        print(f"WARNING: {crypto_type} Duplicate entry found at index {i}")
                    duplicate_entries.append(i)
                    problem_count += 1

            if attempt_fixes:
                for i in reversed(duplicate_entries):
                    self.__RemoveEntry(crypto_type, i)
                    fix_count += 1
                if len(duplicate_entries) > 0:
                    print('Removed duplicate entries.')

            # check for missing entries
            start_date = int(self.databases[crypto_type]['date'][0])
            expected_length = self.__GetCurrentTimeIndex()
            index = 0
            expected_entry = start_date
            missing_entries = []
            missing_entry_dates = []
            while (index < expected_length):
                entry = int(self.databases[crypto_type]['date'][index])
                if expected_entry != entry:
                    problem_count += 1
                    missing_entries.append(index)
                    missing_entry_dates.append(expected_entry)
                    if print_debug:
                        print(f"WARNING: {crypto_type}, missing entry {expected_entry} at index {index}")
                else:
                    index += 1
                if index >= self.databases[crypto_type].shape[0]:
                    break
                expected_entry += self.interval_8601
            
            if attempt_fixes:
                for i, date in zip(reversed(missing_entries), reversed(missing_entry_dates)):
                    if i == 0:
                        continue
                    self.__InsertEntry(crypto_type, self.databases[crypto_type].loc[i - 1], i)
                    self.databases[crypto_type]['date'][i] = date
                    fix_count += 1
                if len(duplicate_entries) > 0:
                    print('Patched missing entries.')

        if print_debug:
            print(f"{problem_count} database problem(s) found, {fix_count} fix(es) applied.")
        return problem_count

    def GetLength(self, crypto_type: str) -> int:
        return self.databases[crypto_type].shape[0]

    def GetAllLength(self) -> int:
        '''Returns the length of the shortest crypto currency database.'''
        lowest = -1
        for ct in self.settings.working_currencies:
            if lowest == -1:
                lowest = self.databases[ct].shape[0]
            if self.databases[ct].shape[0] < lowest:
                lowest = self.databases[ct].shape[0]
        return lowest

    def GetCloseFromIndex(self, crypto_type: str, index: int) -> Decimal:
        if crypto_type == 'USD':
            return Decimal(1)
        return Decimal(self.databases[crypto_type]['close'][index])