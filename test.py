from Data.DatabaseInterface import DatabaseInterface
from TradingWorkers.WorkerObjects.BitBotSettings import BitBotSettings

import time

dbi = DatabaseInterface(BitBotSettings.GetDefault())


for column in dbi.acd_database:
    max_value = 1
    min_value = 1
    for i in range(0, dbi.acd_database.shape[0]):
        value = dbi.acd_database[column][i]
        if value > max_value:
            max_value = value
        if value < min_value:
            min_value = value
    print(f'{column} max = {max_value} min = {min_value}')


#dbi.UpdateDatabases()
#dbi.SaveDataBases()