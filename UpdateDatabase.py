from Data.DatabaseInterface import DatabaseInterface
from TradingWorkers.WorkerObjects.BitBotSettings import BitBotSettings

database_interface = DatabaseInterface(BitBotSettings.GetDefault())
database_interface.UpdateDatabases()
database_interface.SaveDataBases()