from TradingWorkers.BitBotTrader import BitBotTrader
from TradingWorkers.WorkerObjects.BitBotSettings import BitBotSettings

manager = BitBotTrader(settings=BitBotSettings.GetDefault())
manager.Run()