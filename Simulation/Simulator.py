from TradingWorkers.WorkerObjects.BitBotSettings import BitBotSettings
from TradingWorkers.TradingAPIs.SimTradingAPI import SimTradingAPI
from TradingWorkers.Analyst import Analyst
from Data.DatabaseInterface import DatabaseInterface
from AI.AIInterface import AIInterface

class Simulator:
    def __init__(self, settings: BitBotSettings, database_interface: DatabaseInterface):
        self.database_interface = database_interface
        self.trading_api = SimTradingAPI(settings, self.database_interface)
        self.settings = settings
        self.ai_interfaces = {}
        for ct in self.settings.working_currencies:
            self.ai_interfaces[ct] = AIInterface(settings, database_interface.databases[ct], ct)
            self.ai_interfaces[ct].LoadModel()
        self.analyst = Analyst(settings, self.ai_interfaces, self.database_interface)
        self.current_currency = 'USD'

    def Run(self):
        self.AITrade()

    def TradeRandomly(self):
        self.trading_api.SubmitConversion('USD', 'BTC', 1)
        while True:
            self.trading_api.SubmitConversion('BTC', 'EOS', 1)
            self.trading_api.SubmitConversion('EOS', 'BTC', 1)
            self.trading_api.time_index += 500

    def AITrade(self):
        self.trading_api.time_index = 50000
        trade_count = 5000

        for i in range(1, trade_count):
            recommendation = self.analyst.GetRecommendation(self.trading_api.time_index)
            print(f'recommendation = {recommendation}')
            if (recommendation != '') and (recommendation != self.current_currency):
                self.trading_api.SubmitConversion(self.current_currency, recommendation)
                self.current_currency = recommendation
            self.trading_api.time_index += 1
        
        print(f'BALANCE EQUITY = {self.trading_api.GetTotalEquity()}')