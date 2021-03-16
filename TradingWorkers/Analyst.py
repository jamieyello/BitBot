from TradingWorkers.WorkerObjects.BitBotSettings import BitBotSettings
from Data.DatabaseInterface import DatabaseInterface
from AI.AIInterface import AIInterface

class Analyst:
    def __init__(self, settings: BitBotSettings, crypto_aiis, database_interface: DatabaseInterface):
        self.settings = settings
        self.crypto_aiis = crypto_aiis
        self.database_interface = database_interface

    def GetRecommendation(self, time_index = None) -> str:
        '''Returns empty if no buy is recommended.'''
        predicted_prices = {}
        current_price = {}
        look_ahead = self.settings.analyst_strategy.look_ahead
        highest_estimated_gain = 0
        suggested_trade = ''

        # predict the future
        for ct in self.settings.working_currencies:
            predicted_prices[ct] = self.crypto_aiis[ct].Predict(time_index)
            if time_index == None:
                current_price[ct] = self.database_interface.GetLastEntry(ct)['close']
            else:
                current_price[ct] = self.database_interface.databases[ct]['close'][time_index]
            print(f'CURRENT PRICE: {current_price[ct]}, PREDICTED PRICES: (start) {predicted_prices[ct][:look_ahead]} (end) {predicted_prices[ct][look_ahead:]}')
        
        # decide outcome
        for ct in self.settings.working_currencies:
            for i in range(1, look_ahead):
                estimated_gain = float(predicted_prices[ct][i]) / float(current_price[ct])
                if estimated_gain > highest_estimated_gain and estimated_gain > self.settings.analyst_strategy.minimum_profit_target:
                    highest_estimated_gain = estimated_gain
                    suggested_trade = ct
                print(estimated_gain)

        return suggested_trade