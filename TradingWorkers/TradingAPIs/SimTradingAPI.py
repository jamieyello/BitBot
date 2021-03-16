from TradingWorkers.TradingAPIs.TradingAPI import TradingAPI
from TradingWorkers.WorkerObjects.BitBotSettings import BitBotSettings
from Data.DatabaseInterface import DatabaseInterface
from decimal import Decimal

class SimTradingAPI(TradingAPI):
    def __init__(self, settings: BitBotSettings, database_interface: DatabaseInterface):
        self.settings = settings
        self.database_interface = database_interface
        self.usd = Decimal(50)
        self.crypto_wallet = {}
        self.time_index = 0
        for crypto_type in settings.working_currencies:
            self.crypto_wallet[crypto_type] = Decimal(0)

    def GetFunds(self, crypto_type: str) -> Decimal:
        if crypto_type == 'USD':
            return self.usd

        return self.crypto_wallet[crypto_type]

    def __AddFunds(self, crypto_type: str, value: Decimal):
        self.__SetFunds(crypto_type, self.GetFunds(crypto_type) + value)

    def __SetFunds(self, crypto_type: str, value: Decimal):
        if crypto_type == 'USD':
            self.usd = value
            return

        self.crypto_wallet[crypto_type] = value
        
    def GetCurrentPrice(self, crypto_type: str) -> Decimal:
        return self.database_interface.GetCloseFromIndex(crypto_type, self.time_index)

    def GetTotalEquity(self) -> Decimal:
        result = self.usd

        for crypto_type in self.settings.working_currencies:
            result += self.database_interface.GetCloseFromIndex(crypto_type, self.time_index) * self.crypto_wallet[crypto_type]

        return result

    def SubmitConversion(self, from_currency: str, to_currency: str, percentage: Decimal = Decimal(1)):
        from_price = self.database_interface.GetCloseFromIndex(from_currency, self.time_index)
        to_price = self.database_interface.GetCloseFromIndex(to_currency, self.time_index)

        from_result = self.GetFunds(from_currency) * (Decimal(1) - percentage)
        to_result = from_price * self.GetFunds(from_currency) * percentage / to_price
        
        self.__SetFunds(from_currency, from_result)
        self.__AddFunds(to_currency, to_result)

        print(f"Bought {to_result} of {to_currency}")

    def IsTradePending(self, crypto_type: str) -> bool:
        return False