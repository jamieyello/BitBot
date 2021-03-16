from decimal import Decimal
from TradingWorkers.WorkerObjects.BitBotSettings import BitBotSettings

class TradingAPI:
    def __init__(self, settings: BitBotSettings):
        pass

    def GetCurrentPrice(self, crypto_type: str) -> Decimal:
        raise Exception('GetCurrentPrice not implemented.')

    def GetFunds(self, crypto_type: str) -> Decimal:
        raise Exception('GetFunds not implemented.')

    def GetTotalEquity(self) -> Decimal:
        raise Exception('GetCurrentEquity not implemented.')

    def SubmitConversion(self, from_currency: str, to_currency: str, percentage: Decimal = Decimal(1)):
        raise Exception('ConvertCurrency not implemented.')

    def IsTradePending(self, crypto_type: str) -> bool:
        raise Exception('IsTradePending not implemented.')