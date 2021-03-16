import os
from TradingWorkers.WorkerObjects.AnalystSettings import AnalystSettings

class BitBotSettings:
    working_currencies = []
    key = 'yUUmLgLPvUZK1cXb7Kc/lD4T99ccvGmbxCOOgIXdyMeOu28nkEABVf7w'
    secret = 'bqzWKe4qgymEPqG0WMhlh/uDyZHkZ/kIxleXxg48zHwj8gLx7paFs5htZ/Y91Bn7VHx2ViXc07JWy24kBUaklw=='

    def __init__(self):
        pass

    @staticmethod
    def GetDefault():
        result = BitBotSettings()
        result.working_currencies = [
            'BTC',
            'XRP',
            'LTC',
            'ETH',
            #'USDT', # Unnecessary, too recent.
            'EOS',
            #'BCH', # too recent
            #'LEO', # too recent
            #'BSV', # too recent
            'NEO',
            'IOTA'
        ]
        result.wait_seconds = 1
        result.data_comparison_currency = 'USD'
        result.data_start_date = '2018-01-01T00:00:00'
        result.data_time_interval = '15m'
        '''Number of times the downloader will retry a failed download.'''
        result.retry_data_download_count = 30
        result.ccxt_api = 'kraken'
        result.ohlc_batch_download_size = 700
        result.save_folder = os.path.expanduser('~\\Documents') + '\\BitBot'
        result.analyst_strategy = AnalystSettings()
        return result
