import time
from datetime import datetime

from TradingWorkers.WorkerObjects.BitBotSettings import BitBotSettings
from Data.DatabaseInterface import DatabaseInterface

class BitBotTrader:
    def __init__(self, settings: BitBotSettings = None):
        if settings == None:
            self.settings = BitBotSettings.GetDefault()
        else:
            self.settings = settings
        self.database_interface = DatabaseInterface(self.settings)

    def Run(self):
        # update crypto currency value data
        self.database_interface.UpdateDatabases()

        # wait for the second to pass to continue
        second = datetime.now().second
        while second == datetime.now().second:
            time.sleep(.01)
        second = datetime.now().second
        
        while True:
            self.__ProcessOneFrame()
            
            # wait for settings.wait_seconds of seconds
            waited_time = 0
            while True:
                while second == datetime.now().second: 
                    time.sleep(.01)
                second = datetime.now().second
                waited_time += 1
                if waited_time >= self.settings.wait_seconds:
                    break

    def __ProcessOneFrame(self):
        pass