import os
# disable GPU usage
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

from AI.AIInterface import AIInterface
from TradingWorkers.WorkerObjects.BitBotSettings import BitBotSettings
from Data.DatabaseInterface import DatabaseInterface
import time
import sys

settings = BitBotSettings.GetDefault()
database_interface = DatabaseInterface(settings)
acd_ai = AIInterface(settings, database_interface.acd_database, 'ACD')
print(f'current time {time.localtime()}')

epoch = 0

#acd_ai.LoadModel()
acd_ai.CreateModel()
acd_ai.TrainModel(epoch)
acd_ai.SaveModel()
acd_ai.Evaluate()

for ct in settings.working_currencies:
    ai = AIInterface(settings, database_interface.databases[ct], ct)
    ai.CreateModel()
    ai.TrainModel(epoch)
    ai.SaveModel()
    ai.Evaluate()

print(f'finished time {time.localtime()}')

from Simulation.Simulator import Simulator

settings = BitBotSettings.GetDefault()
di = DatabaseInterface(settings)
simulator = Simulator(settings, di)
simulator.Run()