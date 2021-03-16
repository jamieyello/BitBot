import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

from Simulation.Simulator import Simulator
from TradingWorkers.WorkerObjects.BitBotSettings import BitBotSettings
from Data.DatabaseInterface import DatabaseInterface

settings = BitBotSettings.GetDefault()
di = DatabaseInterface(settings)
simulator = Simulator(settings, di)
simulator.Run()