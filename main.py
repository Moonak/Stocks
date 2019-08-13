from stock import load
import numpy as np


AP=load('/home/mohammadreza/Desktop/Stocks/AP.csv')
AP.avarage(nweeks=1)
AP.norm()
AP.train(10,NPW=9,epochs=200,dropout_prob=.0,lstm_units=200, batch_size=10)
#print(AP.vol_avg)


