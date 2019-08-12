from stock import load
import matplotlib.pyplot as plt
import numpy as np


AP=load('/home/mohammadreza/Desktop/Stocks/AP.csv')
AP.avarage(nweeks=1)
AP.norm()
AP.train(9,epochs=50,dropout_prob=.1)
#print(AP.vol_avg)


#plt.plot(np.arange(len(AP.close_avg)) , AP.close_avg)
#plt.plot(np.arange(len(AP.open)) , AP.open)


