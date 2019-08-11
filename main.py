from stock import load
import matplotlib.pyplot as plt
import numpy as np


AP=load('/home/kanjouri/Moonak/Stocks/AP.csv')
AP.avarage(nweeks=2)
AP.norm()
AP.train(4)
#print(AP.vol_avg)


#plt.plot(np.arange(len(AP.close_avg)) , AP.close_avg)
#plt.plot(np.arange(len(AP.open)) , AP.open)


