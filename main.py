from stock import load

AP=load('/home/mohammadreza/Desktop/Stocks/AP.csv')
AP.avarage(nweeks=1)
AP.norm()
AP.train(10,NT=10,NPW=9,epochs=40,dropout_prob=.0,lstm_units=100, batch_size=2)

#print(AP.vol_avg)


