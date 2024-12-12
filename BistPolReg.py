import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import r2_score

veri=yf.download("XU100.IS",start="2020-07-27",interval="1wk")["Adj Close"]
veri=pd.DataFrame(veri)
veri.rename(columns={"XU100.IS":"Fiyat"},inplace=True)

x=np.arange(len(veri["Fiyat"]))
y=veri["Fiyat"]
katsayı=np.polyfit(x,y,2)
polfonk=np.poly1d(katsayı)
trend=polfonk(x)
r2=r2_score(y,trend)
hata=y-trend
ss=np.std(hata)


plt.plot(veri.index,y,"bo",label="Endeks")
plt.plot(veri.index,trend,"r-",label="Trend")
plt.fill_between(veri.index,trend-ss,trend+ss,color="navy",alpha=0.3,label="±1 Standart Sapma")
plt.fill_between(veri.index,trend-2*ss,trend+2*ss,color="darkred",alpha=0.3,label="±2 Standart Sapma")
plt.fill_between(veri.index,trend-3*ss,trend+3*ss,color="gray",alpha=0.3,label="±3 Standart Sapma")
plt.title(f"Bist100 Polinomsal Regresyon (R-Kare: {r2:.2f})")
plt.text(0.005,0.8,"*Gözlemler Haftalık Veridir",ha="left",va="top",transform=plt.gca().transAxes,fontsize=12,color="blue")
plt.legend(loc="upper left")
plt.show()
