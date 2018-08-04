import quandl
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from sklearn import preprocessing
import talib as tb
import plotly as py
import plotly.graph_objs as go
#from tabulate import tabulate

quandl.ApiConfig.api_key = "Zx2MBnZf7N3scjxqfiGv"
# GhFWCbsLK_Qn8sZBVt3R

df1 = quandl.get("NSE/RELIANCE",trim_start = "2010-01-01", trim_end = "2017-12-01")
df2 = quandl.get("NSE/HDFCBANK",trim_start = "2010-01-01", trim_end = "2017-12-01")
df3 = quandl.get("NSE/TCS",trim_start = "2010-01-01", trim_end = "2017-12-01")
df4 = quandl.get("NSE/ITC",trim_start = "2010-01-01", trim_end = "2017-12-01")
df5 = quandl.get("NSE/AUROPHARMA",trim_start = "2010-01-01", trim_end = "2017-12-01")
df6 = quandl.get("NSE/HDFC",trim_start = "2010-01-01", trim_end = "2017-12-01")
df7 = quandl.get("NSE/HINDUNILVR",trim_start = "2010-01-01", trim_end = "2017-12-01")
df8 = quandl.get("NSE/MARUTI",trim_start = "2010-01-01", trim_end = "2017-12-01")
df9 = quandl.get("NSE/INFY",trim_start = "2010-01-01", trim_end = "2017-12-01")
df10 = quandl.get("NSE/ONGC",trim_start = "2010-01-01", trim_end = "2017-12-01")
df11 = quandl.get("NSE/SBIN",trim_start = "2010-01-01", trim_end = "2017-12-01")
df12 = quandl.get("NSE/KOTAKBANK",trim_start = "2010-01-01", trim_end = "2017-12-01")
df13 = quandl.get("NSE/IOC",trim_start = "2010-01-01", trim_end = "2017-12-01")
df14 = quandl.get("NSE/ICICIBANK",trim_start = "2010-01-01", trim_end = "2017-12-01")
df15 = quandl.get("NSE/COALINDIA",trim_start = "2010-01-01", trim_end = "2017-12-01")
df16 = quandl.get("NSE/LT",trim_start = "2010-01-01", trim_end = "2017-12-01")
df17 = quandl.get("NSE/BHARTIARTL",trim_start = "2010-01-01", trim_end = "2017-12-01")
df18 = quandl.get("NSE/NTPC",trim_start = "2010-01-01", trim_end = "2017-12-01")
df19 = quandl.get("NSE/HCLTECH",trim_start = "2010-01-01", trim_end = "2017-12-01")
df20 = quandl.get("NSE/AXISBANK",trim_start = "2010-01-01", trim_end = "2017-12-01")
df21 = quandl.get("NSE/WIPRO",trim_start = "2010-01-01", trim_end = "2017-12-01")
df22 = quandl.get("NSE/SUNPHARMA",trim_start = "2010-01-01", trim_end = "2017-12-01")
df23 = quandl.get("NSE/VEDL",trim_start = "2010-01-01", trim_end = "2017-12-01")
df24 = quandl.get("NSE/ULTRACEMCO",trim_start = "2010-01-01", trim_end = "2017-12-01")
df25 = quandl.get("NSE/ASIANPAINT",trim_start = "2010-01-01", trim_end = "2017-12-01")
df26 = quandl.get("NSE/INDUSINDBK",trim_start = "2010-01-01", trim_end = "2017-12-01")
df27 = quandl.get("NSE/POWERGRID",trim_start = "2010-01-01", trim_end = "2017-12-01")
df28 = quandl.get("NSE/TATAMOTORS",trim_start = "2010-01-01", trim_end = "2017-12-01")
df29 = quandl.get("NSE/BPCL",trim_start = "2010-01-01", trim_end = "2017-12-01")
df30 = quandl.get("NSE/BAJFINANCE",trim_start = "2010-01-01", trim_end = "2017-12-01")
df31 = quandl.get("NSE/MM",trim_start = "2010-01-01", trim_end = "2017-12-01")
df32 = quandl.get("NSE/BAJAJ_AUTO",trim_start = "2010-01-01", trim_end = "2017-12-01")
df33 = quandl.get("NSE/ADANIPORTS",trim_start = "2010-01-01", trim_end = "2017-12-01")
df34 = quandl.get("NSE/EICHERMOT",trim_start = "2010-01-01", trim_end = "2017-12-01")
df35 = quandl.get("NSE/GAIL",trim_start = "2010-01-01", trim_end = "2017-12-01")
df36 = quandl.get("NSE/TATASTEEL",trim_start = "2010-01-01", trim_end = "2017-12-01")
df37 = quandl.get("NSE/HEROMOTOCO",trim_start = "2010-01-01", trim_end = "2017-12-01")
df38 = quandl.get("NSE/YESBANK",trim_start = "2010-01-01", trim_end = "2017-12-01")
df39 = quandl.get("NSE/INFRATEL",trim_start = "2010-01-01", trim_end = "2017-12-01")
df40 = quandl.get("NSE/TECHM",trim_start = "2010-01-01", trim_end = "2017-12-01")
df41 = quandl.get("NSE/HINDPETRO",trim_start = "2010-01-01", trim_end = "2017-12-01")
df42 = quandl.get("NSE/BOSCHLTD",trim_start = "2010-01-01", trim_end = "2017-12-01")
df43 = quandl.get("NSE/ZEEL",trim_start = "2010-01-01", trim_end = "2017-12-01")
df44 = quandl.get("NSE/IBULHSGFIN",trim_start = "2010-01-01", trim_end = "2017-12-01")
df45 = quandl.get("NSE/HINDALCO",trim_start = "2010-01-01", trim_end = "2017-12-01")
df46 = quandl.get("NSE/AMBUJACEM",trim_start = "2010-01-01", trim_end = "2017-12-01")
df47 = quandl.get("NSE/CIPLA",trim_start = "2010-01-01", trim_end = "2017-12-01")
df48 = quandl.get("NSE/UPL",trim_start = "2010-01-01", trim_end = "2017-12-01")
df49 = quandl.get("NSE/DRREDDY",trim_start = "2010-01-01", trim_end = "2017-12-01")
df50 = quandl.get("NSE/LUPIN",trim_start = "2010-01-01", trim_end = "2017-12-01")

# print("alpha=")
# alpha=input()
# alpha=open-close
c=pd.DataFrame()
o=pd.DataFrame()
h=pd.DataFrame()
l=pd.DataFrame()
lt=pd.DataFrame()
a=pd.DataFrame()
v=pd.DataFrame()

my_dict = {"df1":df1, "df2":df2, "df3":df3, "df4":df4, "df5":df5,"df6":df6
,"df7":df7,"df8":df8, "df9":df9, "df10":df10,"df11":df11, "df12":df12, "df13":df13, "df14":df14,"df15":df15
,"df16":df16,"df17":df17, "df18":df18, "df19":df19,"df20":df20, "df21":df21, "df22":df22, "df23":df23, "df24":df24
,"df25":df25,"df26":df26, "df27":df27, "df28":df28,"df29":df29, "df30":df30, "df31":df31, "df32":df32, "df33":df33
,"df34":df34,"df35":df35, "df36":df36, "df37":df37,"df38":df38, "df39":df39, "df40":df40, "df41":df41, "df42":df42
,"df43":df43,"df44":df44, "df45":df45, "df46":df46,"df47":df47, "df48":df48, "df49":df49, "df50":df50}

def mean_sma(x=pd.DataFrame(),k=30):
    y=pd.DataFrame(columns = x.columns)
    for i in range(0,len(x.columns)):
        y.iloc[:,i]=pd.Series(tb.SMA((x.iloc[:,i]).values,timeperiod=k))
    y2=y.set_index(x.index)
    return y2

def mean_ema(x=pd.DataFrame(),k=30):
    y=pd.DataFrame(columns = x.columns)
    for i in range(0,len(x.columns)):
        y.iloc[:,i]=pd.Series(tb.EMA((x.iloc[:,i]).values,timeperiod=k))
    y2=y.set_index(x.index)
    return y2

def cs_rank(x=pd.DataFrame()):
    y2=x.rank(axis=1,pct=True)
    return y2

def cs_mean(x=pd.DataFrame()):
    y=pd.DataFrame(columns=x.columns,index=x.index)
    for i in range(len(x.columns)):
        y.iloc[:,i]=x.mean(axis=1)
    return y

def ts_rank(x=pd.DataFrame(),k=30):
    y2=pd.DataFrame(index=x.index,columns=x.columns)
    for i in range(k,len(x)+1):
        y=(x.iloc[i-k:i,:]).rank(axis=0,pct=True)
        y2.iloc[i-1,:]=y.iloc[-1,:]
    return y2

def sum(x=pd.DataFrame(),k=30):
    y2=pd.DataFrame(index=x.index,columns=x.columns)
    for i in range(k,len(x)+1):
        y=(x.iloc[i-k:i,:]).sum(axis=0)
        y2.iloc[i-1,:]=y
    return y2

def ts_max(x=pd.DataFrame(),k=30):
    y2=pd.DataFrame(index=x.index,columns=x.columns)
    for i in range(k,len(x)+1):
        y=(x.iloc[i-k:i,:]).max(axis=0)
        y2.iloc[i-1,:]=y
    return y2

def cs_max(x=pd.DataFrame()):
    y=pd.DataFrame(columns=x.columns,index=x.index)
    for i in range(len(x.columns)):
        y.iloc[:,i]=x.max(axis=1)
    return y

def ts_min(x=pd.DataFrame(),k=30):
    y2=pd.DataFrame(index=x.index,columns=x.columns)
    for i in range(k,len(x)+1):
        y=(x.iloc[i-k:i,:]).min(axis=0)
        y2.iloc[i-1,:]=y
    return y2

def cs_min(x=pd.DataFrame()):
    y=pd.DataFrame(columns=x.columns,index=x.index)
    for i in range(len(x.columns)):
        y.iloc[:,i]=x.min(axis=1)
    return y

def stddev(x=pd.DataFrame(),k=30,nbdev=1):
    y=pd.DataFrame(columns = x.columns)
    for i in range(0,len(x.columns)):
        y.iloc[:,i]=pd.Series(tb.STDDEV((x.iloc[:,i]).values,timeperiod=k))
    y2=y.set_index(x.index)
    return y2

def ts_zscore(x=pd.DataFrame(),k=30):
    y = pd.DataFrame(columns=x.columns, index=x.index)
    y = (x-mean_sma(x,k))/stddev(x,k)
    return y

def delay(x=pd.DataFrame(),k=30):
    y=x.shift(periods=k)
    return y

def delta(x=pd.DataFrame(),k=30):
    y=x-delay(x,k)
    return y

def momentum(x=pd.DataFrame(),k=30):
    y=pd.DataFrame(columns = x.columns)
    for i in range(0,len(x.columns)):
        y.iloc[:,i]=pd.Series(tb.MOM((x.iloc[:,i]).values,timeperiod=k))
    y2=y.set_index(x.index)
    return y2

for variable in my_dict:
    o[[variable]]=(my_dict[variable])[['Open']]
    h[[variable]]=(my_dict[variable])[['High']]
    l[[variable]]=(my_dict[variable])[['Low']]
    c[[variable]]=(my_dict[variable])[['Close']]
    lt[[variable]]=(my_dict[variable])[['Last']]
    v[[variable]]=(my_dict[variable])[['Total Trade Quantity']]

o=o.dropna(axis=1, how='any')
h=h.dropna(axis=1, how='any')
l=l.dropna(axis=1, how='any')
c=c.dropna(axis=1, how='any')
v=v.dropna(axis=1, how='any')
lt=lt.dropna(axis=1, how='any')

# o=o.fillna(method='ffill')
# h=h.fillna(method='ffill')
# l=l.fillna(method='ffill')
# c=c.fillna(method='ffill')
# v=v.fillna(method='ffill')
# lt=lt.fillna(method='ffill')
# o=o.fillna(method='bfill')
# h=h.fillna(method='bfill')
# l=l.fillna(method='bfill')
# c=c.fillna(method='bfill')
# v=v.fillna(method='bfill')
# lt=lt.fillna(method='bfill')

# print(c)
# # print(v)
# print(h)
temp=c.copy()
temp.iloc[:,:]=1

def plot_graph(a):
    #a=input("alpha=")

    w=str(a)
    a=eval(a)
    if(type(a)==int or type(a)==float):
        a=temp
    # a=(h+l+c)/3-c
    # a = v/mean(v,20)
    # a=mean_exp(mean(c,30),30)
    count=0
    for i in range(0,len(a)):
        if ((a.iloc[i,:]).isnull().all().all()):
            count=count+1
        elif((a.iloc[i,:]).isnull().all().all()==0):
            break
    a=a.dropna()
    # print(count)
    # print(a)
    # a = a.div(abs(sum(axis = 1)),axis = 0)
    cnt=0
    for i in range(len(a)):
        divider =abs(a.iloc[[i]]).sum(axis=1)
        if ((divider.values==0).all()):
            divider=1
        a.iloc[[i]]=a.iloc[[i]]/float(divider)

    # p=a.copy()
    # x = a.values #returns a numpy array
    # min_max_scaler = preprocessing.MinMaxScaler(feature_range=(-1,1))
    # x_scaled = min_max_scaler.fit_transform(a)
    # a = pd.DataFrame(x_scaled,index= p.index,columns = p.columns)
    # print(a)
    b=100000
    n=pd.DataFrame()
    diff=pd.DataFrame()
    # for index, row in a.iterrows():
    #     if index!=datetime.strptime('2010-01-04 00:00:00','%Y-%m-%d %H:%M:%S'):
    #         n=b*a[[index-timedelta(days=1)]]/c[[index-timedelta(days=1)]]
    #         pnl=n*(c[[index]]-c[[index-timedelta(days=1)]])
    # print(pnl)

    for i in range(1,len(a)):
        t = (a.iloc[[i-1]]*b/c.iloc[[i-1+count]])
        x= np.floor(t.astype(np.double))
        # print(x)
        y =c.iloc[[i+count]]-c.iloc[[i-1+count]].values.squeeze()
        n = pd.concat([n,x])
        diff = pd.concat([diff,y])
    # print(diff)
    # print('**')
    # print(n)
    my_dict2 = {}
    my_dict2 = my_dict2.fromkeys(a.columns)

    header2 = n.columns.tolist()
    header2 = ['Date'] + header2
    Date = n.index.tolist()
    for i in range(len(Date)):
	       Date[i]=str(Date[i].strftime('%Y-%m-%d'))
    j=[]
    for i,t in zip(range(1,len(my_dict2)+1),my_dict2.items()):
        variable=t[0]
        my_dict2[variable] = n.iloc[:,i-1].values.tolist()
        j.append(my_dict2[variable])
    values2=[Date]+j

    pnl=diff*n.shift()
    # print(pnl)
    pnl=pnl.dropna()
    pnl=np.around(pnl.astype(np.double),2)

    header3 = pnl.columns.tolist()
    header3 = ['Date'] + header3
    Date2 = pnl.index.tolist()
    for i in range(len(Date2)):
	       Date2[i]=str(Date2[i].strftime('%Y-%m-%d'))
    j2=[]
    for i,t in zip(range(1,len(my_dict2)+1),my_dict2.items()):
        variable=t[0]
        my_dict2[variable] = pnl.iloc[:,i-1].values.tolist()
        j2.append(my_dict2[variable])
    values3=[Date2]+j2

    pnl=pnl.sum(axis=1)
    gross_pnl=pd.DataFrame(columns=['profit'])
    for i in range(0,len(pnl)):
        k=pnl[0:i+1].sum(axis=0)
        gross_pnl.loc[i]=k
    total_pnl=pnl.sum(axis=0)
    gross_pnl=gross_pnl.set_index(pnl.index)
    # print(pnl)
    # print(gross_pnl)
    # plt.plot(gross_pnl.index,gross_pnl.values)
    # plt.show()

    cal = pd.DataFrame(columns=gross_pnl.columns, index=gross_pnl.index)
    for i in range(0, 7):
        if i != 0:
            cal.iloc[i, 0] = gross_pnl.iloc[250 * i-1, 0] - gross_pnl.iloc[250 * (i - 1)-1, 0]
        elif i == 0:
            cal.iloc[i, 0] = gross_pnl.iloc[250 * i-1, 0]

    cal=cal.dropna()
    cal= pd.concat([cal, gross_pnl.iloc[[-1]].values-gross_pnl.iloc[[250*7]]],ignore_index=True)
    cal = cal.dropna()
    cal['year'] = ['2010', '2011', '2012', '2013', '2014', '2015', '2016', '2017']
    cal = cal.set_index('year')

    yearly_pnl = cal
    # yearly_pnl.columns=['Yearly PNL']
    yearly_ret =yearly_pnl*100/b
    yearly_ret=np.around(yearly_ret.astype(np.double),2)
    #yearly_ret=round(yearly_ret,2)
    yearly_ret.columns=['Yearly Returns']
    #needs editing b+grosspnl
    # yearly_ret.columns=['Yearly returns']

    series_frame = pnl.to_frame()
    new_2 = stddev(series_frame,250)
    std = pd.DataFrame(columns=gross_pnl.columns,index=gross_pnl.index)

    for i in range(1, 8):
        std.iloc[i - 1, 0] = new_2.iloc[250 * i  - 1, 0]

    new_3=stddev(series_frame,1968-250*7)
    new_3.columns=['profit']
    std=std.dropna()
    e = new_3.iloc[[-1]]
    std= pd.concat([std, e],ignore_index=True)
    std= std.dropna()

    std['year'] = ['2010', '2011', '2012', '2013', '2014', '2015', '2016', '2017']
    std = std.set_index('year')
    # std.columns=['Yearly standard deviation of PNL']

    ir=(yearly_pnl/250)/std
    #ir=round(ir,2)
    ir.columns=['IR']
    ir=np.around(ir.astype(np.double),2)
    # print(ir)
    sharpe=(ir*15.8)
    #sharpe=round(sharpe,2)
    sharpe.columns=['Sharpe Ratio']
    sharpe=np.around(sharpe.astype(np.double),2)
    # print(sharpe)

    tvr=pd.DataFrame(columns=a.columns,index=a.index)
    for i in range(1,len(a)):
        tvr.iloc[[i]] = abs(a.iloc[[i]] - a.iloc[[i - 1]].values)
    tvr = tvr.dropna()
    tvr_cs = tvr.sum(axis=1)

    x=mean_sma(tvr_cs.to_frame(),250)
    tvr_col = pd.DataFrame(columns=x.columns, index=x.index)
    for i in range(0, 7):
        tvr_col.iloc[i, 0] = x.iloc[250 * i-1, 0]
    new=pd.DataFrame(columns=x.columns, index=x.index)
    new=mean_sma(tvr_cs.to_frame(),1968-250*7)
    tvr_col=tvr_col.dropna()
    tvr_col = pd.concat([tvr_col,new.iloc[[-1]]])
    tvr_col=tvr_col*100
    tvr_col=tvr_col.dropna()
    tvr_col['year'] = ['2010', '2011', '2012', '2013', '2014', '2015', '2016', '2017']
    tvr_col = tvr_col.set_index('year')
    tvr_col.columns=['Turnover']
    tvr_col=np.around(tvr_col.astype(np.double),2)
    # print(tvr_col)

    gross_pnl['HighValue'] = gross_pnl['profit'].cummax()
    gross_pnl['Drawdown'] = (gross_pnl['HighValue'] - gross_pnl['profit'])/(gross_pnl['HighValue']+b)
    mdd =pd.DataFrame()
    for i in range(1, 8):
        mdd = pd.concat([mdd, pd.DataFrame(data=[gross_pnl.iloc[250 * (i - 1):250 * i, 2].max(axis=0)])],ignore_index=True)

    mdd=pd.concat([mdd,pd.DataFrame(data=[gross_pnl.iloc[250*7:1967,2].max(axis=0)])])
    mdd = round(mdd*100, 2)
    # print(mdd)
    mdd['year'] = ['2010', '2011', '2012', '2013', '2014', '2015', '2016', '2017']
    mdd = mdd.set_index('year')
    mdd.columns=['Max Drawdown']
    mdd=np.around(mdd.astype(np.double),2)
    # gross_pnl['HighValue']
    # print(mdd)

    fit=(((yearly_ret/abs(yearly_ret))*(abs(yearly_ret)/tvr_col.values)**0.5)*abs(sharpe.values))
    # fit=round(fit,2)
    fit['year'] = ['2010', '2011', '2012', '2013', '2014', '2015', '2016', '2017']
    fit = fit.set_index('year')
    fit.columns=['Fitness']
    fit=np.around(fit.astype(np.double),2)

    stats=[ir,sharpe,tvr_col,mdd,yearly_ret,fit]
    stat=pd.concat(stats,axis=1)
    #print (tabulate(stat, headers='keys', tablefmt='psql'))
    headers=stat.columns

    data = [go.Scatter(x=gross_pnl.index, y=gross_pnl['profit'],fill='tonexty',
        mode='lines',
        line=dict(
            color='rgb(143, 19, 131)'))]
    layout = go.Layout(title=w,yaxis=dict(title='Gross PNL'),xaxis=dict(title='Dates'))
    fig = go.Figure(data=data, layout=layout)
    #py.offline.plot(fig, filename='backtested.html')

    header = stat.columns.tolist()
    header = ['Year'] + header
    Year = stat.index.tolist()
    IR = stat.iloc[:,0].values.tolist()
    Sharpe_Ratio = stat.iloc[:,1].values.tolist()
    Turnover = stat.iloc[:,2].values.tolist()
    Max_Drawdown = stat.iloc[:,3].values.tolist()
    Yearly_Returns = stat.iloc[:,4].values.tolist()
    Fitness = stat.iloc[:,5].values.tolist()
    values = [Year, IR, Sharpe_Ratio, Turnover, Max_Drawdown, Yearly_Returns, Fitness]

    data_for_x = gross_pnl.index.tolist()
    data_for_y = gross_pnl.iloc[:,0].values.tolist()

    coordinates = [data_for_x, data_for_y, values, values2, values3, header2, header3]

    return coordinates

    # rgb(131, 90, 241)
