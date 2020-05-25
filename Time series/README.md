

```python
import pandas as pd
import numpy as np
import random
import datetime # manipulating date formats
# Viz
import matplotlib.pyplot as plt # basic plotting
import seaborn as sns # for prettier plots
# Settings
import warnings
warnings.filterwarnings("ignore")
```


```python
# TIME SERIES
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from pandas.plotting import autocorrelation_plot
from statsmodels.tsa.stattools import adfuller, acf, pacf,arma_order_select_ic
import statsmodels.formula.api as smf
import statsmodels.tsa.api as smt
import statsmodels.api as sm
import scipy.stats as scs
```


```python
df = pd.DataFrame(np.random.randint(1,200,size=200), columns=['y'])
```


```python
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>y</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>111</td>
    </tr>
    <tr>
      <td>1</td>
      <td>137</td>
    </tr>
    <tr>
      <td>2</td>
      <td>67</td>
    </tr>
    <tr>
      <td>3</td>
      <td>122</td>
    </tr>
    <tr>
      <td>4</td>
      <td>173</td>
    </tr>
  </tbody>
</table>
</div>




```python
plt.figure(figsize=(20,6))
plt.plot(df)
plt.xlabel('Date')
plt.ylabel('Volume')
#plt.xticks(daily_traffic_1['Date'])
plt.legend()
plt.locator_params(nbins=10)
plt.show()
```

    No handles with labels found to put in legend.
    


![png](output_4_1.png)


## Time Series Exploration


```python
plt.figure(figsize=(16,6))
plt.plot(df.rolling(window=24,center=False).mean(),label='Rolling Mean')
plt.plot(df.rolling(window=24,center=False).std(),label='Rolling sd')
plt.legend()
```




    <matplotlib.legend.Legend at 0x19a925ede10>




![png](output_6_1.png)


There is an increasing trend.

### Decomposition

Multiplicative model


```python
import statsmodels.api as sm
res = sm.tsa.seasonal_decompose(df.values,freq=24,model="multiplicative")
fig = res.plot()
fig.show()
#plt.figure(figsize=(100,20))
#plt.plot(res)
```


![png](output_10_0.png)


Additive Model


```python
res = sm.tsa.seasonal_decompose(df,freq=24,model="additive")
#plt.figure(figsize=(16,12))
fig = res.plot()
#fig.show()
```


![png](output_12_0.png)


We check for stationarity using ADF test.


```python
def test_stationarity(timeseries):
    #Perform Dickey-Fuller test:
    print('Results of Dickey-Fuller Test:')
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    print (dfoutput)

test_stationarity(df['y'])
```

    Results of Dickey-Fuller Test:
    Test Statistic                -1.377582e+01
    p-value                        9.494716e-26
    #Lags Used                     0.000000e+00
    Number of Observations Used    1.990000e+02
    Critical Value (1%)           -3.463645e+00
    Critical Value (5%)           -2.876176e+00
    Critical Value (10%)          -2.574572e+00
    dtype: float64
    


```python
# to remove trend
from pandas import Series as Series
# create a differenced series
def difference(dataset, interval=1):
    diff = list()
    for i in range(interval, len(dataset)):
        value = dataset[i] - dataset[i - interval]
        diff.append(value)
    return Series(diff)

# invert differenced forecast
def inverse_difference(last_ob, value):
    return value + last_ob
```


```python
ts=df['y']
ts.astype('float')
plt.figure(figsize=(16,16))
plt.subplot(311)
plt.title('Original')
plt.xlabel('Time')
plt.ylabel('Traffic')
plt.plot(ts)
plt.subplot(312)
plt.title('After De-trend')
plt.xlabel('Time')
plt.ylabel('Traffic')
new_ts=difference(ts)
plt.plot(new_ts)
plt.plot()

plt.subplot(313)
plt.title('After De-seasonalization')
plt.xlabel('Time')
plt.ylabel('Traffic')
new_ts=difference(ts,24)       # assuming the seasonality is 24 units long
plt.plot(new_ts)
plt.plot()
```




    []




![png](output_16_1.png)



```python
# now testing the stationarity again after de-seasonality
test_stationarity(new_ts)
```

    Results of Dickey-Fuller Test:
    Test Statistic                -8.277322e+00
    p-value                        4.618864e-13
    #Lags Used                     1.000000e+00
    Number of Observations Used    1.740000e+02
    Critical Value (1%)           -3.468502e+00
    Critical Value (5%)           -2.878298e+00
    Critical Value (10%)          -2.575704e+00
    dtype: float64
    

Now after the transformations, our p-value for the DF test is well within 5 %. Hence we can assume Stationarity of the series

We can easily get back the original series using the inverse transform function that we have defined above.

Now let's dive into making the forecasts!

### AR, MA and ARMA models


```python
def tsplot(y, lags=None, figsize=(10, 8), style='bmh',title=''):
    if not isinstance(y, pd.Series):
        y = pd.Series(y)
    with plt.style.context(style):    
        fig = plt.figure(figsize=figsize)
        #mpl.rcParams['font.family'] = 'Ubuntu Mono'
        layout = (3, 2)
        ts_ax = plt.subplot2grid(layout, (0, 0), colspan=2)
        acf_ax = plt.subplot2grid(layout, (1, 0))
        pacf_ax = plt.subplot2grid(layout, (1, 1))
        qq_ax = plt.subplot2grid(layout, (2, 0))
        pp_ax = plt.subplot2grid(layout, (2, 1))
        
        y.plot(ax=ts_ax)
        ts_ax.set_title(title)
        smt.graphics.plot_acf(y, lags=lags, ax=acf_ax, alpha=0.5)
        smt.graphics.plot_pacf(y, lags=lags, ax=pacf_ax, alpha=0.5)
        sm.qqplot(y, line='s', ax=qq_ax)
        qq_ax.set_title('QQ Plot')        
        scs.probplot(y, sparams=(y.mean(), y.std()), plot=pp_ax)

        plt.tight_layout()
    return 
```


```python
#
# pick best order by aic 
# smallest aic value wins
best_aic = np.inf 
best_order = None
best_mdl = None

rng = range(5)
for i in rng:
    for j in rng:
        try:
            tmp_mdl = smt.ARMA(new_ts.values, order=(i, j)).fit(method='mle', trend='nc')
            tmp_aic = tmp_mdl.aic
            if tmp_aic < best_aic:
                best_aic = tmp_aic
                best_order = (i, j)
                best_mdl = tmp_mdl
        except: continue


print('aic: {:6.5f} | order: {}'.format(best_aic, best_order))
```

    aic: 2046.12516 | order: (3, 2)
    

We use the code below to predict


```python
best_mdl.predict()
```




    array([  0.        ,  -1.45612143,   4.85387981,  -9.50157879,
             8.53582054,   2.85693511, -16.74706403,   8.71291324,
            12.25765922, -11.65527375,  -1.83981837,   0.42511368,
            -1.83140088,   9.08912164,  -3.07716389, -11.29439101,
            -7.68101543,  23.61766539,   7.77236877, -20.09908974,
           -13.14619254,  23.42560992,   4.04599472,  -7.8930322 ,
            -0.57418014,  -7.62470005,   6.36254447,   5.01525245,
           -18.32117359,   8.6441418 ,  22.02900779, -16.33400869,
            -9.87215726,  -4.48821235,  20.92591603,   4.89891879,
           -20.58729311, -19.40632713,  35.34546651,  15.43458154,
           -27.76046834, -31.85584684,  52.80653073,  10.63437875,
           -50.64708895,  -0.59457442,  49.40092109, -10.91589267,
           -44.97369195,  16.58848695,  24.0900616 ,   5.50127964,
           -34.21309181,   5.04277467,  12.95307586,  10.44040523,
           -18.68553908,   6.28975212,   1.95330758,  -3.59204398,
           -11.53790199,  20.51572486,  -1.24119785, -19.89478245,
            13.52903784,  11.64855682, -26.38714493,  15.29759535,
             9.6837354 ,  -6.35160134, -19.18654134,   2.71502869,
            32.17325289, -16.24896306, -11.52601385,  -9.67798739,
            41.68722213, -23.41572278,  -9.50506014,   8.26237606,
            -2.37534003, -14.7717114 ,  19.34063886,   9.87010747,
           -16.5921554 ,  -0.99755455,   8.33209765,  16.06565684,
           -21.5447084 , -13.37320927,  33.00400083, -13.56789781,
            -5.6533358 ,   0.3134575 ,  -0.5374971 ,  23.13211241,
           -14.18493234, -18.08336504,   3.46193718,  28.88771832,
            -8.25930332, -18.56835437,   2.25707091,  18.61577878,
            -0.79943891,  -4.13082896,  -7.91941394,   3.43352261,
            20.22223339, -16.3464309 , -12.44066374,  11.30424212,
            21.65657243, -15.87653497, -24.82732018,  31.45000426,
             8.75086354, -35.30526301,   9.45626705,  10.99346354,
             4.00264078, -15.28321757,  -1.93278706,  14.75483431,
           -10.46617114, -10.96043492,   4.94080805,  18.34802046,
             4.05978707, -33.52925162,  -0.92076654,  29.44131869,
             3.47980132, -44.22648297,  -0.47483433,  50.25263022,
            -4.75347998, -32.06992517, -20.06667466,  35.3341624 ,
            14.87920434, -18.07143597, -18.51595758,  26.83666653,
             7.85041024, -25.69926035,  -9.19794839,  28.31425403,
             4.04273097, -13.63802303,  -6.63503767,  -6.86172249,
             8.24062599,  22.71738366, -27.04704035,   4.87405863,
            -3.19156188,  22.40720521,   3.77290338, -37.00863431,
            -3.80525401,  31.0875596 ,  16.21803308, -39.88320794,
             5.05431175,  17.68953843,   5.43356257, -27.31681542,
           -10.19823845,  36.22969846,  25.90468493, -51.92620081,
           -10.45575769,  32.47372312,  17.73581325, -22.40395374])




```python
d1 = pd.DataFrame(pd.date_range(start = '2015-11-01 00:00:00',end='2015-11-09 07:00:00', freq = 'H'), columns = ['index'])
```


```python
new_df = d1.join(df['y'], how='outer')
```


```python
new_df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>index</th>
      <th>y</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>2015-11-01 00:00:00</td>
      <td>111</td>
    </tr>
    <tr>
      <td>1</td>
      <td>2015-11-01 01:00:00</td>
      <td>137</td>
    </tr>
    <tr>
      <td>2</td>
      <td>2015-11-01 02:00:00</td>
      <td>67</td>
    </tr>
    <tr>
      <td>3</td>
      <td>2015-11-01 03:00:00</td>
      <td>122</td>
    </tr>
    <tr>
      <td>4</td>
      <td>2015-11-01 04:00:00</td>
      <td>173</td>
    </tr>
    <tr>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <td>195</td>
      <td>2015-11-09 03:00:00</td>
      <td>107</td>
    </tr>
    <tr>
      <td>196</td>
      <td>2015-11-09 04:00:00</td>
      <td>130</td>
    </tr>
    <tr>
      <td>197</td>
      <td>2015-11-09 05:00:00</td>
      <td>88</td>
    </tr>
    <tr>
      <td>198</td>
      <td>2015-11-09 06:00:00</td>
      <td>13</td>
    </tr>
    <tr>
      <td>199</td>
      <td>2015-11-09 07:00:00</td>
      <td>27</td>
    </tr>
  </tbody>
</table>
<p>200 rows × 2 columns</p>
</div>



### Prophet


```python
from fbprophet import Prophet
```

    Importing plotly failed. Interactive plots will not work.
    


```python
ts = new_df
ts.columns=['ds','y']
```


```python
model = Prophet(changepoint_prior_scale= 0.3, growth= 'linear',yearly_seasonality=True) #instantiate Prophet with only yearly seasonality as our data is monthly 
model.fit(ts) #fit the model with your dataframe
```

    INFO:fbprophet:Disabling weekly seasonality. Run prophet with weekly_seasonality=True to override this.
    




    <fbprophet.forecaster.Prophet at 0x19a943b7550>




```python
# 14592
# 2952
# predict for five months in the furure and MS - month start is the frequency
future = model.make_future_dataframe(periods = 168, freq = 'H')  
# now lets make the forecasts
forecast = model.predict(future)
forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(10)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ds</th>
      <th>yhat</th>
      <th>yhat_lower</th>
      <th>yhat_upper</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>358</td>
      <td>2015-11-15 22:00:00</td>
      <td>71.740763</td>
      <td>3.367405</td>
      <td>146.382386</td>
    </tr>
    <tr>
      <td>359</td>
      <td>2015-11-15 23:00:00</td>
      <td>62.911952</td>
      <td>-4.480026</td>
      <td>135.777413</td>
    </tr>
    <tr>
      <td>360</td>
      <td>2015-11-16 00:00:00</td>
      <td>40.407617</td>
      <td>-32.497957</td>
      <td>110.726437</td>
    </tr>
    <tr>
      <td>361</td>
      <td>2015-11-16 01:00:00</td>
      <td>23.114730</td>
      <td>-51.982060</td>
      <td>92.240202</td>
    </tr>
    <tr>
      <td>362</td>
      <td>2015-11-16 02:00:00</td>
      <td>26.704959</td>
      <td>-46.894534</td>
      <td>99.753406</td>
    </tr>
    <tr>
      <td>363</td>
      <td>2015-11-16 03:00:00</td>
      <td>48.506080</td>
      <td>-20.186731</td>
      <td>123.488134</td>
    </tr>
    <tr>
      <td>364</td>
      <td>2015-11-16 04:00:00</td>
      <td>68.950336</td>
      <td>-5.266659</td>
      <td>141.446800</td>
    </tr>
    <tr>
      <td>365</td>
      <td>2015-11-16 05:00:00</td>
      <td>68.902948</td>
      <td>-3.257100</td>
      <td>146.933161</td>
    </tr>
    <tr>
      <td>366</td>
      <td>2015-11-16 06:00:00</td>
      <td>46.931999</td>
      <td>-15.787223</td>
      <td>119.164111</td>
    </tr>
    <tr>
      <td>367</td>
      <td>2015-11-16 07:00:00</td>
      <td>20.448713</td>
      <td>-46.550703</td>
      <td>93.986534</td>
    </tr>
  </tbody>
</table>
</div>




```python
model.plot(forecast)
```




![png](output_33_0.png)




![png](output_33_1.png)



```python
model.plot_components(forecast)
```




![png](output_34_0.png)




![png](output_34_1.png)


## References

https://facebook.github.io/prophet/docs/quick_start.html#python-api

https://www.kaggle.com/jagangupta/time-series-basics-exploring-traditional-ts
