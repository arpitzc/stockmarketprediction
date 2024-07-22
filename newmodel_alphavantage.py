# Import libraries
!pip install yfinance
!pip install pandasql
!pip install quandl
import yfinance as yf
import pandas as pd
from datetime import datetime
import os
import pandas as pd
import pandasql
from pandasql import sqldf
from datetime import date
from datetime import timedelta
import datetime
import quandl
import os
from datetime import date
import requests
import numpy as np
import json

import yfinance as yf
import pandas as pd

stockdata = pd.read_csv('/Users/amritaagarwal/Downloads/Python_stocks/data/stockfulldata.csv')

df_bse = pd.read_csv('/Users/amritaagarwal/Downloads/Python_stocks/data/bse_stock_symbols.csv')

# Group by 'code', 'industry', 'security_name', and 'security_id' and compute the average total_turnover, min date, and max date
grouped = stockdata.groupby(['code', 'Industry', 'Security_Name', 'Security_Id'])
aggregated = grouped.agg({'Total_Turnover': 'mean', 'Date': ['min', 'max']})

# Reset the index and rename columns
aggregated.reset_index(inplace=True)
aggregated.columns = ['Security Code', 'Industry', 'Security Name', 'Security Id', 'marketcap', 'start_date', 'end_date']

# Sort the dataframe by marketcap in descending order
aggregated.sort_values(by='marketcap', ascending=False, inplace=True)

# Select the top 500 rows using head()
top_stocks = aggregated.head(500)

drop_cols=top_stocks["Security Code"]

del stockdata

#fetching security codes for equity instruments and those which are active
df_security_code=df_bse[(df_bse['Instrument']=='Equity') & (df_bse['Status']=='Active' )][["Security Code","Industry","Security Name","Security Id"]]
df_security_code=top_stocks

#securities=['OFSS','M&M','STLTECH','CADILAHC','M&MFIN','ENDURANCE','STAR','FLFL','TATASTLLP','TECHM','IOC','SPICEJET','CENTRUM','WOCKPHARMA','AARTIDRUGS','FRETAIL','GODREJIND','KARURVYSYA','APOLLOHOSP','IFCI','EDELWEISS','ENGINERSIN','SECURKLOUD','GPPL','ASHOKA','VIVIMEDLAB','DHANI','SANOFI','VENKYS','AARTIIND','BBTC','INDUSTOWER','ICIL','ZEEL','FORTIS','AMARAJABAT','RSSOFTWARE','PFS','THOMASCOOK','ONMOBILE','AUBANK','LICHSGFIN','NATIONALUM','HINDCOPPER','APOLLOTYRE','LAURUSLABS','DRREDDY','GODREJAGRO','INDIACEM','AUROPHARMA','OBEROIRLTY','GCMSECU','BEPL','MATRIMONY','KUSHAL','TATAMTRDVR','TIRUMALCHM','TATAMETALI','CYIENT','IBULHSGFIN','TATACHEM','TORNTPHARM','FSL','INTELLECT','ABBOTINDIA','GRANULES','NOCIL','ABAN','THYROCARE','GATI','TATACOMM','ITC','HCL-INSYS','MOTHERSUMI','TAKE','JKPAPER','CHAMBLFERT','EMAMILTD','MARKSANS','UNITECH','IDEA','MOREPENLAB','L&TFH','HDFCAMC','TCS','J&KBANK','BAJAJCON','HDFCLIFE','SMLISUZU','AEGISLOG','ICICIPRULI','JPPOWER','ABFRL']
#df_security_code = df_security_code[df_security_code["Security Id"].isin(securities)]
length=len(df_security_code)



if os.path.exists('/Users/amritaagarwal/Downloads/Python_stocks/data/final_df.csv'):
    os.remove('/Users/amritaagarwal/Downloads/Python_stocks/data/final_df.csv')
else:
    print("Can not delete the file as it doesn't exists")
    
final_df = pd.DataFrame(columns=['date','close','volume','security_id'])

#########################################################################################################################
#api_key='T5NWSO7MD1CN9CKG'
'https://www.alphavantage.co/query?function=EARNINGS&symbol=RELIANCE.BSE&apikey=IFQY9G05XABS7HKR'
api_key='IFQY9G05XABS7HKR'
i=0
for i in range(0,499):
    try:
        final_df = pd.DataFrame(columns=['date','close','volume','security_id'])
        ##################################################
        # replace the "demo" apikey below with your own key from https://www.alphavantage.co/support/#api-key
        #price data
        url = 'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY_ADJUSTED&symbol='+df_security_code["Security Id"].iloc[i]+'.BSE&outputsize=full&apikey='+api_key
        r = requests.get(url)
        # Convert the response to a JSON object
        data = r.json()
        # Get the time series data from the JSON object
        time_series_data = data["Time Series (Daily)"]
        # Convert the time series data to a DataFrame
        df = pd.DataFrame(time_series_data).transpose()
        # Convert the index to a datetime object
        #df.index = pd.to_datetime(df.index)
        
        final_df['close']=df['5. adjusted close']
        final_df['security_id']=df_security_code["Security Id"].iloc[i]
        final_df['date']=df.index
        final_df['volume']=df['6. volume']
        
        
        #RSI
        rsi_url = 'https://www.alphavantage.co/query?function=RSI&symbol='+df_security_code["Security Id"].iloc[i]+'.BSE&interval=daily&time_period=14&series_type=open&apikey='+api_key
        r = requests.get(rsi_url)
        data = r.json()
        time_series_data = data['Technical Analysis: RSI']
        df_rsi = pd.DataFrame(time_series_data).transpose()
        #final_df = final_df.merge(df_rsi, how='left', left_index=True, right_index=True)
        
        #SMA
        sma_url = 'https://www.alphavantage.co/query?function=SMA&symbol='+df_security_code["Security Id"].iloc[i]+'.BSE&interval=daily&time_period=44&series_type=open&apikey='+api_key
        r = requests.get(sma_url)
        data = r.json()
        time_series_data = data['Technical Analysis: SMA']
        df_sma = pd.DataFrame(time_series_data).transpose()
        #final_df = final_df.merge(df_sma, how='left', left_index=True, right_index=True)
        
        #MACD
        macd_url = 'https://www.alphavantage.co/query?function=MACD&symbol='+df_security_code["Security Id"].iloc[i]+'.BSE&interval=daily&series_type=open&apikey='+api_key
        r = requests.get(macd_url)
        data = r.json()
        time_series_data = data['Technical Analysis: MACD']
        df_macd = pd.DataFrame(time_series_data).transpose()
        #final_df = final_df.merge(df_macd, how='left', left_index=True, right_index=True)
        
        #OBV
        obv_url = 'https://www.alphavantage.co/query?function=OBV&symbol='+df_security_code["Security Id"].iloc[i]+'.BSE&interval=daily&apikey='+api_key
        r = requests.get(obv_url)
        data = r.json()
        time_series_data = data['Technical Analysis: OBV']
        df_obv = pd.DataFrame(time_series_data).transpose()
        #final_df = final_df.merge(df_obv, how='left', left_index=True, right_index=True)
        
        #ATR
        atr_url = 'https://www.alphavantage.co/query?function=ATR&symbol='+df_security_code["Security Id"].iloc[i]+'.BSE&interval=daily&time_period=14&apikey='+api_key
        r = requests.get(atr_url)
        data = r.json()
        time_series_data = data['Technical Analysis: ATR']
        df_atr = pd.DataFrame(time_series_data).transpose()
        #final_df = final_df.merge(df_atr, how='left', left_index=True, right_index=True)
        
        #ADX
        adx_url = 'https://www.alphavantage.co/query?function=ADX&symbol='+df_security_code["Security Id"].iloc[i]+'.BSE&interval=daily&time_period=10&apikey='+api_key
        r = requests.get(adx_url)
        data = r.json()
        time_series_data = data['Technical Analysis: ADX']
        df_ad = pd.DataFrame(time_series_data).transpose()
        #final_df = final_df.merge(df_ad, how='left', left_index=True, right_index=True)
        
        #AROON
        aroon_url = 'https://www.alphavantage.co/query?function=AROON&symbol='+df_security_code["Security Id"].iloc[i]+'.BSE&interval=daily&time_period=14&apikey='+api_key
        r = requests.get(aroon_url)
        data = r.json()
        time_series_data = data['Technical Analysis: AROON']
        df_aroon = pd.DataFrame(time_series_data).transpose()
        #final_df = final_df.merge(df_aroon, how='left', left_index=True, right_index=True)
        
        #AROONOSC
        aroonosc_url='https://www.alphavantage.co/query?function=AROONOSC&symbol='+df_security_code["Security Id"].iloc[i]+'.BSE&interval=daily&time_period=44&apikey='+api_key
        r = requests.get(aroonosc_url)
        data = r.json()
        time_series_data = data['Technical Analysis: AROONOSC']
        df_aroonosc = pd.DataFrame(time_series_data).transpose()
        #final_df = final_df.merge(df_ema, how='left', left_index=True, right_index=True)
   
    
        #ADOSC
        adosc_url='https://www.alphavantage.co/query?function=ADOSC&symbol='+df_security_code["Security Id"].iloc[i]+'.BSE&interval=daily&fastperiod=44&apikey='+api_key
        r = requests.get(adosc_url)
        data = r.json()
        time_series_data = data['Technical Analysis: ADOSC']
        df_adosc = pd.DataFrame(time_series_data).transpose()
        #final_df = final_df.merge(df_ema, how='left', left_index=True, right_index=True)
        
        #HT_DCPERIOD
        ht_dcperiod_url='https://www.alphavantage.co/query?function=HT_DCPERIOD&symbol='+df_security_code["Security Id"].iloc[i]+'.BSE&interval=daily&series_type=close&apikey='+api_key
        r = requests.get(ht_dcperiod_url)
        data = r.json()
        time_series_data = data['Technical Analysis: HT_DCPERIOD']
        df_ht_dcperiod = pd.DataFrame(time_series_data).transpose()
        #final_df = final_df.merge(df_ema, how='left', left_index=True, right_index=True)
   
 

        final_df = final_df.join([df_rsi, df_sma,df_macd,df_obv,df_atr,df_ad,df_aroon,
                                  df_ht_dcperiod,df_adosc,df_aroonosc], how='left')
       
        #final_df=final_df[final_df.index>'2005-01-01']

        
        final_df.to_csv(r'//Users/amritaagarwal/Downloads/Python_stocks/data/final_df.csv', mode='a',index=False, header=True)
        print(i)
        print(df_security_code["Security Id"].iloc[i])
    except:    
        print('exception')
        pass;

api_key='IFQY9G05XABS7HKR'   

final_df=pd.read_csv(r'/Users/amritaagarwal/Downloads/Python_stocks/data/final_df.csv')

#final_df2=pd.read_csv(r'/Users/amritaagarwal/Downloads/Python_stocks/data/final_df copy 2.csv')


#final_df = pd.concat([final_df, final_df2], axis=0)


#add industry column
final_df = pd.merge(final_df, df_bse[['Security Id', 'Industry']], left_on='security_id', right_on='Security Id', how='left')
final_df.columns

#add market cap column

final_df = pd.merge(final_df, df_bse[['Security Id', 'ISIN No']], on='Security Id', how='left')

final_df.columns

marketcap=pd.read_csv(r'/Users/amritaagarwal/Downloads/Python_stocks/data/security_marketcap.csv')

import pandas as pd

# Assuming final_df and marketcap are your DataFrames
# First merge attempt on 'ISIN No'
#first_merge = pd.merge(final_df, marketcap[['ISIN No', 'marketcap']], on='ISIN No', how='left')

# Identify rows where 'marketcap' did not match
#unmatched = first_merge[pd.isnull(first_merge['marketcap'])]

# Prepare the second DataFrame for merge, renaming columns for the merge
#marketcap_renamed = marketcap.rename(columns={'symbol': 'security_id'})

# Second merge attempt for unmatched rows, now on 'security_id'
#second_merge = pd.merge(unmatched.drop(columns='marketcap'), marketcap_renamed[['security_id', 'marketcap']], on='security_id', how='left')

# Combine matched rows from the first merge with matched rows from the second merge
# Since unmatched now includes 'marketcap' from the second merge, we update the original first_merge DataFrame
#first_merge.update(second_merge)

# The final DataFrame with both sets of matched rows
#final_df = first_merge

final_df.dtypes

final_df = pd.merge(final_df, marketcap[['ISIN No' ,'marketcap']], on='ISIN No', how='left')
final_df['marketcap'].value_counts()

# Download sensex data 
symbol = "^BSESN"

# Download data
sensex_data = yf.download(symbol, start="2005-01-01", end=datetime.datetime.now())

# delete yesterday file if exists
if os.path.exists('/Users/amritaagarwal/Downloads/Python_stocks/data/sensex.csv'):
    os.remove('/Users/amritaagarwal/Downloads/Python_stocks/data/sensex.csv')
else:
    print("Can not delete the file as it doesn't exists")
    
sensex_data.to_csv('/Users/amritaagarwal/Downloads/Python_stocks/data/sensex.csv')

sensex_data = sensex_data.reset_index()
sensex_data = sensex_data.rename(columns={'Date': 'date'})
sensex_data = sensex_data[['date', 'Adj Close']]
sensex_data = sensex_data.rename(columns={'Adj Close': 'sensex_close'})



sensex_data['sensex_close_change'] = 0.0
sensex_data = sensex_data.sort_values('date',ascending=True)
sensex_data['avg_close_60'] = sensex_data['sensex_close'].rolling(window=60, min_periods=1).mean()
sensex_data['avg_close_7'] = sensex_data['sensex_close'].rolling(window=7, min_periods=1).mean()
sensex_data['sensex_close_change'] = ((sensex_data['avg_close_60'] / sensex_data['avg_close_7']) - 1)*100.0

mmi_df=pd.read_csv(r'/Users/amritaagarwal/Downloads/Python_stocks/data/mmi.csv')
mmi_df.columns
mmi_df = mmi_df.drop_duplicates()
mmi_df['Date'] = pd.to_datetime(mmi_df['Date'],format='%d/%m/%Y', errors='coerce')


stock_data = final_df[final_df['date'] != 'date']
stock_data.shape


if os.path.exists('/Users/amritaagarwal/Downloads/Python_stocks/data/stock_data.csv'):
    os.remove('/Users/amritaagarwal/Downloads/Python_stocks/data/stock_data.csv')
else:
    print("Can not delete the file as it doesn't exists")

final_df.columns
cols=['close','RSI', 'MACD', 'MACD_Signal', 'MACD_Hist', 'OBV', 'ATR',
       'Aroon Down', 'Aroon Up', 'ADX', 'AROONOSC', 'ADOSC',
       'DCPERIOD', 'SMA']
#cols = ['close','RSI','SMA','MACD','MACD_Signal','MACD_Hist','OBV','ATR','ADX','Aroon Down','Aroon Up','EMA','SlowK','SlowD','PHASE','QUADRATURE','HT_DCPHASE','DCPERIOD','TRENDMODE','SINE','LEAD SINE','HT_TRENDLINE','ADOSC','NATR','TRANGE','SAR','MIDPRICE','MIDPOINT','Real Upper Band','Real Middle Band','Real Lower Band','PLUS_DM','MINUS_DM','PLUS_DI','MINUS_DI','DX','ULTOSC','TRIX','MFI','AROONOSC','ROCR','ROC','CMO','WMA','DEMA','TEMA','TRIMA','KAMA','MAMA','FAMA','T3','FastK','FastD','WILLR','ADXR','APO','PPO','MOM','BOP','CCI']

# convert columns to float
for col in cols:
    stock_data[col] = pd.to_numeric(stock_data[col], errors='coerce')
    
#stock_data[cols] = stock_data[cols].astype(float)
stock_data.dtypes


# Assuming df is your DataFrame with columns: 'security_id', 'industry', 'date', and 'close'

# Convert 'date' column to datetime if it's not already
stock_data['date'] = pd.to_datetime(stock_data['date'])

# Ensure the DataFrame is sorted by 'security_id' and 'date'
stock_data = stock_data.sort_values(by=['security_id', 'date'])

# Calculate the close price from a year ago for each security
stock_data['close_1_year_ago'] = stock_data.groupby('security_id')['close'].shift(252) # Approximating 252 trading days in a year

# Calculate the growth rate for each security over the last year
stock_data['yearly_growth_rate'] = (stock_data['close'] - stock_data['close_1_year_ago']) / stock_data['close_1_year_ago']

# Group by both 'Industry' and 'date' to calculate the average growth rate for each industry on each date
industry_daily_growth_rates = stock_data.groupby(['Industry', 'date'])['yearly_growth_rate'].mean().reset_index(name='Industry_average_growth_rate')
marketcap_daily_growth_rates = stock_data.groupby(['marketcap', 'date'])['yearly_growth_rate'].mean().reset_index(name='Marketcap_average_growth_rate')


# Merge the average industry growth rate back into the original DataFrame
stock_data = stock_data.merge(industry_daily_growth_rates, on=['Industry', 'date'], how='left')
stock_data = stock_data.merge(marketcap_daily_growth_rates, on=['marketcap', 'date'], how='left')

stock_data.dtypes
def next_2month_10percent(df):
    df['next_2month_10percent'] = 0
    for security_id, group in df.groupby('security_id'):
        group = group.sort_values('date',ascending=False)
        group['avg_close_60'] = group['close'].rolling(window=60, min_periods=1).mean()
        group['min_close_60'] = group['close'].rolling(window=60, min_periods=1).min()
        group['next_2month_10percent'] = (((group['avg_close_60'] / group['close']) - 1) >= 0.1) & (((group['min_close_60'] / group['close']) - 1) >= -0.05)
        df.loc[group.index, 'next_2month_10percent'] = group['next_2month_10percent'].astype(int)
    return df

def next_2month_5percentdown(df):
    df['next_2month_5percentdown'] = 0
    for security_id, group in df.groupby('security_id'):
        group = group.sort_values('date',ascending=False)
        group['avg_close_60'] = group['close'].rolling(window=60, min_periods=1).mean()
        group['max_close_60'] = group['close'].rolling(window=60, min_periods=1).max()
        group['next_2month_5percentdown'] = (((group['avg_close_60'] / group['close']) - 1) <= -0.05) & (((group['max_close_60'] / group['close']) - 1) <= 0.05)
        df.loc[group.index, 'next_2month_5percentdown'] = group['next_2month_5percentdown'].astype(int)
    return df


#close change

def close_change(df):
    df['close_change'] = 0
    for security_id, group in df.groupby('security_id'):
        group = group.sort_values('date',ascending=True)
        group['avg_close'] = group['close'].rolling(window=14, min_periods=1).mean()
        group['close_change'] = ((group['close'] / group['avg_close']) - 1)*100.0
        df.loc[group.index, 'close_change'] = group['close_change'].astype(int)
    return df

def close_change_60(df):
    df['close_change_60'] = 0
    for security_id, group in df.groupby('security_id'):
        group = group.sort_values('date',ascending=True)
        group['avg_close_60'] = group['close'].rolling(window=60, min_periods=1).mean()
        group['avg_close_7'] = group['close'].rolling(window=7, min_periods=1).mean()
        group['close_change_60'] = ((group['avg_close_7'] / group['avg_close_60']) - 1)*100.0
        df.loc[group.index, 'close_change_60'] = group['close_change_60'].astype(int)
    return df

def obv_change(df):
    df['obv_change'] = 0
    for security_id, group in df.groupby('security_id'):
        group = group.sort_values('date',ascending=True)
        group['avg_obv'] = group['OBV'].rolling(window=14, min_periods=1).mean()
        group['obv_change'] = ((group['OBV'] / group['avg_obv']) - 1)*100.0
        group['obv_change'] = group['obv_change'].apply(lambda x: float(x) if np.isfinite(x) else np.nan)
        df.loc[group.index, 'obv_change'] = group['obv_change']
    return df

def rsi_change(df):
    df['rsi_change'] = 0
    for security_id, group in df.groupby('security_id'):
        group = group.sort_values('date',ascending=True)
        group['avg_rsi'] = group['RSI'].rolling(window=14, min_periods=1).mean()
        group['rsi_change'] = ((group['RSI'] / group['avg_rsi']) - 1)*100.0
        group['rsi_change'] = group['rsi_change'].apply(lambda x: int(x) if np.isfinite(x) else np.nan)
        df.loc[group.index, 'rsi_change'] = group['rsi_change']
    return df

def rsi_change_60(df):
    df['rsi_change_60'] = 0
    for security_id, group in df.groupby('security_id'):
        group = group.sort_values('date',ascending=True)
        group['avg_rsi'] = group['RSI'].rolling(window=60, min_periods=1).mean()
        group['avg_rsi_7'] = group['RSI'].rolling(window=7, min_periods=1).mean()
        group['rsi_change_60'] = ((group['avg_rsi_7'] / group['avg_rsi']) - 1)*100.0
        group['rsi_change_60'] = group['rsi_change_60'].apply(lambda x: int(x) if np.isfinite(x) else np.nan)
        df.loc[group.index, 'rsi_change_60'] = group['rsi_change_60']
    return df

def atr_change(df):
    df['atr_change'] = 0
    for security_id, group in df.groupby('security_id'):
        group = group.sort_values('date',ascending=True)
        group['avg_atr'] = group['ATR'].rolling(window=14, min_periods=1).mean()
        group['atr_change'] = ((group['ATR'] / group['avg_atr']) - 1)*100.0
        group['atr_change'] = group['atr_change'].apply(lambda x: int(x) if np.isfinite(x) else np.nan)
        df.loc[group.index, 'atr_change'] = group['atr_change']
    return df

def Aroon_Down_change(df):
    df['Aroon_Down_change'] = 0
    for security_id, group in df.groupby('security_id'):
        group = group.sort_values('date',ascending=True)
        group['avg_aroon_down'] = group['Aroon Down'].rolling(window=14, min_periods=1).mean()
        group['Aroon_Down_change'] = ((group['Aroon Down'] / group['avg_aroon_down']) - 1)*100.0
        group['Aroon_Down_change'] = group['Aroon_Down_change'].apply(lambda x: int(x) if np.isfinite(x) else np.nan)
        df.loc[group.index, 'Aroon_Down_change'] = group['Aroon_Down_change']
    return df

def Aroon_Up_change(df):
    df['Aroon_Up_change'] = 0
    for security_id, group in df.groupby('security_id'):
        group = group.sort_values('date',ascending=True)
        group['avg_aroon_up'] = group['Aroon Up'].rolling(window=14, min_periods=1).mean()
        group['Aroon_Up_change'] = ((group['Aroon Up'] / group['avg_aroon_up']) - 1)*100.0
        group['Aroon_Up_change'] = group['Aroon_Up_change'].apply(lambda x: int(x) if np.isfinite(x) else np.nan)
        df.loc[group.index, 'Aroon_Up_change'] = group['Aroon_Up_change']
    return df
def PHASE_change(df):
    df['PHASE_change'] = 0
    for security_id, group in df.groupby('security_id'):
        group = group.sort_values('date',ascending=True)
        group['avg_PHASE'] = group['PHASE'].rolling(window=14, min_periods=1).mean()
        group['PHASE_change'] = ((group['PHASE'] / group['avg_PHASE']) - 1)*100.0
        group['PHASE_change'] = group['PHASE_change'].apply(lambda x: int(x) if np.isfinite(x) else np.nan)
        df.loc[group.index, 'PHASE_change'] = group['PHASE_change']
    return df

def QUADRATURE_change(df):
    df['QUADRATURE_change'] = 0
    for security_id, group in df.groupby('security_id'):
        group = group.sort_values('date',ascending=True)
        group['avg_QUADRATURE'] = group['QUADRATURE'].rolling(window=14, min_periods=1).mean()
        group['QUADRATURE_change'] = ((group['QUADRATURE'] / group['avg_QUADRATURE']) - 1)*100.0
        group['QUADRATURE_change'] = group['QUADRATURE_change'].apply(lambda x: int(x) if np.isfinite(x) else np.nan)
        df.loc[group.index, 'QUADRATURE_change'] = group['QUADRATURE_change']
    return df
def HT_DCPHASE_change(df):
    df['HT_DCPHASE_change'] = 0
    for security_id, group in df.groupby('security_id'):
        group = group.sort_values('date',ascending=True)
        group['avg_HT_DCPHASE'] = group['HT_DCPHASE'].rolling(window=14, min_periods=1).mean()
        group['HT_DCPHASE_change'] = ((group['HT_DCPHASE'] / group['avg_HT_DCPHASE']) - 1)*100.0
        group['HT_DCPHASE_change'] = group['HT_DCPHASE_change'].apply(lambda x: int(x) if np.isfinite(x) else np.nan)
        df.loc[group.index, 'HT_DCPHASE_change'] = group['HT_DCPHASE_change']
    return df
def DCPERIOD_change(df):
    df['DCPERIOD_change'] = 0
    for security_id, group in df.groupby('security_id'):
        group = group.sort_values('date',ascending=True)
        group['avg_DCPERIOD'] = group['DCPERIOD'].rolling(window=14, min_periods=1).mean()
        group['DCPERIOD_change'] = ((group['DCPERIOD'] / group['avg_DCPERIOD']) - 1)*100.0
        group['DCPERIOD_change'] = group['DCPERIOD_change'].apply(lambda x: int(x) if np.isfinite(x) else np.nan)
        df.loc[group.index, 'DCPERIOD_change'] = group['DCPERIOD_change']
    return df
def SINE_change(df):
    df['SINE_change'] = 0
    for security_id, group in df.groupby('security_id'):
        group = group.sort_values('date',ascending=True)
        group['avg_SINE'] = group['SINE'].rolling(window=14, min_periods=1).mean()
        group['SINE_change'] = ((group['SINE'] / group['avg_SINE']) - 1)*100.0
        group['SINE_change'] = group['SINE_change'].apply(lambda x: int(x) if np.isfinite(x) else np.nan)
        df.loc[group.index, 'SINE_change'] = group['SINE_change']
    return df
def LEAD_SINE_change(df):
    df['LEAD_SINE_change'] = 0
    for security_id, group in df.groupby('security_id'):
        group = group.sort_values('date',ascending=True)
        group['avg_LEAD_SINE'] = group['LEAD SINE'].rolling(window=14, min_periods=1).mean()
        group['LEAD_SINE_change'] = ((group['LEAD SINE'] / group['avg_LEAD_SINE']) - 1)*100.0
        group['LEAD_SINE_change'] = group['LEAD_SINE_change'].apply(lambda x: int(x) if np.isfinite(x) else np.nan)
        df.loc[group.index, 'LEAD_SINE_change'] = group['LEAD_SINE_change']
    return df
def HT_TRENDLINE_change(df):
    df['HT_TRENDLINE_change'] = 0
    for security_id, group in df.groupby('security_id'):
        group = group.sort_values('date',ascending=True)
        group['avg_HT_TRENDLINE'] = group['HT_TRENDLINE'].rolling(window=14, min_periods=1).mean()
        group['HT_TRENDLINE_change'] = ((group['HT_TRENDLINE'] / group['avg_HT_TRENDLINE']) - 1)*100.0
        group['HT_TRENDLINE_change'] = group['HT_TRENDLINE_change'].apply(lambda x: int(x) if np.isfinite(x) else np.nan)
        df.loc[group.index, 'HT_TRENDLINE_change'] = group['HT_TRENDLINE_change']
    return df

def ADOSC_change(df):
    df['ADOSC_change'] = 0
    for security_id, group in df.groupby('security_id'):
        group = group.sort_values('date',ascending=True)
        group['avg_ADOSC'] = group['ADOSC'].rolling(window=14, min_periods=1).mean()
        group['ADOSC_change'] = ((group['ADOSC'] / group['avg_ADOSC']) - 1)*100.0
        group['ADOSC_change'] = group['ADOSC_change'].apply(lambda x: int(x) if np.isfinite(x) else np.nan)
        df.loc[group.index, 'ADOSC_change'] = group['ADOSC_change']
    return df

def NATR_change(df):
    df['NATR_change'] = 0
    for security_id, group in df.groupby('security_id'):
        group = group.sort_values('date',ascending=True)
        group['avg_NATR'] = group['NATR'].rolling(window=14, min_periods=1).mean()
        group['NATR_change'] = ((group['NATR'] / group['avg_NATR']) - 1)*100.0
        group['NATR_change'] = group['NATR_change'].apply(lambda x: int(x) if np.isfinite(x) else np.nan)
        df.loc[group.index, 'NATR_change'] = group['NATR_change']
    return df
def TRANGE_change(df):
    df['TRANGE_change'] = 0
    for security_id, group in df.groupby('security_id'):
        group = group.sort_values('date',ascending=True)
        group['avg_TRANGE'] = group['TRANGE'].rolling(window=14, min_periods=1).mean()
        group['TRANGE_change'] = ((group['TRANGE'] / group['avg_TRANGE']) - 1)*100.0
        group['TRANGE_change'] = group['TRANGE_change'].apply(lambda x: int(x) if np.isfinite(x) else np.nan)
        df.loc[group.index, 'TRANGE_change'] = group['TRANGE_change']
    return df

summary = stock_data.describe().apply(lambda x: x.apply(lambda y: y[0] if isinstance(y, tuple) else y))
# Calculate the number of NaN values for each column and append it to the summary
summary.loc['num_nulls'] = stock_data.isnull().sum()

# Calculate the number of inf and -inf values for each column and append it to the summary
summary.loc['num_infs'] = stock_data.replace([np.inf, -np.inf], np.nan).isnull().sum() - summary.loc['num_nulls']

# Print the summary table
print(summary)
summary.to_csv('summary.csv')

stock_data.to_csv('test.csv')
stock_data.shape
stock_data = stock_data.dropna(subset=['close'])
stock_data['date'] = pd.to_datetime(stock_data['date'])

#stock_data.dropna(inplace=True)
stock_data.shape

stock_data = obv_change(stock_data)
stock_data = next_2month_10percent(stock_data)
stock_data = next_2month_5percentdown(stock_data)
#stock_data = next_2month_beatmarket(stock_data)

stock_data = close_change(stock_data)
stock_data = close_change_60(stock_data)
stock_data = rsi_change(stock_data)
stock_data = rsi_change_60(stock_data)
stock_data = atr_change(stock_data)
stock_data = Aroon_Down_change(stock_data)
stock_data = Aroon_Up_change(stock_data)
stock_data = DCPERIOD_change(stock_data)
stock_data = ADOSC_change(stock_data)


#stock_data = PHASE_change(stock_data)
#stock_data['PHASE_binary'] = np.where(stock_data['PHASE'] > 0, 1, 0)
#stock_data['QUADRATURE_binary'] = np.where(stock_data['QUADRATURE'] > 0, 1, 0)
#stock_data = QUADRATURE_change(stock_data)
#stock_data = HT_DCPHASE_change(stock_data)
#stock_data = SINE_change(stock_data)
#stock_data = LEAD_SINE_change(stock_data)
#stock_data = HT_TRENDLINE_change(stock_data)
#stock_data = NATR_change(stock_data)
#stock_data = TRANGE_change(stock_data)

stock_data.to_csv(r'/Users/amritaagarwal/Downloads/Python_stocks/data/stock_data.csv',index=False,header=True)
stock_data=pd.read_csv(r'/Users/amritaagarwal/Downloads/Python_stocks/data/stock_data.csv')
stock_data['date'] = pd.to_datetime(stock_data['date'])

############### feature engineering ####################

def divide_into_buckets(stock_data, lower_bound, upper_bound):
    step = (upper_bound - lower_bound) / 3
    bins = np.arange(lower_bound, upper_bound + step, step)
    buckets = np.digitize(stock_data, bins)
    return buckets, bins


def calculate_ratio(f2, b2):
    try:
        return (f2/b2) - 1
    except:
        return 0

# rsi buckets:10,90
buckets, bins = divide_into_buckets(stock_data["RSI"], 20,80)
stock_data["rsi_buckets"] = buckets

# rsi 60 buckets:-30,30
buckets, bins = divide_into_buckets(stock_data["rsi_change_60"], -40,40)
stock_data["rsi_buckets_60"] = buckets

#macd hist
buckets, bins = divide_into_buckets(stock_data["MACD_Hist"], -2,2)
stock_data["macd_buckets"] = buckets

#atr buckets
buckets, bins = divide_into_buckets(calculate_ratio(stock_data["ATR"],stock_data["close"]), -0.975,-0.875)
stock_data["atr_buckets"] = buckets

#adx buckets 0,100,20,40
buckets, bins = divide_into_buckets(stock_data["ADX"], 0,100)
stock_data["adx_buckets"] = buckets

#sma buckets -.3,.5   -.2,.3
buckets, bins = divide_into_buckets(calculate_ratio(stock_data["SMA"],stock_data["close"]), -.2,.3)
stock_data["sma_buckets"] = buckets


#close buckets -40,40
buckets, bins = divide_into_buckets(stock_data["close_change"],-40,40)
stock_data["close_buckets"] = buckets

#close 60 buckets -40,40
buckets, bins = divide_into_buckets(stock_data["close_change_60"],-40,40)
stock_data["close_buckets_60"] = buckets

#obv buckets -40,10
buckets, bins = divide_into_buckets(stock_data["obv_change"],-40,10)
stock_data["obv_buckets"] = buckets

#rsi buckets -40,40 change -30,30
buckets, bins = divide_into_buckets(stock_data["rsi_change"],-20,20)
stock_data["rsi_change_buckets"] = buckets

#atr buckets -30,30 change -20,20
buckets, bins = divide_into_buckets(stock_data["atr_change"],-20,20)
stock_data["atr_change_buckets"] = buckets

#aroon buckets down 
buckets, bins = divide_into_buckets(stock_data["Aroon_Down_change"],-20,20)
stock_data["aroon_down_change_buckets"] = buckets

#aroon buckets up 
buckets, bins = divide_into_buckets(stock_data["Aroon_Up_change"],-20,20)
stock_data["aroon_up_change_buckets"] = buckets

#sensexclose 60 buckets -40,40
buckets, bins = divide_into_buckets(sensex_data["sensex_close_change"],-20,20)
sensex_data["sensex_close_change_buckets"] = buckets

#ADOSC
buckets, bins = divide_into_buckets(stock_data["ADOSC"],-1000000,1000000)
stock_data["ADOSC_buckets"] = buckets

#ADOSC_change
buckets, bins = divide_into_buckets(stock_data["ADOSC_change"],-500,500)
stock_data["ADOSC_change_buckets"] = buckets

#DCPERIOD_change
buckets, bins = divide_into_buckets(stock_data["DCPERIOD_change"],-20,20)
stock_data["DCPERIOD_change_buckets"] = buckets



# fund rate US
url = 'https://www.alphavantage.co/query?function=FEDERAL_FUNDS_RATE&interval=daily&apikey='+api_key
r = requests.get(url)
data = r.json()

time_series_data = data['data']
df_fundrate = pd.DataFrame(time_series_data)

df_fundrate['date'] = pd.to_datetime(df_fundrate['date'])
df_fundrate['date'] = df_fundrate['date'].dt.strftime('%Y-%m-%d')

df_fundrate['fundrate_change'] = 0.0
df_fundrate = df_fundrate.sort_values('date',ascending=True)
df_fundrate['avg_fundrate'] = df_fundrate['value'].rolling(window=60, min_periods=1).mean()
df_fundrate['fundrate_change'] = ((df_fundrate['value'].astype(float) / df_fundrate['avg_fundrate']) - 1)*100.0
df_fundrate['fundrate_change'] = df_fundrate['fundrate_change'].apply(lambda x: int(x) if np.isfinite(x) else np.nan)

#fundrate 60 buckets -40,40
buckets, bins = divide_into_buckets(df_fundrate["fundrate_change"],-10,10)
df_fundrate["fundrate_change_buckets"] = buckets

#merge with sensex data
stock_data['date'] = pd.to_datetime(stock_data['date'])
stock_data['date'] = stock_data['date'].dt.strftime('%Y-%m-%d')

sensex_data['date'] = pd.to_datetime(sensex_data['date'])
sensex_data['date'] = sensex_data['date'].dt.strftime('%Y-%m-%d')

mmi_df['Date'] = pd.to_datetime(mmi_df['Date'])
mmi_df['date'] = mmi_df['Date'].dt.strftime('%Y-%m-%d')

# Function to divide a value into buckets of 20
def divide_into_buckets(value):
    return (value // 20) * 20

# Apply the function to the column and create a new column "bucket"
mmi_df['mmi_bucket'] = mmi_df['Market Mood Index'].apply(divide_into_buckets)
mmi_df.rename(columns={'Market Mood Index': 'mmi'}, inplace=True)
mmi_df.to_csv('mmitest.csv')


stock_data1 = pd.merge(stock_data, sensex_data[['date','sensex_close','sensex_close_change_buckets']], how="inner", left_on=['date'], right_on=['date']);
stock_data2 = pd.merge(stock_data1, mmi_df[['date','mmi_bucket','mmi']], how="left", left_on=['date'], right_on=['date']);
stock_data['date'].head()
sensex_data.head()
stock_data3 = pd.merge(stock_data2, df_fundrate[['date','fundrate_change_buckets','value']], how="left", left_on=['date'], right_on=['date']);
stock_data3['date'].max()

#model_data=stock_data3[['date', 'close', 'volume', 'security_id', 'RSI', 'SMA', 'MACD','MACD_Signal', 'MACD_Hist', 'OBV', 'ATR', 'ADX', 'Aroon Down','Aroon Up', 'EMA', 'SlowK', 'SlowD',  'rsi_buckets', 'macd_buckets','atr_buckets', 'adx_buckets', 'sma_buckets', 'ema_buckets', 'stoch_buckets','close_buckets','obv_buckets','rsi_change_buckets','atr_change_buckets','close_buckets_60','rsi_buckets_60','obv_change',
#'next_2month_10percent', 'next_2month_5percentdown', 'close_change','sensex_close_change_buckets','sensex_close',
#'close_change_60', 'rsi_change', 'rsi_change_60', 'atr_change','Aroon_Down_change','Aroon_Up_change','mmi_bucket','fundrate_change_buckets',
#'PHASE_binary','QUADRATURE_binary','TRENDMODE','TRANGE_change_buckets','NATR_change_buckets','ADOSC_change_buckets','ADOSC_buckets','HT_TRENDLINE_change_buckets','LEAD_SINE_change_buckets','SINE_change_buckets','DCPERIOD_change_buckets','PHASE_change_buckets','HT_DCPHASE_change_buckets']]
stock_data3.columns
model_data=stock_data3[['date', 'close', 'volume', 'security_id', 'RSI', 'SMA', 'MACD','MACD_Signal', 'MACD_Hist', 'OBV', 'ATR', 'ADX', 'Aroon Down','Aroon Up', 'rsi_buckets', 'macd_buckets','atr_buckets', 'adx_buckets', 'sma_buckets', 'close_buckets','obv_buckets','rsi_change_buckets','atr_change_buckets','close_buckets_60','rsi_buckets_60','obv_change',
'next_2month_10percent', 'next_2month_5percentdown', 'close_change','sensex_close_change_buckets','sensex_close',
'close_change_60', 'rsi_change', 'rsi_change_60', 'atr_change','Aroon_Down_change','Aroon_Up_change','mmi_bucket','fundrate_change_buckets','value','mmi',
'ADOSC_change_buckets','ADOSC_buckets','DCPERIOD_change_buckets','Industry',
'marketcap', 'yearly_growth_rate','Industry_average_growth_rate', 'Marketcap_average_growth_rate','ISIN No']]
model_data['date'] = pd.to_datetime(model_data['date'])

def next_month_beat_marketby5(df):
    df['next_month_beat_marketby5'] = 0.0
    for security_id, group in df.groupby('security_id'):
        group = group.sort_values('date',ascending=False)
        group['avg_close_30'] = group['close'].rolling(window=30, min_periods=1).mean()
        group['avg_sensexclose_30'] = group['sensex_close'].rolling(window=30, min_periods=1).mean()
        group['next_month_beat_marketby5'] = (((group['avg_close_30'] / group['close']) - 1)*100.0 - ((group['avg_sensexclose_30'] / group['sensex_close']) - 1)*100.0) >= 5 
        df.loc[group.index, 'next_month_beat_marketby5'] = group['next_month_beat_marketby5'].astype(int)
    return df


model_data = next_month_beat_marketby5(model_data)
model_data.shape
model_data['security_id'].nunique()

model_data.to_csv(r'/Users/amritaagarwal/Downloads/Python_stocks/data/stock_data.csv',index=False,header=True)
model_data.columns
(model_data.tail(1000)).to_csv('test_data.csv')

#############################################################################################################

