from datetime import datetime, timedelta
import subprocess
import pandas as pd




headers=["DateTime", "Ticker", "ExpiryDT", "Strike", "F&O", "Option", "Volume", "Open", "High", "Low", "Close", "OpenInterest"]
# df =pd.read_csv("FINNIFTY-I.NFO_2020-01-03.csv", header=None).reset_index(drop=True)
# df.columns=headers
# print(df.head(10))
ExeDF=pd.DataFrame(columns=["Symbol", "Pos", "Date", "Strike", "ExpiryDT", "Option", "EnTime", "SPrice", "ExTime","BPrice"])

OptionDf = pd.DataFrame()
PEOptionDf = pd.DataFrame()
FilteredDf = pd.DataFrame()
UnavailDateList = []
count=0; flag=None; curr_date=None
Pprice=0
high= float("-inf")
low = float("inf")


i=0
def query(**kwargs):
    """
    :param instrument: String
    :param expry_dt: Datetime
    :param strike: numpy int
    :param option_type: CE  PE
    :param start_date: In Datetime
    :param end_date: In Datetime
    """
    # instrument, f_o, expry_dt, strike, option_type, start_date, end_date)
    global ticker, UnavailDateList

    start_date = kwargs['start_date'].strftime("%Y-%m-%d") + 'T' + "09:15:00"
    end_date = kwargs['end_date'].strftime("%Y-%m-%d") + 'T' + "15:30:00"
    if kwargs['f_o'] == 'O':
        ticker = (kwargs['instrument'] + kwargs['expiry_dt'].strftime("%d%b%y") + str(kwargs['strike']) + kwargs[
            'option_type']).upper() + '.NFO'  # nfo FOR OHLCV
    elif kwargs['f_o'] == 'F':
        ticker = kwargs['instrument'] + '-I' + '.NFO'  #+kwargs['start_date'].strftime("%Y-%m-%d")

    print(ticker, start_date, end_date)
    try:
        subprocess.call(["/home/admin/query_client/query_ohlcv", ticker, start_date, end_date])

        # df = pd.read_csv(f"~/query_client/{ticker}.csv", parse_dates=['__time'])

        df = pd.read_csv(f"{ticker}.csv", header=None, low_memory=False).reset_index(drop=True)

        # print(df.head())

        df.columns = ['DateTime', 'Ticker', 'ExpiryDT', 'Strike', 'FnO', 'Option', 'Volume',
                    'Open', 'High', 'Low', 'Close', 'OI']
        # df['Time'] = pd.to_datetime((df['DateTime'])).apply(lambda x: x.time())

        df['Time'] = pd.to_datetime((df['DateTime'])).dt.strftime("%H:%M:%S")

        df["Date"] = pd.to_datetime((df['DateTime'])).dt.strftime("%Y-%m-%d")
        subprocess.call(['unlink', ticker + '.csv'])  # This deletes the file from storage after reading it to memory
        
        # print(df.tail())
        return df
    
    except Exception as e:

        print("Exception occured",e)
        df=pd.DataFrame()
        date = kwargs['start_date'].strftime("%Y-%m-%d")
        if date not in UnavailDateList:
            UnavailDateList.append(date)
        return df


def get_expiry(date):

    ExpDf = pd.read_excel("NIFTYData_20230626.xls")

    ExpDf["DataTime"] = pd.to_datetime(ExpDf["DataTime"])

    date=pd.to_datetime(date)

    mask = ExpDf["DataTime"] >= date
    
    # Find the index of the first occurrence of True in the mask
    next_greater_index = mask.idxmax()

    # Select the row with the next greater date
    next_greater_date_row = ExpDf.loc[next_greater_index]

    return next_greater_date_row["DataTime"]

def future_data_fn():

    future = query(f_o='F', instrument='BANKNIFTY', start_date=pd.to_datetime('2022-01-03'),
                      end_date=pd.to_datetime('2022-12-31'), STime="09:15:00")
    dates = list(future["Date"].unique())  
   
    print(future)
  
    future['Timestamp'] = pd.to_datetime(future['DateTime'])
     # Convert 'DateTime' to datetime format
    future.set_index('DateTime', inplace=True)  # Set 'DateTime' as the index
     # Resample data to 5 minute candle
    future_data = future.set_index('Timestamp').resample('5T').agg({

        'Date': 'first',
        'Time': 'first',
        'Ticker': 'first',
        'Volume': 'sum',
        'Open': 'first',
        'High': 'max',
        'Low': 'min',
        'Close': 'last'
    })
    future_data.reset_index(inplace=True)
    future_data = future_data.dropna(subset=['Date'])
    future_data.to_csv("myresampled_data2022.csv")
    
  
    return future_data

# entry conditions


def calculate_EMA_with_signals(df):
    df['EMA_5'] = df["Close"].ewm(span=5, adjust=False).mean()

    start_time = pd.to_datetime("9:30:00").time()
    end_time = pd.to_datetime("15:15:00").time()
    
    print(df)
    
    # Assuming df is your DataFrame
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    df.set_index('Timestamp', inplace=True)
    future_data = df.between_time(start_time, end_time)
    
    # Create a new column 'stoploss' in the DataFrame
    print(future_data)
    
    # DataFrame index  reset 
    future_data = future_data.reset_index(drop=True)
  
    # Create 'signal' and 'ATMSP' columns if they don't exist
    if 'signal' not in future_data.columns:
        future_data['signal'] = ''
    if 'ATMSP' not in future_data.columns:
        future_data['ATMSP'] = 0

    
    for index, row in future_data.iterrows():
        

        if index > 0:
            if (row['Close'] > row['EMA_5']) and (future_data['Close'][index - 1] <= future_data['EMA_5'][index - 1]) and (pd.to_datetime(row['Time']).time() < end_time):
                
                future_data.at[index, 'signal'] = 'Buy'
                future_data.at[index, 'ATMSP'] = round(row['Close'] / 100) * 100
                
            elif ((row['Close'] < row['EMA_5']) and (future_data['Close'][index - 1] >= future_data['EMA_5'][index - 1]) or (pd.to_datetime(row['Time']).time() == end_time)) :
                future_data.at[index, 'signal'] = 'Sell'
                
        else:
            print("Not enough data for calculation.")
            
 
    future_data = future_data[(future_data['signal'] != '')]
    
    # Group by 'Date' and check if the first row in each group has a 'Sell' signal
    future_data = future_data.groupby('Date').apply(lambda group: group.iloc[1:] if group.iloc[0]['signal'] == 'Sell' else group).reset_index(drop=True)
    
    # Reset the index
    future_data.reset_index(inplace=True)
    
    # Drop the index column
    future_data = future_data.drop(columns=['index'])
    
    
    future_data.to_csv("mydf_callBN.csv")
    
    return future_data

def main(future_data):    
    OptionCE = pd.DataFrame()
    new_data_list = []# List to store new_data dictionaries
    for index, row in future_data.iterrows():
        if row['signal'] == 'Buy':
            # Create a dictionary with the desired values
        
            buy_atmsp = row['ATMSP']
            buy_date = row['Date']
            buy_time = (pd.to_datetime(row['Time']) + timedelta(minutes=5)).strftime('%H:%M:%S') 
            expiry_date = get_expiry(buy_date)

            if all([buy_atmsp, buy_date, buy_time, expiry_date]):
                # Fetch option data
                OptionCE = query(f_o='O', instrument='BANKNIFTY', expiry_dt=expiry_date, strike=row['ATMSP'], option_type="CE", start_date=pd.to_datetime(pd.to_datetime(row['Date']).date()), end_date=pd.to_datetime(pd.to_datetime(row['Date']).date()))

                # Check if option data is not empty
                if not OptionCE.empty:
                    # Extract option data at the same datetime
                    option_buy = OptionCE[OptionCE['Time'] == buy_time]
                    option_buyprice = 0   # Replace with a suitable default value

                    # Check if option data at the same datetime is not empty
                    if not option_buy.empty:
                        option_buyprice = option_buy['Close'].values[0]
                    if option_buyprice >= 10:
                        # Check if the next row is a 'Sell' signal
                        next_row = future_data.iloc[index + 1] if index + 1 < len(future_data) else None
                        if next_row is not None and next_row['signal'] == 'Sell':
                            sell_time = (pd.to_datetime(next_row['Time']) + timedelta(minutes=5)).strftime('%H:%M:%S')

                            # Extract option data at the same datetime
                            option_sell = OptionCE[OptionCE['Time'] == sell_time]
                            option_sellprice = 0  # Replace with a suitable default value

                            # Check if option data at the same datetime is not empty
                            if not option_sell.empty:
                                option_sellprice = option_sell['Close'].values[0]
                            

                            new_data = {
                                    
                                    'Date': buy_date,
                                    'ExpiryDt': expiry_date,
                                    'EnTime': buy_time,
                                    'Bprice': option_buyprice,
                                    'ExTime': sell_time,
                                    'SPrice':option_sellprice
                                    }
                            
                            # Create a new DataFrame for new_data and concatenate it with the original OptionCE DataFrame
                            new_data_list.append(new_data)
                else:
                    print("Option data is empty for Buy signal.")
  
        else:
            print("Skipping row because the signal is not 'Buy'.")
    return new_data_list 


df = future_data_fn()
sigdf = calculate_EMA_with_signals(df)
tradelist = main(sigdf)
tradedf = pd.DataFrame(tradelist)
tradedf.to_csv("mytradedf_callBN.csv")








# df.at[i, 'ATMSP'] = round(df['Close'][i] / 100) * 100



