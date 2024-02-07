import pandas as pd
import requests
import concurrent.futures
from datetime import datetime, timedelta, time
import numpy as np
import scipy
from scipy.stats import norm
import matplotlib.pyplot as plt
from tvDatafeed import TvDatafeed, Interval
import pytz
import os
from dotenv import load_dotenv

load_dotenv()
api = os.getenv('API')

class Gamma:
    def __init__(self, symbol, days_to_expo=0):
        self.symbol = symbol.upper()
        self.days_to_expo = days_to_expo
        self.quote = self.get_quote()

    def get_days_until_expiration(self, days_to_expo):
        today = datetime.now()
        expiration_date = datetime.strptime(days_to_expo, '%Y-%m-%d')
        
        days_left = abs((today-expiration_date).days)
        return days_left

    def get_quote(self):
        if self.symbol == 'SPX':
            tv = TvDatafeed()
            price = tv.get_hist('SPXUSD', exchange='FOREXCOM', interval=Interval.in_1_minute, n_bars=1)
            price = price['close'].iloc[-1]
            return price
        else:
            response = requests.get('https://api.tradier.com/v1/markets/quotes',
                params={'symbols': self.symbol, 'greeks': 'false'},
                headers={'Authorization': f'Bearer {api}', 'Accept': 'application/json'}
            )
            json_response = response.json()
            data = json_response['quotes']['quote']['last']
            return data

    def get_expirations(self):
        expirations = []
        response = requests.get('https://api.tradier.com/v1/markets/options/expirations',
            params={
                'symbol': self.symbol,
                'includeAllRoots': 'true',
                'strikes': 'false',
                'contractSize': 'true',
                'expirationType': 'true'
            },
            headers={'Authorization': f'Bearer {api}', 'Accept': 'application/json'}
        )
        json_response = response.json()
        data = json_response['expirations']['expiration']
        
        for date in data:
            expirations.append(datetime.strptime(date['date'], '%Y-%m-%d'))
        
        expirations.sort()  # Sort the dates
        if self.calculate_fraction_day_remaining() == 0:
            expirations.pop(0)
        
        unique_expirations = [date.strftime('%Y-%m-%d') for date in expirations]
        return unique_expirations



    def calculate_fraction_day_remaining(self):
        est = pytz.timezone('US/Eastern')
        market_close = time(16, 0)  # Market closes at 4:00 PM EST

        now = datetime.now(est)

        if now.time() > market_close:
            return 0
        else:
            market_close_dt = est.localize(datetime.combine(now.date(), market_close))
            time_remaining = market_close_dt - now

            trading_day_duration = timedelta(hours=6.5)
            fraction_remaining = time_remaining.total_seconds() / trading_day_duration.total_seconds()
            return fraction_remaining

    def get_expirations_days_left(self, days_to_expo=0):
        today = datetime.now()
        expirations =  self.get_expirations()
        expirations_x_days_left = []

        for expiration_date in expirations:
            expiration_date = datetime.strptime(expiration_date, '%Y-%m-%d')  # Parse expiration to datetime
            days_left = (expiration_date - today).days

            if days_left <= days_to_expo:  # Changed condition here
                expirations_x_days_left.append(expiration_date.strftime('%Y-%m-%d'))  # Convert back to string
        return expirations_x_days_left
    
    def extract_greeks(self, df):
        try:
            if 'greeks' in df.columns:
                greek_columns = ['delta', 'gamma', 'theta', 'mid_iv']
                for col in greek_columns:
                    df[col] = df['greeks'].apply(lambda x: x[col] if col in x else None)
                df.drop('greeks', axis=1, inplace=True)
            else:
                pass
        except:
            print('Error processing greeks...')

    def get_nearest_strike(self, price, strikes):
        return min(strikes, key=lambda x: abs(x - price))



    def get_chain(self, expiration, price):
        
        
        response = requests.get('https://api.tradier.com/v1/markets/options/chains',
            params={'symbol': self.symbol, 'expiration': expiration, 'greeks': 'true'},
            headers={'Authorization': f'Bearer {api}', 'Accept': 'application/json'}
        )
        data = response.json()
        options = data['options']['option']
        
        strikes = sorted(set(option['strike'] for option in options))  # Extract unique strikes
        
        nearest_strike =  self.get_nearest_strike(price, strikes)
        
        nearest_index = strikes.index(nearest_strike)
        
        surrounding_strikes = strikes[max(0, nearest_index - 20):nearest_index + 21]
        surrounding_strikes = list(reversed(surrounding_strikes))  # Reverse the list to get descending order
        
        filtered_options = [option for option in options if option['strike'] in surrounding_strikes]

        # Separate call and put data
        call_data = [option for option in filtered_options if option['option_type'] == 'call']
        put_data = [option for option in filtered_options if option['option_type'] == 'put']

        calls_df = pd.DataFrame(call_data)
        puts_df = pd.DataFrame(put_data)
        calls_df['price'] = price
        puts_df['price'] = price
        calls_df['days_to_expo'] =  self.get_days_until_expiration(expiration)
        puts_df['days_to_expo'] =  self.get_days_until_expiration(expiration)
    
        columns = ['symbol', 'option_type', 'last', 'volume', 'open_interest', 'greeks', 'price', 'days_to_expo']

        # Set strike as index for both DataFrames
        calls_df.set_index('strike', inplace=True)
        puts_df.set_index('strike', inplace=True)

        calls_df = calls_df[columns]
        puts_df = puts_df[columns]
        # print(calls_df)
        # print(puts_df)


        self.extract_greeks(calls_df)
        self.extract_greeks(puts_df)
        merged_chain = pd.merge(calls_df, puts_df, on='strike', suffixes=('_call', '_put'))

        merged_chain.index = merged_chain.index.astype(float)
        merged_chain['CallIV'] = merged_chain['mid_iv_call'].astype(float)
        merged_chain['PutIV'] = merged_chain['mid_iv_put'].astype(float)
        merged_chain['CallGamma'] = merged_chain['gamma_call'].astype(float)
        merged_chain['PutGamma'] = merged_chain['gamma_put'].astype(float)
        merged_chain['CallOpenInt'] = merged_chain['open_interest_call'].astype(float)
        merged_chain['PutOpenInt'] = merged_chain['open_interest_put'].astype(float)
        merged_chain['CallVolume'] = merged_chain['volume_call'].astype(float)
        merged_chain['PutVolume'] = merged_chain['volume_put'].astype(float)


        # merged_chain.drop(columns=['mid_iv_call', 'mid_iv_put','gamma_call','gamma_put', 'open_interest_call', 'open_interest_put'])

        merged_chain.rename(columns={'price_call': 'Price', 'days_to_expo_call':'Days'}, inplace=True)

        # Select specific columns
        columns = ['CallOpenInt', 'CallVolume', 'CallIV','CallGamma', 'PutOpenInt', 'PutVolume', 'PutIV','PutGamma', 'Price', 'Days']
        merged_chain = merged_chain[columns]
        return merged_chain
    



    def calc_gamma(self, chain):
        call_gamma_list = []
        put_gamma_list = []

        for index, row in chain.iterrows():
            if row['Days'] == 0:
                T = self.calculate_fraction_day_remaining()
            else:
                T = row['Days']
            # Calculate gamma exposure for calls and puts separately
            call_gamma = self.calcGammaEx(row['Price'], index, row['CallIV'], T, 0, 0, "call", row['CallOpenInt'])
            put_gamma = self.calcGammaEx(row['Price'], index, row['PutIV'], T, 0, 0, "put", row['PutOpenInt'])
            
            # Append the calculated gamma values to the lists
            call_gamma_list.append(call_gamma)
            put_gamma_list.append(put_gamma)

        # Add new columns 'CallGamma' and 'PutGamma' to the DataFrame chain
        chain['CallGamma'] = call_gamma_list
        chain['PutGamma'] = put_gamma_list
        return chain


    def find_gex_flip(self, df):
            df = df.sort_index(ascending=True)

            df['CumulativeGEX'] = df['WeightedGEX'].cumsum()
            max = df['CumulativeGEX'].idxmin()
            
            df['Flip'] = max            
            return df



    def calcGammaEx(self, S, K, vol, T, r, q, optType, OI):
        if T == 0 or vol == 0:
            return 0

        dp = (np.log(S/K) + (r - q + 0.5*vol**2)*T) / (vol*np.sqrt(T))
        dm = dp - vol*np.sqrt(T) 

        if optType == 'call':
            gamma = np.exp(-q*T) * norm.pdf(dp) / (S * vol * np.sqrt(T))
            return OI * 100 * S * S * 0.01 * gamma 
        else: # Gamma is same for calls and puts. This is just to cross-check
            gamma = K * np.exp(-r*T) * norm.pdf(dm) / (S * S * vol * np.sqrt(T))
            return OI * 100 * S * S * 0.01 * gamma 
        

    def gexify(self, df):
        df['CallGEX'] = df['CallGamma'] * (df['CallOpenInt']) * 100 * df['Price'] * df['Price'] * 0.01
        df['PutGEX'] = df['PutGamma'] * (df['PutOpenInt']) * 100 * df['Price'] * df['Price'] * 0.01 * -1
        df['TotalGamma'] = (df.CallGEX + df.PutGEX)


        return df
    
    def weigh_gex(self, df):
        if (df['Days'] < 1).any():  # Check if any 'Days' value is less than 1
            
            fraction_day_remaining = self.calculate_fraction_day_remaining()
            
            # Replace 'Days' less than 1 with the calculated fraction or 1 if it's 0
            df.loc[df['Days'] < 1, 'Days'] = max(fraction_day_remaining, 1)  

        df['Weights'] = 1 / df['Days']
        df['CallGEX'] = df['CallGEX'] * df['Weights']
        df['PutGEX'] = df['PutGEX'] * df['Weights']
        df['WeightedGEX'] = df['TotalGamma'] * df['Weights']

        # Get the weighted sum for GEX
        return df




    def merge_and_sum_gex(self, dataframes):
        merged_df = pd.concat(dataframes)  # Concatenate all DataFrames into a single DataFrame

        summed_df = merged_df.groupby('strike').agg({
            'Price': 'first',  # Take the first value of 'Price' for each 'strike'
            'CallGEX': 'sum',
            'PutGEX': 'sum',
            'TotalGamma': 'sum',
            'WeightedGEX': 'sum',
            'CallVolume': 'sum',
            'PutVolume': 'sum'
        }).reset_index()
        summed_df = summed_df.set_index('strike').sort_index(ascending=False)

        return summed_df

    def get_chain_and_gexify(self, expiration, price):
        chain =  self.get_chain(expiration, price)
        chain =  self.gexify(chain)
        chain =  self.weigh_gex(chain)
        return chain
    

    def get_same_day(self):
        csv_path = f'static/csv/{self.symbol}-chain.csv'
        
        
        expiration= self.get_expirations()[0]
        price  =  self.quote
        chain =  self.get_chain_and_gexify(expiration=expiration, price=price)
        chain =  self.find_gex_flip(chain)
        path_chain = self.plot_gex(chain)
        if os.path.exists(csv_path):
            path_vol = self.plot_volume(chain)
            chain.to_csv(csv_path, index=True)
        else:
            chain.to_csv(csv_path, index=True)
            path_vol = self.plot_volume(chain)
        return path_chain, chain, path_vol
    

    def get_all_chains(self):
        price =  self.quote
        expirations =  self.get_expirations_days_left(self.days_to_expo)

        dataframes = []
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = {executor.submit( self.get_chain_and_gexify, exp, price): exp for exp in expirations}

            for future in concurrent.futures.as_completed(futures):
                try:
                    chain = future.result()
                    self.extract_greeks(chain)
                    dataframes.append(chain)
                except Exception as e:
                    print(f"Error getting chain: {str(e)}")

        gex_df =  self.merge_and_sum_gex(dataframes)

        # Filter strikes within a specific range around the current price
        current_price = gex_df['Price'].iloc[0]  # Replace with the actual column name
        range_to_consider = 100  # Number of strikes to consider above and below the current price
        gex_df = gex_df.loc[(gex_df.index >= (current_price - range_to_consider)) &
                            (gex_df.index <= (current_price + range_to_consider))]

        gex_df =  self.find_gex_flip(gex_df)
        path = self.plot_gex_all(gex_df)
        path_vol = self.plot_volume_all(gex_df)
        return path, gex_df, path_vol


    def plot_gex(self, merged_df):
        plt.figure(figsize=(12, 8))  # Adjust the figure size as needed
        # plt.subplots_adjust(right=0.9)
        
        # Sort the DataFrame by index (strike) in ascending order
        merged_df = merged_df.sort_index(ascending=True)

        bars = plt.barh(merged_df.index.astype(str), merged_df['TotalGamma'],
                        color=np.where(merged_df['TotalGamma'] >= 0, 'blue', 'red'), height=0.5)
        
        max_gamma = merged_df['TotalGamma'].max()
        min_gamma = merged_df['TotalGamma'].min()

        range_max = abs(max_gamma)
        range_min = abs(min_gamma)

        # Find the maximum absolute value between max_gamma and min_gamma
        range_max = max(range_max, range_min)

        # Define the range for plotting
        plot_range = (-1.1 * range_max, 1.1 * range_max)

        
        # Get the price value (convert to string)
        price = str(merged_df['Price'].iloc[-1])

        # Find the closest index value to the price
        closest_index = str(merged_df.index.to_series().astype(float).sub(float(price)).abs().idxmin())
        flip = str(merged_df['Flip'].iloc[-1])
        
        plt.axhline(y=closest_index, color='black', linestyle='--', label='Price')
        plt.text(plot_range[0], str(closest_index), f'Price: {price}', verticalalignment='bottom', horizontalalignment='right')

        plt.axhline(y=flip, color='red', linestyle='--', label='GEX Flip')

        plt.text(range_max * -1, flip, f'GEX FLIP: {flip}', verticalalignment='bottom', horizontalalignment='right')


        put_wall = merged_df['PutGEX'].idxmin()
        call_wall = merged_df['CallGEX'].idxmax()
        
        if put_wall != call_wall:
            plt.axhline(y=str(put_wall), color='purple', linestyle='--', label='Put Wall')
            plt.axhline(y=str(call_wall), color='green', linestyle='--', label='Call Wall')
            plt.text(range_max * -1, str(put_wall), f'Put Wall: {put_wall}', verticalalignment='bottom', horizontalalignment='right')
            plt.text(range_max * -1, str(call_wall), f'Call Wall: {call_wall}', verticalalignment='bottom', horizontalalignment='right')
        else:
            plt.axhline(y=str(put_wall), color='blue', linestyle='--', label='GAMMA Wall')
            plt.text(range_max*-1, str(put_wall), f'Gamma Wall: {put_wall}', verticalalignment='bottom')

        # Other plot configurations...
        
        plt.ylabel('Strike')
        plt.xlabel('GEX ($)')
        plt.xlim(plot_range)
        plt.ticklabel_format(style='plain', axis='x')        
        plt.title(f'{self.symbol} - 0DTE - GEX by Strike')
        plt.grid(axis='x')
        plt.gca().invert_xaxis()

        plt.tight_layout()
        # Adjusting y-axis ticks
        
        
        path = f'static/gammas/same_day/{self.symbol}-gex-sd.png'
        plt.savefig(path)
        # plt.show()
        return path
    
    def plot_gex_all(self, merged_df):
        plt.figure(figsize=(12, 8))  # Adjust the figure size as needed
        # plt.subplots_adjust(right=0.9)
        
        # Sort the DataFrame by index (strike) in ascending order
        merged_df = merged_df.sort_index(ascending=True)

        bars = plt.barh(merged_df.index.astype(str), merged_df['TotalGamma'],
                        color=np.where(merged_df['TotalGamma'] >= 0, 'blue', 'red'), height=0.5)
        
        max_gamma = merged_df['TotalGamma'].max()
        min_gamma = merged_df['TotalGamma'].min()

        range_max = abs(max_gamma)
        range_min = abs(min_gamma)

        # Find the maximum absolute value between max_gamma and min_gamma
        range_max = max(range_max, range_min)

        # Define the range for plotting
        plot_range = (-1.1 * range_max, 1.1 * range_max)

        
        # Get the price value (convert to string)
        price = str(merged_df['Price'].iloc[-1])

        # Find the closest index value to the price
        closest_index = str(merged_df.index.to_series().astype(float).sub(float(price)).abs().idxmin())
        flip = str(merged_df['Flip'].iloc[-1])
        
        plt.axhline(y=closest_index, color='black', linestyle='--', label='Price')
        plt.text(plot_range[0], str(closest_index), f'Price: {price}', verticalalignment='bottom', horizontalalignment='right')

        plt.axhline(y=flip, color='red', linestyle='--', label='GEX Flip')

        plt.text(range_max * -1, flip, f'GEX FLIP: {flip}', verticalalignment='bottom', horizontalalignment='right')


        put_wall = merged_df['PutGEX'].idxmin()
        call_wall = merged_df['CallGEX'].idxmax()
        
        if put_wall != call_wall:
            plt.axhline(y=str(put_wall), color='purple', linestyle='--', label='Put Wall')
            plt.axhline(y=str(call_wall), color='green', linestyle='--', label='Call Wall')
            plt.text(range_max * -1, str(put_wall), f'Put Wall: {put_wall}', verticalalignment='bottom', horizontalalignment='right')
            plt.text(range_max * -1, str(call_wall), f'Call Wall: {call_wall}', verticalalignment='bottom', horizontalalignment='right')
        else:
            plt.axhline(y=str(put_wall), color='blue', linestyle='--', label='GAMMA Wall')
            plt.text(range_max*-1, str(put_wall), f'Gamma Wall: {put_wall}', verticalalignment='bottom')

        # Other plot configurations...
        
        plt.ylabel('Strike')
        plt.xlabel('GEX ($)')
        plt.xlim(plot_range)
        plt.ticklabel_format(style='plain', axis='x')        
        plt.title(f'{self.symbol} - {self.days_to_expo}DTE - GEX by Strike')
        plt.grid(axis='x')
        plt.gca().invert_xaxis()

        plt.tight_layout()
        # Adjusting y-axis ticks
        
        
        path = f'static/gammas/all_expo/{self.symbol}-gex-all.png'
        plt.savefig(path)
        # plt.show()
        return path
    
    def plot_volume(self, merged_df):
        plt.figure(figsize=(12, 8))  # Adjust the figure size as needed
        
        # Sort the DataFrame by index (strike) in ascending order
        merged_df = merged_df.sort_index(ascending=True)
        csv_path = f"static/csv/{self.symbol}-chain.csv"  # Replace with the path used in the get_same_day function
        saved_chain = pd.read_csv(csv_path).set_index('strike')
        # print(saved_chain)

    # Calculate the difference in volume between the new data and the saved data
        merged_df['CallVolumeDiff'] = merged_df['CallVolume'] - saved_chain['CallVolume']
        merged_df['PutVolumeDiff'] = merged_df['PutVolume'] - saved_chain['PutVolume']
        # print(merged_df)

    # Create a mask for the new volume
        new_call_volume_mask = merged_df['CallVolumeDiff'] > 0
        new_put_volume_mask = merged_df['PutVolumeDiff'] > 0

    # Update volume bars with the new values and different colors
        call_bars_existing = plt.barh(merged_df.index.astype(str), merged_df['CallVolume'] - merged_df['CallVolumeDiff'],
                                        color='lightblue', height=0.25, label='Existing Call Volume')

        call_bars_new = plt.barh(merged_df.index.astype(str), merged_df['CallVolumeDiff'],
                                color='blue', height=0.25, label='New Call Volume', left=merged_df['CallVolume'] - merged_df['CallVolumeDiff'])

        put_bars_existing = plt.barh(merged_df.index.astype(str), -merged_df['PutVolume'] + merged_df['PutVolumeDiff'],
                                        color='lightcoral', height=0.25, label='Existing Put Volume')

        put_bars_new = plt.barh(merged_df.index.astype(str), merged_df['PutVolumeDiff'],
                                color='red', height=0.25, label='New Put Volume', left=-merged_df['PutVolume'])

        
        max_call = merged_df['CallVolume'].idxmax()
        max_put = merged_df['PutVolume'].idxmax()
        
        max_gamma = merged_df['CallVolume'].max()
        min_gamma = merged_df['PutVolume'].max()

        range_max = abs(max_gamma)
        range_min = abs(min_gamma)
        range_max = max(range_max, range_min)
        
        plot_range = (-1.1 * range_max, 1.1 * range_max)
        
        price = str(merged_df['Price'].iloc[-1])

        # Find the closest index value to the price
        closest_index = str(merged_df.index.to_series().astype(float).sub(float(price)).abs().idxmin())
        plt.axhline(y=closest_index, color='black', linestyle='--', label='Price')
        plt.text(range_max*-.8, str(closest_index), f'Price: {price}', verticalalignment='bottom')
        plt.axhline(y=str(max_call), color='blue', linestyle='--', label='call Wall')
        plt.text(range_max*-.8, str(max_call), f'Call Wall: {max_call}', verticalalignment='bottom')
        
        plt.axhline(y=str(max_put), color='red', linestyle='--', label='put Wall')
        plt.text(range_max*-.8, str(max_put), f'Put Wall: {max_put}', verticalalignment='bottom')

        # Other configurations for the plot
        plt.xlabel('Volume')
        plt.ylabel('Strike')
        plt.xlim(plot_range)
        plt.title(f'Call and Put Volume by Strike - {self.days_to_expo} DTE')
        plt.legend()
        plt.grid(axis='x')
        plt.gca().invert_xaxis()
        plt.tight_layout()
        path = f'static/volume/same_day/{self.symbol}-volume-sd.png'
        plt.savefig(path)
        return path
    
    def plot_volume_all(self, merged_df):
        plt.figure(figsize=(12, 8))  # Adjust the figure size as needed
        
        # Sort the DataFrame by index (strike) in ascending order
        merged_df = merged_df.sort_index(ascending=True)
        csv_path = f"static/csv/{self.symbol}-chain.csv"  # Replace with the path used in the get_same_day function
        saved_chain = pd.read_csv(csv_path).set_index('strike')
        # print(saved_chain)

    # Calculate the difference in volume between the new data and the saved data
        merged_df['CallVolumeDiff'] = merged_df['CallVolume'] - saved_chain['CallVolume']
        merged_df['PutVolumeDiff'] = merged_df['PutVolume'] - saved_chain['PutVolume']
        # print(merged_df)

    # Create a mask for the new volume
        new_call_volume_mask = merged_df['CallVolumeDiff'] > 0
        new_put_volume_mask = merged_df['PutVolumeDiff'] > 0

    # Update volume bars with the new values and different colors
        call_bars_existing = plt.barh(merged_df.index.astype(str), merged_df['CallVolume'] - merged_df['CallVolumeDiff'],
                                        color='lightblue', height=0.25, label='Existing Call Volume')

        call_bars_new = plt.barh(merged_df.index.astype(str), merged_df['CallVolumeDiff'],
                                color='blue', height=0.25, label='New Call Volume', left=merged_df['CallVolume'] - merged_df['CallVolumeDiff'])

        put_bars_existing = plt.barh(merged_df.index.astype(str), -merged_df['PutVolume'] + merged_df['PutVolumeDiff'],
                                        color='lightcoral', height=0.25, label='Existing Put Volume')

        put_bars_new = plt.barh(merged_df.index.astype(str), merged_df['PutVolumeDiff'],
                                color='red', height=0.25, label='New Put Volume', left=-merged_df['PutVolume'])

        
        max_call = merged_df['CallVolume'].idxmax()
        max_put = merged_df['PutVolume'].idxmax()
        
        max_gamma = merged_df['CallVolume'].max()
        min_gamma = merged_df['PutVolume'].max()

        range_max = abs(max_gamma)
        range_min = abs(min_gamma)
        range_max = max(range_max, range_min)
        
        plot_range = (-1.1 * range_max, 1.1 * range_max)
        
        price = str(merged_df['Price'].iloc[-1])

        # Find the closest index value to the price
        closest_index = str(merged_df.index.to_series().astype(float).sub(float(price)).abs().idxmin())
        plt.axhline(y=closest_index, color='black', linestyle='--', label='Price')
        plt.text(range_max*-.8, str(closest_index), f'Price: {price}', verticalalignment='bottom')
        plt.axhline(y=str(max_call), color='blue', linestyle='--', label='call Wall')
        plt.text(range_max*-.8, str(max_call), f'Call Wall: {max_call}', verticalalignment='bottom')
        
        plt.axhline(y=str(max_put), color='red', linestyle='--', label='put Wall')
        plt.text(range_max*-.8, str(max_put), f'Put Wall: {max_put}', verticalalignment='bottom')

        # Other configurations for the plot
        plt.xlabel('Volume')
        plt.ylabel('Strike')
        plt.xlim(plot_range)
        plt.title(f'Call and Put Volume by Strike - {self.days_to_expo} DTE')
        plt.legend()
        plt.grid(axis='x')
        plt.gca().invert_xaxis()
        plt.tight_layout()
        path = f'static/volume/all_expo/{self.symbol}-volume-all.png'
        plt.savefig(path)
        return path
    
    
    def plot_oi(self, merged_df):
        plt.figure(figsize=(12, 8))  # Adjust the figure size as needed
        
        # Sort the DataFrame by index (strike) in ascending order
        merged_df = merged_df.sort_index(ascending=True)

        # Call volume bars
        call_bars = plt.barh(merged_df.index.astype(str), merged_df['CallOpenInt'],
                            color='blue', height=0.25, label='Call OI')

        # Put volume bars (with negative values)
        put_bars = plt.barh(merged_df.index.astype(str), -merged_df['PutOpenInt'],
                            color='red', height=0.25, label='Put OI')
        
        max_call = merged_df['CallOpenInt'].idxmax()
        max_put = merged_df['PutOpenInt'].idxmax()
        
        max_gamma = merged_df['CallOpenInt'].max()
        min_gamma = merged_df['PutOpenInt'].max()

        range_max = abs(max_gamma)
        range_min = abs(min_gamma)
        range_max = max(range_max, range_min)
        
        price = str(merged_df['Price'].iloc[-1])

        # Find the closest index value to the price
        closest_index = str(merged_df.index.to_series().astype(float).sub(float(price)).abs().idxmin())
        plt.axhline(y=closest_index, color='black', linestyle='--', label='Price')
        plt.text(range_max*-.8, str(closest_index), f'Price: {price}', verticalalignment='bottom')
        plt.axhline(y=str(max_call), color='blue', linestyle='--', label='call Wall')
        plt.text(range_max*-.8, str(max_call), f'Call Wall: {max_call}', verticalalignment='bottom')
        
        plt.axhline(y=str(max_put), color='red', linestyle='--', label='put Wall')
        plt.text(range_max*-.8, str(max_put), f'Put Wall: {max_put}', verticalalignment='bottom')

        # Other configurations for the plot
        plt.xlabel('Volume')
        plt.ylabel('Strike')
        plt.title(f'Call and Put OI by Strike - {self.days_to_expo} DTE')
        plt.legend()
        plt.grid(axis='x')
        plt.gca().invert_xaxis()
        plt.tight_layout()
        # plt.show()
        path = f'static/oi/{self.symbol}-oi-{datetime.now().strftime("%Y%m%d")}.png'
        plt.savefig(path)
        # plt.show()
        return path

# spx = Gamma('SPX', 98)
# spx.get_same_day()
# spx.get_all_chains()
