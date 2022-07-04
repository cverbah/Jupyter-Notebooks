# -*- coding: utf-8 -*-
"""
Created on Sat Jul  2 21:27:50 2022

@author: c_ver
"""
import json
import requests
import numpy as np
import pandas as pd
import time
from datetime import date, datetime, timezone
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
from collections import Counter
plt.style.use('seaborn')


##FUNCTIONS FOR NFT ANALYSIS

def get_info_by_ranking(rank,ordered_holders):
    assert rank < len(ordered_holders), f'{rank} > {len(list)}:length list'
    wallet_info = ordered_holders[rank-1]
    
    wallet_dict = dict()
    wallet_dict['rank'] = rank
    wallet_dict['wallet'] = wallet_info[0]
    wallet_dict['amount'] =  wallet_info[1]['amount']
    wallet_dict['Tokens ID'] = wallet_info[1]['mints']    
    return wallet_dict


def get_data(collection_id):
    '''gets all the trades from a collection calling the solscan API''' 
    
    url = f"https://api.solscan.io/collection/trade?collectionId={collection_id}&offset=0&limit=all"
    response = requests.get(url)
    
    data = json.loads(response.text)['data']
    df = pd.DataFrame(data)
    
    df['tradeTime'] = df['tradeTime'].apply(lambda trade: datetime.fromtimestamp(trade, tz=timezone.utc))
    df['price'] = df['price'].apply(lambda trade: trade/10**9) #the price is in lamports, need to be transform to sol unit
    
    return df

def get_collection_stats_ME(symbol):
    url = f'https://api-mainnet.magiceden.dev/v2/collections/{symbol}/stats'
    response = requests.get(url)
    response = json.loads(response.text)
    return response

def amount_nfts_per_wallet(wallet, ordered_holders):
    for holder in ordered_holders:
        if holder[0] == wallet:
            amount = holder[1]['amount']
            return amount
    return 0


def group_df_by_interval(df, interval): 
    df = df[['price', 'tradeTime']]
    df.set_index('tradeTime', inplace=True)
    
    #grouping by interval
    df2 = df.resample(rule=interval, closed='left').mean() #avg price paid per interval
    df2 = df2.fillna(method='ffill') #propagate[s] last valid observation forward to next valid (if there are no sales in a day, keep the previous price)
    df2['volume'] = df.resample(rule=interval, closed='left').sum()
    
    #change in %
    df2['price_change']= df2['price'].pct_change().fillna(0)
    df2['volume_change']= df2['volume'].pct_change().fillna(0)
    
    #reordering
    df2 = df2.rename(columns={'price': 'avg_price', 'volume' :'total_volume'})
    df2 = df2.reindex(columns=['avg_price','price_change', 'total_volume', 'volume_change'])
    return df2


def read_interval(df_interval):
    interval = (df_interval.index[1] - df_interval.index[0]).days #in days
    
    if interval == 1:
        return 'day'
    if interval == 7:
        return 'week'
    else:
        return 'hour' #not taking in cosideration months

def plot_price_volume(df_interval):
    values = {'price':'avg_price', 'volume': 'total_volume'}
    
    interval = read_interval(df_interval)
    for val in values.keys():
        fig, ax = plt.subplots(1,2, figsize=(20, 5))
        plt.suptitle(f'{values[val].capitalize()} Evolution',fontsize=20)
        
        #price
        df_interval[f'{values[val]}'].plot(ax=ax[0], color='blue', label=f'{values[val].capitalize()} per {interval}')
        ax[0].legend(fontsize=17)
        
        #volume
        df_interval[f'{val}_change'].plot(ax=ax[1], color='green', label=f'{val.capitalize()} Change per {interval} %')
        ax[1].axhline(0, linestyle='--', color='red')
        ax[1].legend(fontsize=17)
        ax[1].yaxis.set_major_formatter(mtick.PercentFormatter(1))
        
        plt.tight_layout()
        plt.show()
    

def get_collection_traits(df):
    collection_traits = set()
    
    for i in df['attributes']:
        for j in i:
            nft_traits = [j['trait_type'] for j in i]
            nft_traits = set(nft_traits)
            
        collection_traits = collection_traits.union(nft_traits)
        
    collection_traits_dict = {trait:[] for trait in collection_traits}    
    for i in df['attributes']: #second loop
        for j in i:
            trait = set({j['trait_type']}).intersection(collection_traits)
            trait = list(trait).pop()
            value = j['value']
            collection_traits_dict[trait].append(value)
        
    return(collection_traits_dict)


def check_collection_traits(collection_traits):
    dict_collection_values={}
    
    for key,value in collection_traits.items():
        dict_collection_values[key] = set(np.unique(value)) 

    for key,value in dict_collection_values.items():
        print(f'{key}: {value} \n')

def get_nft_traits(df,name):
    #check NFT ID in the data collected
    assert len(df.loc[df['name']== name].head(1)) > 0, 'NFT ID have not been traded yet'
    
    nft = df.loc[df['name']== name].head(1)
    
    nft_token_id = nft.loc[:,'mint'].values[0]
    print(f'Token ID: {nft_token_id}')
    nft_attributes = nft.loc[:,'attributes'].to_numpy()
    
    nft_traits = {nft_trait['trait_type']:nft_trait['value'] for nft_trait in nft_attributes[0]} #dict
    return nft_traits  


def plot_traits(collection_traits,dict_keys):
    c = Counter(collection_traits[dict_keys])
    c = c.most_common() #ordering 
    c = dict(c)
    
    #formatting the values for the plot
    values = c.values()
    values = list(map(lambda val:(val/sum(values))*100, values))
    
    #plot
    fig, ax = plt.subplots(1,1, figsize=(5, 9))
    sns.barplot(x=values, y=list(c.keys()), palette='Set2')
    ax.xaxis.set_major_formatter(mtick.PercentFormatter())
    plt.title(f'Traits Distribution for {dict_keys}')
    plt.show()
    plt.tight_layout()
    

def check_attribute(row, attribute,value):
    for i in row:
        if (i['trait_type'] == attribute):
                if (i['value'] == value):
                    return 1
                else:
                    return 0

def filter_attribute(df,attribute,value):
    data_copy = df.copy()
    
    data_copy[value] = data_copy.apply(lambda row: check_attribute(row['attributes'],attribute,value), axis=1)
    data_filter = data_copy.loc[(data_copy[value]==1)]
    
    data_filter.reset_index(inplace=True, drop=True)
    return data_filter

def plot_nft_price_evolution(df,token_id):
    fig, ax = plt.subplots(1,figsize=(12, 4))
    
    row = df.loc[(df['mint']==token_id)]
    nft = row.loc[row.index,'name'].to_numpy()
    nft = nft[0]
    sns.scatterplot(data=row, x='tradeTime', y='price',marker='o', s=150, ax=ax)
    plt.title(f'Evolution of {nft} Price ')
    plt.tight_layout()
    plt.show()
    
def df_trades_analysis(df,option):
    '''option={'buyer', 'seller'}'''
    
    option_dict ={'buyer':['Buy', 'Buys', 'Spent'], 'seller': ['Sell', 'Sales', 'Sold']}
    kpis = ['min_date_trade', 'max_date_trade', 'count_trade', 'total_trade']
    for kpi in kpis:         
        #min date
        if kpi == 'min_date_trade':           
            min_date_trade = pd.DataFrame(df.groupby([option])['tradeTime'].min())        
            min_date_trade.rename(columns={'tradeTime': f'Min Date ({option_dict[option][0]} Trade)'}, inplace=True)
            min_date_trade.reset_index(inplace=True)
            
        #max date
        if kpi == 'max_date_trade':
            max_date_trade = pd.DataFrame(df.groupby([option])['tradeTime'].max())
            max_date_trade.rename(columns={'tradeTime': f'Max Date ({option_dict[option][0]} Trade)'}, inplace=True)
            max_date_trade.reset_index(inplace=True)
            
        #count
        if kpi == 'count_trade':
            count_trade = pd.DataFrame(df.groupby([option]).count()['price'])
            count_trade.rename(columns={'price': f'Trades ({option_dict[option][1]})'}, inplace=True)
            count_trade.reset_index(inplace=True)
            
        if kpi == 'total_trade': 
            total_trade = df.groupby([option]).sum()
            total_trade.rename(columns={'price': f'Total SOL {option_dict[option][2]}'}, inplace=True)
            total_trade.reset_index(inplace=True)
    
    df_trades = total_trade.merge(count_trade,on=option)
    df_trades[f'SOL Cost Avg ({option_dict[option][1]})'] = round(df_trades[f'Total SOL {option_dict[option][2]}']/df_trades[f'Trades ({option_dict[option][1]})'],2)
    df_trades = df_trades.merge(min_date_trade,on=option).merge(max_date_trade,on=option).sort_values(f'Total SOL {option_dict[option][2]}', ascending=False)
    df_trades.rename(columns={option:'wallet'}, inplace=True)
    df_trades.reset_index(inplace=True, drop= True)
    
    return df_trades
