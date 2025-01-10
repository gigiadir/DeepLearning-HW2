import pandas as pd
import matplotlib.pyplot as plt

stocks = pd.read_csv('data/sp-500-stock-prices/SP 500 Stock Prices 2014-2017.csv')

def section_1():
    df_google = stocks[stocks['symbol'] == 'GOOGL']
    df_amzn = stocks[stocks['symbol'] == 'AMZN']

    plt.style.use('default')

    plt.figure(figsize=(14,7))
    plt.plot(df_google['date'], df_google['high'], label='GOOGL', color='blue')
    plt.plot(df_amzn['date'], df_amzn['high'], label='AMZN', color='orange')

    plt.title('Daily High Prices for GOOGL and AMZN', fontsize=20)
    plt.xlabel('Date', fontsize = 16)
    plt.xticks(df_google['date'][::60], rotation=45)

    plt.ylabel('Daily High Price', fontsize = 16)
    plt.legend()
    plt.grid()
    plt.tight_layout()

    plt.savefig('output/sp500/section1.png')
    plt.show()


if __name__ == '__main__':
    section_1()