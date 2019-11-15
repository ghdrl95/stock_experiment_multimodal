import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import talib
"""This cell defines the plot_candles function"""

def plot_candles(pricing=[], title='test.jpg',
                 volume_bars=False,
                 color_function=None,
                 overlays=[],
                 technicals=[],
                 technicals_titles=[],
                 volume_overlay = [],
                 bband=[],
                 kospi=[],
                 technical_y=[]
                 ):
    """ Plots a candlestick chart using quantopian pricing data.

    Author: Daniel Treiman

    Args:
      pricing: A pandas dataframe with columns ['open_price', 'close_price', 'high', 'low', 'volume']
      title: An optional title for the chart
      volume_bars: If True, plots volume bars
      color_function: A function which, given a row index and price series, returns a candle color.
      overlays: A list of additional data series to overlay on top of pricing.  Must be the same length as pricing.
      technicals: A list of additional data series to display as subplots.
      technicals_titles: A list of titles to display for each technical indicator.
    """

    def default_color(index, open_price, close_price, low, high):
        return 'b' if open_price[index] > close_price[index] else 'r'

    color_function = color_function or default_color
    overlays = overlays or []
    technicals = technicals or []
    technicals_titles = technicals_titles or []


    subplot_count = 0
    subplot_idx = 0
    acc_idx = 0
    if type(pricing) is not list:
        subplot_count += 1
    if bband:
        subplot_count += 1
    if volume_bars:
        subplot_count += 1
    if kospi:
        subplot_count += 1
        acc_idx = 1
    if technicals:
        subplot_count += len(technicals)

    if subplot_count == 1:
        fig, ax1 = plt.subplots(1, 1)
    else:
        ratios = np.insert(np.full(subplot_count - 1, 1), 0, 3)
        fig, subplots = plt.subplots(subplot_count, 1, sharex=True, gridspec_kw={'height_ratios': ratios})
        ax1 = subplots[0]
    if type(pricing) is not list:
        open_price = pricing['open_price']
        close_price = pricing['close_price']
        low = pricing['low']
        high = pricing['high']
        oc_min = pd.concat([open_price, close_price], axis=1).min(axis=1)
        oc_max = pd.concat([open_price, close_price], axis=1).max(axis=1)
        x = np.arange(len(pricing))


        candle_colors = [color_function(i, open_price, close_price, low, high) for i in x]
        candles = ax1.bar(x, oc_max - oc_min, bottom=oc_min, color=candle_colors, linewidth=0)
        lines = ax1.vlines(x, low, high, color=candle_colors, linewidth=1)
        ax1.xaxis.grid(False)
        lines = ax1.vlines(x, low, high, color=candle_colors, linewidth=1)
    #ax1.xaxis.set_tick_params(which='major', length=3.0, direction='in', top='off')
    # Assume minute frequency if first two bars are in the same day.
    frequency = 'day'
    time_format = '%Y%m%d'
    if frequency == 'minute':
        time_format = '%H:%M'
    # Set X axis tick labels.
    #plt.xticks(x, [date for date in pricing.index], rotation='vertical')
    for i, overlay in enumerate(overlays):
        if len(overlay) > 1:
            ax1.plot(x, overlay, linewidth=2)
        else:
            ax1.scatter(x, overlay, linewidth=2)

    #볼린저 밴드
    for bb in bband:
        if len(bb) > 1:
            ax1.plot(np.arange(len(bb)), bb, 'k', linewidth=2)
        else:
            ax1.scatter([1], bb, linewidth=2)

    # Plot volume bars if needed
    if volume_bars:
        ax2 = subplots[1]
        volume = pricing['volume']
        volume_scale = None

        ax2.bar(x, volume, color=candle_colors)

        ax2.xaxis.grid(False)
        for over in volume_overlay:
            if len(over) > 1:
                ax2.plot(x, over, linewidth=2)
            else:
                ax2.scatter(x, over, linewidth=2)
    # Plot additional technical over
    for (i, technical) in enumerate(technicals):
        #ax = subplots[i]  # Technical indicator plots are shown last
        ax = ax1
        for extra_data in technical:
            #if len(extra_data) != 20:
            #    print('error 주범 :',i)
            if len(extra_data)> 1:
                try:
                    ax.plot(np.arange(len(extra_data)), extra_data, linewidth=2)
                except:
                    print(i)
            else:
                print('scatter :',i)
                ax.scatter(np.arange(len(extra_data)),extra_data,linewidths=2)
        if technical_y is not None:
            if technical_y[i]:
                ax.set_ylim(technical_y[i])
        ax.yaxis.grid(True)
        #ax.margins(x=0,y=0)
        #if i < len(technicals_titles):
        #    ax.set_title(technicals_titles[i])
    #print(ax1,fig)
    if kospi:
        #draw KOSPI candle chart
        ax3 = subplots[-1]
        open_price = kospi['open_price']
        close_price = kospi['close_price']
        low = kospi['low']
        high = kospi['high']
        oc_min = pd.concat([open_price, close_price], axis=1).min(axis=1)
        oc_max = pd.concat([open_price, close_price], axis=1).max(axis=1)
        candle_colors = [color_function(i, open_price, close_price, low, high) for i in x]
        candles = ax3.bar(x, oc_max - oc_min, bottom=oc_min, color=candle_colors, linewidth=0)
        lines = ax3.vlines(x, low, high, color=candle_colors, linewidth=1)

        ax3.yaxis.grid(True)

    ax1.margins(x=0, y=0)
    #ax2.margins(x=0)
    datasize = 20
    # 1 = 100px
    fig.set_figwidth(0.1*datasize, forward=False)
    fig.set_figheight(4.5+(subplot_count-2))
    fig.savefig(title, bbox_inches='tight')
    fig.clf()
    plt.close()
    return

#data preprocessing
def data_generate(chart_data, range = 1, start_idx = 0, KOSPI_DATA = None, KOSPI_IDX = None, actions=None, title = 'test.png', type =0):
    if start_idx == 0:
        #range += 1
        ma5 = talib.SMA(chart_data[-(range+5):,1],5)[5:]
        ma10 = talib.SMA(chart_data[-(range+10):, 1], 10)[10:]
        #ma20 = talib.SMA(chart_data[-(range+20):, 1], 20)[20:] #볼린저밴드로 대체
        ma60 = talib.SMA(chart_data[-(range+60):, 1], 60)[60:]
        ma120 = talib.SMA(chart_data[-(range+120):, 1], 120)[120:]
        volume_ma = talib.SMA(chart_data[-(range+20):, 5], 20)[20:]
        upper, ma20, lower = talib.BBANDS(chart_data[-(range+20):, 1], timeperiod=20, nbdevup=2, nbdevdn=2)
        upper = upper[20:]
        ma20 = ma20[20:]
        lower = lower[20:]
        data = np.hstack([chart_data[-range:,1:5], chart_data[-range:,5:6]])
        volume_fp = chart_data[-range:, 9]
        volume_co= chart_data[-range:, 10]
        volume_pp= chart_data[-range:, 11]
        #chdegree   = [chart_data[-range:, 6]]
        #sojinrate  = [chart_data[-range:, 7]]
        #changerate =[chart_data[-range:, 8]]
        slowk, slowd = talib.STOCH(chart_data[- (range + 17):, 4],
                                   chart_data[- (range + 17):, 3],
                                   chart_data[- (range + 17):, 1], fastk_period=10, slowk_period=5,
                                   slowk_matype=0, slowd_period=5,
                                   slowd_matype=0)
        slow_stock = [[80] * range, [20] * range, slowk[17:], slowd[17:]]

        macd, macdsignal, _ = talib.MACD(chart_data[-(range + 33):, 5], fastperiod=12, slowperiod=26, signalperiod=9 )
        macd = [macd[33:], macdsignal[33:], _[33:]]

        obv_all = talib.OBV(chart_data[-range:,1], chart_data[-range:, 5])
        obv_fp  = talib.OBV(chart_data[-range:, 1], chart_data[-range:,9])
        obv_co  = talib.OBV(chart_data[-range:, 1], chart_data[-range:,10])
        obv_pp  = talib.OBV(chart_data[-range:, 1], chart_data[-range:,11])

        dmi = talib.ADX(chart_data[ - (range + 14):, 4],
                        chart_data[ - (range + 14):, 3],
                        chart_data[ - (range + 14):, 1], timeperiod=14)[14:]
        dmi_minus = talib.MINUS_DI(chart_data[ - (range + 14):, 4],
                                   chart_data[ - (range + 14):, 3],
                                   chart_data[ - (range + 14):, 1], timeperiod=14)[14:]
        dmi_plus = talib.PLUS_DI(chart_data[ - (range + 14):, 4],
                                 chart_data[ - (range + 14):, 3],
                                 chart_data[ - (range + 14):, 1], timeperiod=14)[14:]

        dmi = [dmi, dmi_minus, dmi_plus]
    else:
        start_idx += 1
        ma5 = talib.SMA(chart_data[start_idx-(range + 5):start_idx, 1], 5)[5:]
        ma10 = talib.SMA(chart_data[start_idx-(range + 10):start_idx, 1], 10)[10:]
        # ma20 = talib.SMA(chart_data[start_idx-(range+20):start_idx, 1], 20)[20:] #볼린저밴드로 대체
        ma60 = talib.SMA(chart_data[  start_idx-(range + 60):start_idx, 1], 60)[60:]
        ma120 = talib.SMA(chart_data[start_idx-(range + 120):start_idx, 1], 120)[120:]
        volume_ma = talib.SMA(chart_data[start_idx-(range + 20):start_idx, 5], 20)[20:]
        upper, ma20, lower = talib.BBANDS(chart_data[start_idx-(range + 20):start_idx, 1], timeperiod=20, nbdevup=2, nbdevdn=2)
        upper = upper[20:]
        ma20 = ma20[20:]
        lower = lower[20:]
        data = np.hstack([chart_data[start_idx-(range):start_idx, 1:5], chart_data[start_idx-(range):start_idx, 5:6]])
        volume_fp = chart_data[ start_idx-(range):start_idx, 9]
        volume_co = chart_data[ start_idx-(range):start_idx, 10]
        volume_pp = chart_data[ start_idx-(range):start_idx, 11]
        #chdegree =[chart_data[  start_idx-(range):start_idx, 6]]
        #sojinrate =[chart_data[ start_idx-(range):start_idx, 7]]
        #changerate =[chart_data[start_idx-(range):start_idx, 8]]
        #stochastic -slowk, slowd 순
        slowk, slowd = talib.STOCH(chart_data[ start_idx-(range+17):start_idx, 4], chart_data[ start_idx-(range+17):start_idx, 3], chart_data[ start_idx-(range+17):start_idx, 1], fastk_period=10, slowk_period=5, slowk_matype=0, slowd_period=5,
                             slowd_matype=0)
        slow_stock = [[80] * range, [20] * range, slowk[17:], slowd[17:]]
        macd, macdsignal, _ = talib.MACD(chart_data[start_idx-(range + 33):start_idx, 5], fastperiod=12, slowperiod=26, signalperiod=9 )
        macd = [macd[33:], macdsignal[33:], _[33:]]

        obv_all = talib.OBV(chart_data[start_idx-range:start_idx, 1], chart_data[start_idx-range:start_idx,5])
        obv_fp  = talib.OBV(chart_data[start_idx-range:start_idx, 1], chart_data[start_idx-range:start_idx,9])
        obv_co  = talib.OBV(chart_data[start_idx-range:start_idx, 1], chart_data[start_idx-range:start_idx,10])
        obv_pp  = talib.OBV(chart_data[start_idx-range:start_idx, 1], chart_data[start_idx-range:start_idx,11])

        dmi =       talib.ADX(chart_data[start_idx-(range + 27):start_idx, 4], chart_data[start_idx-(range + 27):start_idx, 3], chart_data[start_idx-(range + 27):start_idx, 1], timeperiod=14)[27:]
        dmi_minus = talib.MINUS_DI(chart_data[start_idx-(range + 14):start_idx, 4], chart_data[start_idx-(range + 14):start_idx, 3],chart_data[start_idx-(range + 14):start_idx, 1], timeperiod=14)[14:]
        dmi_plus =  talib.PLUS_DI(chart_data[start_idx-(range + 14):start_idx, 4], chart_data[start_idx-(range + 14):start_idx, 3],chart_data[start_idx-(range + 14):start_idx, 1], timeperiod=14)[14:]

        dmi = [dmi, dmi_minus, dmi_plus]
    #date = chart_data[:range,0].astype(np.int32).astype('S')
    #'open_price', 'close_price', 'high', 'low', 'volume'
    '''
    if KOSPI_IDX:

        #kospi = KOSPI_DATA[KOSPI_IDX-(range):KOSPI_IDX, 1]
        #extra_data.append([kospi])

        kospi = KOSPI_DATA[KOSPI_IDX - (range):KOSPI_IDX, 1:]
        pdData_kospi = pd.DataFrame(data=kospi, columns=['close_price', 'open_price', 'low', 'high'])
    '''
    # 0 캔들차트 + 이평선 + 거래량
    # 1 볼린저 밴드
    # 2 MACD
    # 3 OBV
    # 4 DMI
    # 5 스토캐스틱
    # 6 모든지표 합친거
    # 7 KOSPI 데이터 차트
    if type == 0:
        overlay_chart = [ma5, ma10, ma60, ma120]
        overlay_volume = [volume_ma]
        pdData = pd.DataFrame(data=data,columns=['close_price','open_price','low','high','volume'])

        plot_candles(pdData, title=title, overlays=overlay_chart, volume_bars=True, volume_overlay=overlay_volume)
    elif type == 1:
        bband = [upper, lower, ma20]
        plot_candles(bband=bband, title=title)
    elif type == 2:
        extra_data = [macd]  # [chdegree, sojinrate, changerate, macd ]
        extra_data_y_limit = [[]]
        plot_candles(technicals=extra_data, technical_y=extra_data_y_limit, title=title)
    elif type == 3:
        extra_data = [[obv_all, obv_fp, obv_co, obv_pp] ]  # [chdegree, sojinrate, changerate, macd ]
        extra_data_y_limit = [[]]
        plot_candles(technicals=extra_data, technical_y=extra_data_y_limit, title=title)
    elif type == 4:
        extra_data = [dmi]  # [chdegree, sojinrate, changerate, macd ]
        extra_data_y_limit = [[0,100]]
        plot_candles(technicals=extra_data, technical_y=extra_data_y_limit, title=title)
    elif type ==5 :
        extra_data = [slow_stock]  # [chdegree, sojinrate, changerate, macd ]
        extra_data_y_limit = [[0, 100]]
        plot_candles(technicals=extra_data, technical_y=extra_data_y_limit, title=title)
    elif type == 7:
        # kospi = KOSPI_DATA[KOSPI_IDX-(range):KOSPI_IDX, 1]
        # extra_data.append([kospi])

        kospi = KOSPI_DATA[KOSPI_IDX - (range):KOSPI_IDX, 1:]
        pdData_kospi = pd.DataFrame(data=kospi, columns=['close_price', 'open_price', 'low', 'high'])
        plot_candles(pdData_kospi, title=title)

#data load
'''

def data_load(file, range = 1, start_idx = 0):
    chart_data = np.flip(np.genfromtxt("./chart_data/" + file + "_1.csv", delimiter=','), 0)
    if start_idx == 0:
        #range += 1

        ma5 = talib.SMA(chart_data[-(range+5):,1],5)[5:]
        ma10 = talib.SMA(chart_data[-(range+10):, 1], 10)[10:]
        #ma20 = talib.SMA(chart_data[-(range+20):, 1], 20)[20:] #볼린저밴드로 대체
        ma60 = talib.SMA(chart_data[-(range+60):, 1], 60)[60:]
        ma120 = talib.SMA(chart_data[-(range+120):, 1], 120)[120:]
        volume_ma = talib.SMA(chart_data[-(range+20):, 5], 20)[20:]
        upper, ma20, lower = talib.BBANDS(chart_data[-(range+20):, 1], timeperiod=20, nbdevup=2, nbdevdn=2)
        upper = upper[20:]
        ma20 = ma20[20:]
        lower = lower[20:]
        data = np.hstack([chart_data[-range:,1:5], chart_data[-range:,5:6]])
        volume_fp = chart_data[-range:, 9]
        volume_co= chart_data[-range:, 10]
        volume_pp= chart_data[-range:, 11]
        chdegree   = [chart_data[-range:, 6]]
        sojinrate  = [chart_data[-range:, 7]]
        changerate =[chart_data[-range:, 8]]
        macd, macdsignal, _ = talib.MACD(chart_data[-(range + 33):, 5], fastperiod=12, slowperiod=26, signalperiod=9 )
        macd = [macd[33:], macdsignal[33:],_[33:]]
    else:
        start_idx += 1
        ma5 = talib.SMA(chart_data[-(range + 5+start_idx):-start_idx, 1], 5)[5:]
        ma10 = talib.SMA(chart_data[-(range + 10+start_idx):-start_idx, 1], 10)[10:]
        # ma20 = talib.SMA(chart_data[-(range+20+start_idx):-start_idx, 1], 20)[20:] #볼린저밴드로 대체
        ma60 = talib.SMA(chart_data[-(range + 60+start_idx):-start_idx, 1], 60)[60:]
        ma120 = talib.SMA(chart_data[-(range + 120+start_idx):-start_idx, 1], 120)[120:]
        volume_ma = talib.SMA(chart_data[-(range + 20+start_idx):-start_idx, 5], 20)[20:]
        upper, ma20, lower = talib.BBANDS(chart_data[-(range + 20+start_idx):-start_idx, 1], timeperiod=20, nbdevup=2, nbdevdn=2)
        upper = upper[20:]
        ma20 = ma20[20:]
        lower = lower[20:]
        data = np.hstack([chart_data[-(range+start_idx):-start_idx, 1:5], chart_data[-(range+start_idx):-start_idx, 5:6]])
        volume_fp = chart_data[-(range+start_idx):-start_idx, 9]
        volume_co = chart_data[-(range+start_idx):-start_idx, 10]
        volume_pp = chart_data[-(range+start_idx):-start_idx, 11]
        chdegree =[chart_data[-(range + start_idx):-start_idx, 6]]
        sojinrate =[chart_data[-(range + start_idx):-start_idx, 7]]
        changerate =[chart_data[-(range + start_idx):-start_idx, 8]]
        macd, macdsignal, _ = talib.MACD(chart_data[-(range + start_idx + 33):-start_idx, 5], fastperiod=12, slowperiod=26, signalperiod=9 )
        macd = [macd[33:], macdsignal[33:], _[33:]]
    #date = chart_data[:range,0].astype(np.int32).astype('S')
    #'open_price', 'close_price', 'high', 'low', 'volume'
    overlay_chart = [ma5, ma10, ma60, ma120]
    overlay_volume = [volume_ma, volume_fp, volume_co, volume_pp]
    extra_data = [chdegree, sojinrate, changerate, macd]
    bband = [upper, lower, ma20]
    pdData = pd.DataFrame(data=data,columns=['close_price','open_price','low','high','volume'])

    plot_candles(pdData, title='', overlays=overlay_chart, volume_bars=True, volume_overlay=overlay_volume, technicals=extra_data, bband=bband)
'''


if __name__ == '__main__':
    from time import sleep
    #data_load('000087', 25, 0)
    pass
































