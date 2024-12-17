# AAPL-test-2

import yfinance as yf
import pandas as pd
from pyecharts import options as opts
from pyecharts.charts import Kline, Line, Grid, Bar
from pyecharts.globals import ThemeType
import os
from datetime import datetime
from pyecharts.commons.utils import JsCode

# Parameters
symbol = 'AAPL'
start_date = '2023-01-01'  # Adjust as needed
end_date = "2024-12-18"    # Set to December 18, 2024

def download_stock_data(symbol: str, start: str, end: str) -> pd.DataFrame:
    print(f"Fetching data for {symbol} from {start} to {end}...")
    data = yf.download(symbol, start=start, end=end)
    if data.empty:
        raise ValueError("Data is empty. Check the symbol and date range.")
    print("Data downloaded. Columns:", data.columns)
    print("Data preview:")
    print(data.head())
    return data

def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    # Simple Moving Averages
    df['MA50'] = df['Close'].rolling(window=50).mean()
    df['MA200'] = df['Close'].rolling(window=200).mean()

    # Exponential Moving Averages
    df['EMA20'] = df['Close'].ewm(span=20, adjust=False).mean()
    df['EMA50'] = df['Close'].ewm(span=50, adjust=False).mean()

    # Bollinger Bands (20-day, 2 std)
    period = 20
    df['MB'] = df['Close'].rolling(window=period).mean()
    df['STD'] = df['Close'].rolling(window=period).std()
    multiplier = 2
    df['UB'] = df['MB'] + multiplier * df['STD']
    df['LB'] = df['MB'] - multiplier * df['STD']

    # VWAP Calculation
    df['CumVol'] = df['Volume'].cumsum()
    df['CumVolPrice'] = (df['Volume'] * df['Close']).cumsum()
    df['VWAP'] = df['CumVolPrice'] / df['CumVol']

    # Ichimoku Kinko Hyo
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
    high_9 = df['High'].rolling(window=9).max()
    low_9 = df['Low'].rolling(window=9).min()
    df['Tenkan'] = (high_9 + low_9) / 2

    # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
    high_26 = df['High'].rolling(window=26).max()
    low_26 = df['Low'].rolling(window=26).min()
    df['Kijun'] = (high_26 + low_26) / 2

    # Senkou Span A (Leading Span A): (Tenkan + Kijun) / 2 shifted forward by 26 periods
    df['SpanA'] = ((df['Tenkan'] + df['Kijun']) / 2).shift(26)

    # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2 shifted forward by 26 periods
    high_52 = df['High'].rolling(window=52).max()
    low_52 = df['Low'].rolling(window=52).min()
    df['SpanB'] = ((high_52 + low_52) / 2).shift(26)

    # Chikou Span (Lagging Span): Today's Close shifted 26 periods behind
    df['Chikou'] = df['Close'].shift(-26)

    print("\nMAs, EMAs, BOLL, VWAP, and Ichimoku calculated. Preview:")
    print(df[['Open', 'High', 'Low', 'Close', 'Volume', 'MA50', 'MA200', 
              'EMA20', 'EMA50', 'MB', 'UB', 'LB', 'VWAP', 'Tenkan', 'Kijun', 'SpanA', 'SpanB', 'Chikou']].head())
    return df

def get_fundamentals(symbol: str) -> (str, str):
    stock = yf.Ticker(symbol)
    info = stock.info
    pe_ratio = info.get('trailingPE', 'N/A')
    market_cap = info.get('marketCap', 'N/A')

    if market_cap != 'N/A' and market_cap is not None:
        if market_cap > 1e9:
            market_cap_str = f"{market_cap / 1e9:.2f}B"
        elif market_cap > 1e6:
            market_cap_str = f"{market_cap / 1e6:.2f}M"
        else:
            market_cap_str = str(market_cap)
    else:
        market_cap_str = "N/A"

    return pe_ratio, market_cap_str

def prepare_kline_data(df: pd.DataFrame) -> list:
    # Kline data format: [Open, Close, Low, High]
    return df[['Open', 'Close', 'Low', 'High']].values.tolist()

def prepare_volume_data(df: pd.DataFrame) -> list:
    if 'Volume' in df.columns:
        vol = df['Volume']
        # Ensure vol is a Series
        if isinstance(vol, pd.DataFrame):
            vol = vol.iloc[:, 0]
        return vol.tolist()
    raise KeyError("Volume column not found in the DataFrame.")

def create_kline_chart(df: pd.DataFrame, k_data: list, symbol: str, pe_ratio: str, market_cap: str) -> Kline:
    dates = df.index.strftime('%Y-%m-%d').tolist()
    subtitle_text = f"P/E Ratio: {pe_ratio}, Market Cap: {market_cap}"

    kline = (
        Kline()
        .add_xaxis(dates)
        .add_yaxis(
            series_name="K线",
            y_axis=k_data,
            markpoint_opts=opts.MarkPointOpts(
                data=[
                    opts.MarkPointItem(type_="max", name="Highest", value_dim="highest"),
                    opts.MarkPointItem(type_="min", name="Lowest", value_dim="lowest")
                ],
                symbol="pin",
                symbol_size=50,
                label_opts=opts.LabelOpts(
                    position="inside",
                    color="#fff",
                    font_weight="bold",
                    formatter=JsCode("function(params){return Math.round(params.value);}")
                )
            )
        )
        .set_global_opts(
            title_opts=opts.TitleOpts(
                title=f"{symbol} Price Chart", 
                subtitle=subtitle_text,
                pos_top="1%", pos_left="center"
            ),
            legend_opts=opts.LegendOpts(pos_top="6%", pos_left="center"),
            xaxis_opts=opts.AxisOpts(is_scale=True),
            yaxis_opts=opts.AxisOpts(is_scale=True),
            datazoom_opts=[opts.DataZoomOpts()],
            toolbox_opts=opts.ToolboxOpts(
                feature={
                    "dataZoom": {"yAxisIndex": "none"},
                    "restore": {},
                    "saveAsImage": {},
                }
            ),
        )
    )

    # Lines: MA, EMA, BOLL, VWAP
    ma50 = Line().add_xaxis(dates).add_yaxis("MA50", df['MA50'].tolist(),
        linestyle_opts=opts.LineStyleOpts(width=2, color="blue"),
        label_opts=opts.LabelOpts(is_show=False))

    ma200 = Line().add_xaxis(dates).add_yaxis("MA200", df['MA200'].tolist(),
        linestyle_opts=opts.LineStyleOpts(width=2, color="orange"),
        label_opts=opts.LabelOpts(is_show=False))

    ema20 = Line().add_xaxis(dates).add_yaxis("EMA20", df['EMA20'].tolist(),
        linestyle_opts=opts.LineStyleOpts(width=2, color="#8A2BE2"),
        label_opts=opts.LabelOpts(is_show=False))

    ema50 = Line().add_xaxis(dates).add_yaxis("EMA50", df['EMA50'].tolist(),
        linestyle_opts=opts.LineStyleOpts(width=2, color="#6B8E23"),
        label_opts=opts.LabelOpts(is_show=False))

    bb_mid = Line().add_xaxis(dates).add_yaxis("BOLL MID", df['MB'].tolist(),
        linestyle_opts=opts.LineStyleOpts(width=1.5, color="#7266BA"),
        label_opts=opts.LabelOpts(is_show=False))

    bb_upper = Line().add_xaxis(dates).add_yaxis("BOLL UP", df['UB'].tolist(),
        linestyle_opts=opts.LineStyleOpts(width=1, color="#45A3E5"),
        label_opts=opts.LabelOpts(is_show=False))

    bb_lower = Line().add_xaxis(dates).add_yaxis("BOLL LOW", df['LB'].tolist(),
        linestyle_opts=opts.LineStyleOpts(width=1, color="#45A3E5"),
        label_opts=opts.LabelOpts(is_show=False))

    vwap_line = Line().add_xaxis(dates).add_yaxis("VWAP", df['VWAP'].tolist(),
        linestyle_opts=opts.LineStyleOpts(width=2, color="#FF8C00"),
        label_opts=opts.LabelOpts(is_show=False))

    # Ichimoku lines: Tenkan, Kijun, SpanA, SpanB, Chikou
    tenkan_line = Line().add_xaxis(dates).add_yaxis("Tenkan-sen", df['Tenkan'].tolist(),
        linestyle_opts=opts.LineStyleOpts(width=1.5, color="red"),
        label_opts=opts.LabelOpts(is_show=False))

    kijun_line = Line().add_xaxis(dates).add_yaxis("Kijun-sen", df['Kijun'].tolist(),
        linestyle_opts=opts.LineStyleOpts(width=1.5, color="green"),
        label_opts=opts.LabelOpts(is_show=False))

    span_a_line = Line().add_xaxis(dates).add_yaxis("Senkou Span A", df['SpanA'].tolist(),
        linestyle_opts=opts.LineStyleOpts(width=1.5, color="#FF7F50"),
        label_opts=opts.LabelOpts(is_show=False))

    span_b_line = Line().add_xaxis(dates).add_yaxis("Senkou Span B", df['SpanB'].tolist(),
        linestyle_opts=opts.LineStyleOpts(width=1.5, color="#87CEFA"),
        label_opts=opts.LabelOpts(is_show=False))

    chikou_line = Line().add_xaxis(dates).add_yaxis("Chikou Span", df['Chikou'].tolist(),
        linestyle_opts=opts.LineStyleOpts(width=1.5, color="#708090"),
        label_opts=opts.LabelOpts(is_show=False))

    # Overlap all lines onto the Kline
    kline.overlap(ma50)
    kline.overlap(ma200)
    kline.overlap(ema20)
    kline.overlap(ema50)
    kline.overlap(bb_mid)
    kline.overlap(bb_upper)
    kline.overlap(bb_lower)
    kline.overlap(vwap_line)

    kline.overlap(tenkan_line)
    kline.overlap(kijun_line)
    kline.overlap(span_a_line)
    kline.overlap(span_b_line)
    kline.overlap(chikou_line)

    kline.set_series_opts(label_opts=opts.LabelOpts(is_show=False))

    return kline

def create_volume_chart(df: pd.DataFrame, volume_data: list) -> Bar:
    dates = df.index.strftime('%Y-%m-%d').tolist()
    volume_chart = (
        Bar()
        .add_xaxis(dates)
        .add_yaxis(
            series_name="Volume",
            y_axis=volume_data,
            itemstyle_opts=opts.ItemStyleOpts(
                color="#5793f3",
                border_color="#5793f3",
                border_width=0
            ),
            label_opts=opts.LabelOpts(is_show=False)
        )
        .set_global_opts(
            xaxis_opts=opts.AxisOpts(
                type_="category",
                grid_index=1,
                boundary_gap=False,
                axislabel_opts=opts.LabelOpts(is_show=False),
                axisline_opts=opts.AxisLineOpts(is_show=False),
                splitline_opts=opts.SplitLineOpts(is_show=False),
            ),
            yaxis_opts=opts.AxisOpts(
                grid_index=1,
                axislabel_opts=opts.LabelOpts(is_show=False),
                axisline_opts=opts.AxisLineOpts(is_show=False),
                splitline_opts=opts.SplitLineOpts(is_show=False),
            ),
            tooltip_opts=opts.TooltipOpts(trigger="axis", axis_pointer_type="cross"),
            legend_opts=opts.LegendOpts(is_show=False)
        )
    )
    return volume_chart

def render_chart(kline: Kline, volume: Bar, symbol: str):
    grid = Grid(init_opts=opts.InitOpts(theme=ThemeType.LIGHT, width="1400px", height="900px"))
    grid.add(kline, grid_opts=opts.GridOpts(pos_left="5%", pos_right="5%", pos_top="10%", height="60%"))
    grid.add(volume, grid_opts=opts.GridOpts(pos_left="5%", pos_right="5%", pos_top="75%", height="15%"))
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.expanduser(f"~/aapl_stock_charts_{timestamp}.html")
    grid.render(output_file)
    print(f"\nThe chart is saved to {output_file}")
    print("Open the file in your browser to view the chart.")

def main():
    data = download_stock_data(symbol, start_date, end_date)
    data = calculate_indicators(data)
    pe_ratio, market_cap = get_fundamentals(symbol)
    kline_data = prepare_kline_data(data)
    volume_data = prepare_volume_data(data)
    kline_chart = create_kline_chart(data, kline_data, symbol, pe_ratio, market_cap)
    volume_chart = create_volume_chart(data, volume_data)
    render_chart(kline_chart, volume_chart, symbol)
    print("\nAll tasks completed!")

if __name__ == "__main__":
    main()
