import pandas as pd
import yfinance as yf
import os

def calculate_benchmark_returns_to_file(periods, output_filename="returns_benchmark_sp500.txt", benchmark_ticker="^GSPC"):
    print(f"1. Downloading historical data for benchmark: {benchmark_ticker}...")

    start_date = min(p['start'] for p in periods)
    end_date = max(p['end'] for p in periods)

    try:
        benchmark_data = yf.download(benchmark_ticker, start=start_date, end=pd.to_datetime(end_date) + pd.DateOffset(days=1))
        if benchmark_data.empty:
            raise ValueError("No data found. Check ticker or date range.")
        print("Benchmark data downloaded successfully.")
    except Exception as e:
        print(f"Error occured downloading data: {e}")
        return

    results = []
    print("\n2. Calculating returns for each analysis period...")

    for period in periods:
        period_name = period['name']
        market_regime = period['label']
        start = period['start']
        end = period['end']

        try:
            period_data = benchmark_data.loc[start:end]
            if period_data.empty:
                print(f"   - WARNING: No trading data found for period: '{period_name}'")
                continue

            start_price = period_data['Close'].values[0].item()
            end_price = period_data['Close'].values[-1].item()

            period_return = (end_price / start_price) - 1

            results.append({
                "Period": period_name,
                "Market regime": market_regime,
                "Start price ($)": start_price,
                "End price ($)": end_price,
                "Benchmark return (%)": period_return * 100
            })
            print(f"   - Period '{period_name}': calculated return = {period_return:.2%}")

        except IndexError:
            print(f"   - WARNING: Insufficient data for period '{period_name}'.")

    if not results:
        print("\nNo results to save.")
        return

    results_df = pd.DataFrame(results)

    formatters = {
        'Start price ($)': '{:,.2f}'.format,
        'End price ($)': '{:,.2f}'.format,
        'Benchmark return (%)': '{:,.2f}%'.format
    }

    try:
        with open(output_filename, 'w', encoding='utf-8') as f:
            f.write(f"Benchmark returns ({benchmark_ticker}) per analysis period\n")
            f.write("=" * 70 + "\n\n")
            f.write(results_df.to_string(index=False, formatters=formatters))

        print(f"\nAnalysis complete! Results saved in '{output_filename}'.")
        print("\n" + "="*70)
        print("Summary:")
        print(results_df.to_string(index=False, formatters=formatters))
        print("="*70)

    except Exception as e:
        print(f"Error occured saving file: {e}")

periods_to_analyze = [
    {"name": "2000-08-01 - 2002-09-03", "label": "Bear Market", "start": "2000-08-01", "end": "2002-09-03"},
    {"name": "2002-09-03 - 2007-10-01", "label": "Bull Market", "start": "2002-09-03", "end": "2007-10-01"},
    {"name": "2007-10-01 - 2009-02-02", "label": "Bear Market", "start": "2007-10-01", "end": "2009-02-02"},
    {"name": "2009-02-02 - 2019-12-02", "label": "Bull Market", "start": "2009-02-02", "end": "2019-12-02"},
    {"name": "2019-12-02 - 2020-08-03", "label": "COVID V-Shape", "start": "2019-12-02", "end": "2020-08-03"},
    {"name": "2020-08-03 - 2021-12-01", "label": "Bull Market", "start": "2020-08-03", "end": "2021-12-01"},
    {"name": "2021-12-01 - 2022-09-01", "label": "Bear Market", "start": "2021-12-01", "end": "2022-09-01"}
]

if __name__ == "__main__":
    calculate_benchmark_returns_to_file(periods_to_analyze)