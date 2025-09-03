import pandas as pd
import re

def load_and_prepare_data(report_filename="complete_report.txt"):
    print(f"1. Reading and preparing data from '{report_filename}'...")
    try:
        with open(report_filename, 'r', encoding='utf-8') as f:
            report_text = f.read()
    except FileNotFoundError:
        print(f"Error occured: File '{report_filename}' not found.")
        return None

    features_data = []
    performance_data = []
    current_company, current_ticker, current_strategy = None, None, None
    parsing_table = None
    header = []

    for line in report_text.strip().split('\n'):
        line = line.strip()
        if not line: continue

        company_match = re.match(r"##\s+(.*)\s+\((.*)\)", line)
        if company_match:
            current_company, current_ticker = company_match.group(1).strip(), company_match.group(2).strip()
            parsing_table = None
            continue
        
        if "### Trade Details" in line:
            parsing_table = None
            continue

        if "### Historical Asset Features" in line:
            parsing_table = 'features'
            header = []
            continue
        
        strategy_match = re.match(r"####\s+Strategy:\s+(.*)", line)
        if strategy_match:
            current_strategy = strategy_match.group(1).strip()
            parsing_table = 'performance'
            header = []
            continue

        if parsing_table and line.startswith('|'):
            cells = [cell.strip() for cell in line.split('|')[1:-1]]
            if len(cells) < 2: continue

            if "Period" in cells[0]:
                header = cells
                continue
            if '---' in cells[0]: continue

            row_data = dict(zip(header, cells))
            row_data['Company'], row_data['Ticker'] = current_company, current_ticker

            if parsing_table == 'features':
                features_data.append(row_data)
            elif parsing_table == 'performance':
                row_data['Strategy'] = current_strategy
                performance_data.append(row_data)
    
    features_df = pd.DataFrame(features_data)
    performance_df = pd.DataFrame(performance_data)

    if features_df.empty or performance_df.empty:
        print("Error occured: Insufficient feature or performance data found.")
        return None

    features_df['Beta'] = pd.to_numeric(features_df['Beta'], errors='coerce')
    features_df['Historical Volatility'] = pd.to_numeric(features_df['Historical Volatility'].str.replace('%', '', regex=False), errors='coerce') / 100.0
    features_df['Average Volume ($)'] = pd.to_numeric(features_df['Average Volume ($)'].str.replace('$', '', regex=False).str.replace('M', 'e6', regex=False).str.replace(',', '', regex=False), errors='coerce')
    performance_df['P&L %'] = pd.to_numeric(performance_df['P&L %'].str.replace('%', '', regex=False), errors='coerce') / 100.0
    
    merged_df = pd.merge(performance_df, features_df.drop(columns=['Company', 'Market Phase']), on=['Period', 'Ticker'])
    merged_df.dropna(subset=['Beta', 'Historical Volatility', 'Average Volume ($)', 'P&L %'], inplace=True)
    
    print("Successfully loaded and merged the data.")
    return merged_df

def analyze_performance_by_archetype(df, benchmark_df):
    if df is None or df.empty:
        print("DInvalid DataFrame. Analysis interrupted.")
        return

    print("\n2. Creating 'Low', 'Medium', 'High' groups for each feature...")
    labels = ['Low', 'Medium', 'High']
    df['Beta_Level'] = df.groupby('Period')['Beta'].transform(lambda x: pd.qcut(x, q=3, labels=labels, duplicates='drop'))
    df['Volatility_Level'] = df.groupby('Period')['Historical Volatility'].transform(lambda x: pd.qcut(x, q=3, labels=labels, duplicates='drop'))
    df['Volume_Level'] = df.groupby('Period')['Average Volume ($)'].transform(lambda x: pd.qcut(x, q=3, labels=labels, duplicates='drop'))
    df['Archetype'] = df['Beta_Level'].astype(str) + '-' + df['Volatility_Level'].astype(str) + '-' + df['Volume_Level'].astype(str)
    print("Groups created.")

    print("\n3. Calculating average performance...")
    archetype_performance = df.groupby(['Period', 'Archetype', 'Strategy'])['P&L %'].agg(['mean', 'count']).reset_index()
    archetype_performance.rename(columns={'mean': 'Avg_P&L_%', 'count': 'Num_Observations'}, inplace=True)
    print("Performance calculation completed.")

    print("\n4. Calculating Alpha...")
    analysis_df = pd.merge(archetype_performance, benchmark_df, on='Period', how='left')
    
    analysis_df['Alpha_%'] = analysis_df['Avg_P&L_%'] - analysis_df['Benchmark_Return_%']
    
    def format_alpha(row):
        strategy_return = row['Avg_P&L_%']
        benchmark_return = row['Benchmark_Return_%']
        alpha = row['Alpha_%']
        
        emoji = '✅' if alpha >= 0 else '❌'
        
        s_ret_str = f"{strategy_return:.2%}"
        b_ret_str = f"({benchmark_return:.2%})"
        alpha_str = f"{alpha:+.2%}"
        
        return f"{emoji} Alpha = {s_ret_str} - {b_ret_str} = {alpha_str}"

    analysis_df['Alpha_Analysis'] = analysis_df.apply(format_alpha, axis=1)
    print("Alpha calculation completed.")

    output_filename = "performance_by_archetype_with_alpha.csv"
    try:
        analysis_df['Avg_P&L_%%'] = analysis_df['Avg_P&L_%'].apply(lambda x: f"{x:.2%}")

        final_columns = ['Period', 'Archetype', 'Strategy', 'Avg_P&L_%', 'Num_Observations', 'Alpha_Analysis']
        final_df = analysis_df[final_columns]

        final_df.to_csv(output_filename, index=False)
        print(f"\nAnalysis completed! Results saved in '{output_filename}'.")
        
        print(final_df.head(10).to_string())
    except Exception as e:
        print(f"Erorr occured saving file: {e}")

if __name__ == '__main__':
    benchmark_dict = {
        'Period': [
            "2000-08-01 - 2002-09-03", "2002-09-03 - 2007-10-01", "2007-10-01 - 2009-02-02",
            "2009-02-02 - 2019-12-02", "2019-12-02 - 2020-08-03", "2020-08-03 - 2021-12-01",
            "2021-12-01 - 2022-09-01"
        ],
        'Benchmark_Return_%': [
            -0.3895, 0.7620, -0.4664, 2.7724, 0.0580, 0.3698, -0.1210
        ]
    }
    benchmark_df = pd.DataFrame(benchmark_dict)

    full_merged_data = load_and_prepare_data("complete_report.txt")
    if full_merged_data is not None:
        analyze_performance_by_archetype(full_merged_data, benchmark_df)