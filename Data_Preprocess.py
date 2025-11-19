import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def process_market_data(file_path):
    # 1. 读取数据
    # 即使原数据看起来是数字，也可能包含非数值字符，errors='coerce' 会将这些强制转为 NaN 方便清洗
    df = pd.read_csv(file_path)
    
    # 2. 数据清洗
    # 确保 'SP500' 列是数值型
    df['SP500'] = pd.to_numeric(df['SP500'], errors='coerce')
    
    # 删除包含 NaN 的行 (对应只有日期没有价格的情况，如假期)
    # 您的数据中有一些行是空的，这一步非常关键
    df = df.dropna(subset=['SP500'])
    
    # 转换日期格式
    df['observation_date'] = pd.to_datetime(df['observation_date'])
    
    # 按日期排序，确保计算收益率时顺序正确
    df = df.sort_values('observation_date')
    
    # 3. 计算对数收益率
    # 公式: r_t = ln(P_t) - ln(P_{t-1})
    # shift(1) 将数据下移一行，用于计算前一天的价格
    df['log_return'] = np.log(df['SP500'] / df['SP500'].shift(1))
    
    # 计算收益率后，第一行会变成 NaN (因为它没有前一天的数据)，需要删除
    df_final = df.dropna()
    
    return df_final

# --- 使用示例 ---
file_path = 'SP500.csv'  # 请确保文件名与上传的一致
clean_data = process_market_data(file_path)

# 查看前几行数据
print("处理后的数据预览:")
print(clean_data.head())

# 查看基本统计量 (均值、标准差等)
print("\n数据统计描述:")
print(clean_data['log_return'].describe())

# --- 可视化检查 (推荐) ---
plt.figure(figsize=(12, 5))
plt.plot(clean_data['observation_date'], clean_data['log_return'], linewidth=0.5)
plt.title('Daily Log Returns of S&P 500')
plt.xlabel('Date')
plt.ylabel('Log Return')
plt.grid(True, alpha=0.3)
plt.show()

# 将清洗后的数据保存到新文件，不保存索引(index)
clean_data.to_csv('SP500_clean.csv', index=False)

print("文件已保存为 SP500_clean.csv")