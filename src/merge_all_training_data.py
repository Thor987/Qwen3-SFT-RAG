import pandas as pd

# 读取 Literature.parquet 文件
df1 = pd.read_parquet('./data/collect_data/all_papers.parquet')
df12 = pd.read_json('./data/collect_data/data.jsonl', lines=True)

# 将df12[text]和df1[text]合并
# 分别再添加一列source，标识数据来源，分为literature和wiki
df1['source'] = 'literature'
df12['source'] = 'wiki'

# 合并两个DataFrame
merged_df = pd.concat([df1, df12], ignore_index=True)
with open('./data/collect_data/merged_data.jsonl', 'w', encoding='utf-8') as f:
    for record in merged_df.to_dict(orient='records'):

        f.write(f"{pd.io.json.dumps(record, ensure_ascii=False)}\n")
