# 1、从./pdf_data/获取PDF文件，转换为文本文件，以原pdf名称命名

# 2、将文本文件转换为json文件，每个txt文件对应一个json对象，包含以下字段：Full Text, Article Title, Source Title, Keywords Plus
# Abstract, Publisher, Publication Year, DOI, Open Access Designations
# 其中Full Text字段为文本文件的内容，其余字段需要从./xlsx_data/all_papers.xlsx中获取，匹配字段为txt的文件名和xlsx的Article Title字段
# 记录读取的pdf数量，Article Title字段匹配优先完全匹配，如何找不到可以使用模糊匹配（阈值95%）否则返回失败的文件标题名

# 3、将json文件转换为parquet文件，存储在./parquet_data/目录下
import os
import pdfplumber
import pandas as pd
import json
from rapidfuzz import fuzz, process
import pyarrow as pa
import pyarrow.parquet as pq
import time
import re
from multi_column import column_boxes
import pymupdf

input_dir = 'path/to/data/papers/'
output_dir = 'path/to/data/txt_data/'
JSON_DIR = 'path/to/data/json_data/'
PARQUET_DIR = 'path/to/data/parquet_data/'
XLSX_PATH = 'path/to/data/xlsx_data/all_papers.xlsx'
LOG_DIR = 'path/to/output/log_parquet/'

os.makedirs(output_dir, exist_ok=True)
os.makedirs(JSON_DIR, exist_ok=True)
os.makedirs(PARQUET_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# # 1. PDF转TXT

def pdf_to_text_pymupdf(input_dir, output_dir):
    total_time = 0
    processed_count = 0
    skipped_count = 0
    
    for filename in os.listdir(input_dir):
        if not filename.lower().endswith('.pdf'):  # 只处理PDF文件
            continue
            
        pdf_path = os.path.join(input_dir, filename)
        txt_filename = os.path.splitext(filename)[0] + '.txt'
        txt_path = os.path.join(output_dir, txt_filename)
        
        # 检查TXT文件是否已存在
        if os.path.exists(txt_path):
            print(f"跳过已存在的文件: {txt_filename}")
            skipped_count += 1
            continue
        
        print(f"正在转换: {filename}")
        start_time = time.time()
        
        try:
            doc = pymupdf.open(pdf_path)
            full_text = ""
            for page in doc:
                bboxes = column_boxes(
                    page, 
                    footer_margin=50, 
                    no_image_text=True
                )
                
                for rect in bboxes:
                    full_text += page.get_text(
                        "text", 
                        clip=rect, 
                        sort=True
                    )
                full_text += "\n" + "-" * 80 + "\n"
            
            with open(txt_path, 'w', encoding='utf-8') as f:
                f.write(full_text)
            
            doc.close()  # 记得关闭文档
            processed_count += 1
            
        except Exception as e:
            print(f"处理文件 {filename} 时出错: {e}")
            continue
            
        end_time = time.time()
        total_time += (end_time - start_time)
    
    print(f"PDF转换完成: 处理了 {processed_count} 个文件，跳过了 {skipped_count} 个文件，总耗时: {total_time:.2f}秒")
pdf_to_text_pymupdf(input_dir, output_dir)

# 2. TXT+Excel转JSON
df = pd.read_excel(XLSX_PATH, engine='openpyxl')
df['Article Title'] = df['Article Title'].astype(str)
fields = [
    "Full Text", "Article Title", "Source Title", "Keywords Plus",
    "Abstract", "Publisher", "Publication Year", "DOI", "Open Access Designations"
]
failed_files = []
for txt_file in os.listdir(output_dir):
    if not txt_file.endswith('.txt'):
        continue
    print(f"正在处理: {txt_file}")  # 加打印
    t0 = time.time()
    txt_path = os.path.join(output_dir, txt_file)
    with open(txt_path, 'r', encoding='utf-8') as f:
        full_text = f.read()
    full_text = re.sub(r'(?<!\n)\n(?!\n)', ' ', full_text)
    base_name = os.path.splitext(txt_file)[0]
    # 完全匹配
    match = df[df['Article Title'] == base_name]
    if match.empty:
        # 模糊匹配
        choices = df['Article Title'].tolist()
        # 监控模糊匹配耗时
        t1 = time.time()
        best, score, idx = process.extractOne(base_name, choices, scorer=fuzz.ratio)
        print(f"模糊匹配耗时: {time.time()-t1:.2f}s, 匹配分数: {score}, best: {best}")
        if score >= 95:
            match = df.iloc[[idx]]
        else:
            failed_files.append(base_name)
            continue
    print(f"处理{txt_file}总耗时: {time.time()-t0:.2f}s")
    row = match.iloc[0]
    def safe_get(key):
        v = row.get(key, "")
        if pd.isna(v):
            return ""
        if isinstance(v, (int, float, bool)):
            return v
        return str(v)
    data = {
        "Full Text": full_text,
        "Article Title": safe_get("Article Title"),
        "Source Title": safe_get("Source Title"),
        "Keywords Plus": safe_get("Keywords Plus"),
        "Abstract": safe_get("Abstract"),
        "Publisher": safe_get("Publisher"),
        "Publication Year": safe_get("Publication Year"),
        "DOI": safe_get("DOI"),
        "Open Access Designations": safe_get("Open Access Designations")
    }
    json_path = os.path.join(JSON_DIR, base_name + '.json')
    with open(json_path, 'w', encoding='utf-8') as jf:
        json.dump(data, jf, ensure_ascii=False, indent=2)

# 保存无法匹配的PDF文件名
with open(os.path.join(LOG_DIR, 'failed_files.txt'), 'w', encoding='utf-8') as f:
    for name in failed_files:
        f.write(name + '\n')
# 将无法匹配的PDF移动到 not_match 文件夹
NOT_MATCH_DIR = '/data/home/aim/aim_lipengfa/aim_lipengfa/data/not_match'
os.makedirs(NOT_MATCH_DIR, exist_ok=True)
for name in failed_files:
    src_pdf = os.path.join(input_dir, name + '.pdf')
    dst_pdf = os.path.join(NOT_MATCH_DIR, name + '.pdf')
    if os.path.exists(src_pdf):
        try:
            os.rename(src_pdf, dst_pdf)
        except Exception as e:
            print(f"移动文件失败: {src_pdf} -> {dst_pdf}, 错误: {e}")

# print(f"PDF总数: {len(pdf_files)}，无法处理的PDF: {error_pdfs}")
print(f"无法匹配元数据的PDF: {failed_files}")

# 3. JSON转Parquet
json_files = [f for f in os.listdir(JSON_DIR) if f.endswith('.json')]
records = []
for jf in json_files:
    with open(os.path.join(JSON_DIR, jf), 'r', encoding='utf-8') as f:
        records.append(json.load(f))
if records:
    table = pa.Table.from_pandas(pd.DataFrame(records))
    pq.write_table(table, os.path.join(PARQUET_DIR, 'all_papers.parquet'))

