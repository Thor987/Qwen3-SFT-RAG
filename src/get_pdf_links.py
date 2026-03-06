## 通过DOI获取文献的PDF链接
# 通过读取./data/xls_folder中的Excel文件DOI列的获取文献的DOI
# 将获取到的DOI与'https://api.unpaywall.org/v2/'和‘'email=XXXX@gmail.com'进行拼接
# 得到https://api.unpaywall.org/v2/DOI?email=XXXX@gmail.com

## XXXX@gmail.com替换为真实的邮箱

import os
import pandas as pd
import requests
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

excel_path = './data/xls_folder/en_version_1.xlsx'
df = pd.read_excel(excel_path)
doi_list = df['DOI'].dropna().unique()  # 假设DOI列名为"DOI"

pdf_links = []
success_dois = []
failed_dois = []
titles = []

for doi in doi_list:
    doi_str = str(doi).strip()
    api_url = f'https://api.unpaywall.org/v2/{doi_str}?email=XXXX@gmail.com'
    try:
        resp = requests.get(api_url, timeout=10, verify=False)
        if resp.status_code != 200:
            failed_dois.append(doi_str)
            continue
        data = resp.json()
        best_oa = data.get('best_oa_location')
        title = data.get('title')
        pdf_url = best_oa.get('url_for_pdf') if best_oa else None
        if pdf_url:
            pdf_links.append(pdf_url)
            success_dois.append(doi_str)
            titles.append(title)
            print(f"已收集: {pdf_url}")
        else:
            failed_dois.append(doi_str)
    except Exception:
        failed_dois.append(doi_str)

# 写入txt文件
with open('./output/log/pdf_link/all_pdf_links.txt', 'w', encoding='utf-8') as f:
    f.write('\n'.join(pdf_links))

with open('./output/log/pdf_link/success_pdfs.txt', 'w', encoding='utf-8') as f:
    f.write('\n'.join(success_dois))

with open('./output/log/pdf_link/failed_dois.txt', 'w', encoding='utf-8') as f:
    f.write('\n'.join(failed_dois))

with open('./output/log/pdf_link/titles.txt', 'w', encoding='utf-8') as f:
    f.write('\n'.join(titles))
if len(success_dois) == len(titles):
    print("所有文献标题已成功获取")
print("所有PDF链接已写入 all_pdf_links.txt")
print("成功DOI已写入 success_pdfs.txt")
print("失败DOI已写入 failed_dois.txt")
print("文献标题已写入 titles.txt")
