## 第一步 通过DOI获取文献的PDF链接

# 通过读取./xls_folder中的Excel文件DOI列的获取文献的DOI
# 将获取到的DOI与'https://api.unpaywall.org/v2/'和‘'email=lipengfa12@gmail.com'进行拼接
# 得到https://api.unpaywall.org/v2/DOI?email=lipengfa12@gmail.com

# 第二步 通过PDF链接获取pdf文件

import os
import pandas as pd
import requests
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

excel_path = './xls_folder/en_version_1.xlsx'
df = pd.read_excel(excel_path)
doi_list = df['DOI'].dropna().unique()  # 假设DOI列名为"DOI"

pdf_links = []
success_dois = []
failed_dois = []
titles = []

for doi in doi_list:
    doi_str = str(doi).strip()
    api_url = f'https://api.unpaywall.org/v2/{doi_str}?email=lipengfa12@gmail.com'
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
with open('./log/pdf_link/all_pdf_links.txt', 'w', encoding='utf-8') as f:
    f.write('\n'.join(pdf_links))

with open('./log/pdf_link/success_pdfs.txt', 'w', encoding='utf-8') as f:
    f.write('\n'.join(success_dois))

with open('./log/pdf_link/failed_dois.txt', 'w', encoding='utf-8') as f:
    f.write('\n'.join(failed_dois))

with open('./log/pdf_link/titles.txt', 'w', encoding='utf-8') as f:
    f.write('\n'.join(titles))
if len(success_dois) == len(titles):
    print("所有文献标题已成功获取")
print("所有PDF链接已写入 all_pdf_links.txt")
print("成功DOI已写入 success_pdfs.txt")
print("失败DOI已写入 failed_dois.txt")
print("文献标题已写入 titles.txt")

# # 重新处理失败的DOI
# import os

# re_dir = './log/pdf_link/all_pdf_links/re'
# os.makedirs(re_dir, exist_ok=True)

# failed_dois_path = './log/pdf_link/failed_dois.txt'
# if os.path.exists(failed_dois_path):
#     with open(failed_dois_path, 'r', encoding='utf-8') as f:
#         retry_dois = [line.strip() for line in f if line.strip()]
#     pdf_links_re = []
#     success_dois_re = []
#     failed_dois_re = []
#     titles_re = []
#     for doi in retry_dois:
#         api_url = f'https://api.unpaywall.org/v2/{doi}?email=lipengfa12@gmail.com'
#         try:
#             resp = requests.get(api_url, timeout=10, verify=False)
#             if resp.status_code != 200:
#                 failed_dois_re.append(doi)
#                 continue
#             data = resp.json()
#             best_oa = data.get('best_oa_location')
#             title = data.get('title')
#             pdf_url = best_oa.get('url_for_pdf') if best_oa else None
#             if pdf_url:
#                 pdf_links_re.append(pdf_url)
#                 success_dois_re.append(doi)
#                 titles_re.append(title)
#                 print(f"重试已收集: {pdf_url}")
#             else:
#                 failed_dois_re.append(doi)
#         except Exception:
#             failed_dois_re.append(doi)
#     with open(os.path.join(re_dir, 'all_pdf_links.txt'), 'w', encoding='utf-8') as f:
#         f.write('\n'.join(pdf_links_re))
#     with open(os.path.join(re_dir, 'success_pdfs.txt'), 'w', encoding='utf-8') as f:
#         f.write('\n'.join(success_dois_re))
#     with open(os.path.join(re_dir, 'failed_dois.txt'), 'w', encoding='utf-8') as f:
#         f.write('\n'.join(failed_dois_re))
#     with open(os.path.join(re_dir, 'titles.txt'), 'w', encoding='utf-8') as f:
#         f.write('\n'.join(titles_re))
#     print("重试PDF链接已写入 all_pdf_links/re/all_pdf_links.txt")
#     print("重试成功DOI已写入 all_pdf_links/re/success_pdfs.txt")
#     print("重试失败DOI已写入 all_pdf_links/re/failed_dois.txt")
#     print("重试文献标题已写入 all_pdf_links/re/titles.txt")
