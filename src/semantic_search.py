"""
地质灾害语义检索模型

该模块实现了基于Sentence-BERT的语义检索系统，用于：
1. 将所有专家标注数据编码为向量，并使用FAISS构建索引。
2. 从./data/collect_data/merged_data.jsonl文件中高效检索语义上最匹配的文本数据。
"""

import json
import pickle
import re
import time
import faiss
from typing import List, Dict
from pathlib import Path
import numpy as np
from tqdm import tqdm
import logging
from sentence_transformers import SentenceTransformer

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class SemanticSearcher:
    """地质灾害语义检索模型"""
    
    def __init__(self, merged_data_path: str = "./data/collect_data/merged_data.jsonl", model_name: str = 'all-MiniLM-L6-v2'):
        self.merged_data_path = Path(merged_data_path)
        logger.info(f"正在加载语义模型: {model_name} (这可能需要一些时间)...")
        self.model = SentenceTransformer(model_name)
        logger.info("语义模型加载完成。")
        
        self.index = None  # FAISS 索引
        self.corpus_info = []  # 存储语料库原文及类别信息

    def load_annotation_data(self) -> List[Dict]:
        """加载并合并所有标注数据到一个列表中"""
        all_items = []
        MAX_SAMPLES_PER_DISASTER = 205
        
        expert_path = Path("./annotation_data_expert")
        llm_path = Path("./annotation_data_llm")
        
        if not expert_path.exists() and not llm_path.exists():
            logger.error("专家标注和LLM扩增数据路径都不存在。")
            return []
        
        logger.info("开始加载和合并专家与LLM标注数据...")

        def load_data_from_path(data_path: Path) -> Dict[str, List[Dict]]:
            path_data = {}
            if not data_path.exists(): return path_data
            for category_folder in data_path.iterdir():
                if category_folder.is_dir():
                    for json_file in category_folder.glob("*.json"):
                        with open(json_file, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                        disaster_key = f"{category_folder.name}_{json_file.stem}"
                        path_data[disaster_key] = data if isinstance(data, list) else [data]
            return path_data
        
        expert_data = load_data_from_path(expert_path)
        llm_data = load_data_from_path(llm_path)
        
        all_disaster_keys = set(expert_data.keys()) | set(llm_data.keys())
        
        for disaster_key in all_disaster_keys:
            combined_data = expert_data.get(disaster_key, []) + llm_data.get(disaster_key, [])
            if len(combined_data) > MAX_SAMPLES_PER_DISASTER:
                combined_data = combined_data[:MAX_SAMPLES_PER_DISASTER]
            all_items.extend(combined_data)
        
        logger.info(f"数据加载与合并完成。总计 {len(all_items)} 条记录。")
        return all_items

    def build_index(self, annotation_items: List[Dict]):
        """使用所有标注数据构建FAISS索引"""
        if not annotation_items:
            logger.error("没有标注数据可用于构建索引。")
            return

        logger.info("开始为标注数据构建FAISS索引...")
        
        # 1. 提取文本并存储信息
        corpus_texts = []
        for idx, item in enumerate(annotation_items):
            corpus_texts.append(item.get('text', ''))
            self.corpus_info.append({
                'text': item.get('text', ''),
                'major_category': item.get('major categories', 'Unknown'),
                'sub_category': item.get('sub-categories', 'Unknown').strip()
            })
            
        # 2. 将文本编码为向量
        logger.info(f"正在将 {len(corpus_texts)} 条标注文本编码为向量...")
        corpus_embeddings = self.model.encode(corpus_texts, convert_to_numpy=True, show_progress_bar=True)
        
        # 3. 构建FAISS索引
        embedding_dim = corpus_embeddings.shape[1]
        self.index = faiss.IndexFlatIP(embedding_dim)  # 使用内积作为相似度度量
        
        # 4. 归一化向量并添加到索引
        faiss.normalize_L2(corpus_embeddings)
        self.index.add(corpus_embeddings)
        
        logger.info(f"FAISS索引构建完成。索引中包含 {self.index.ntotal} 个向量。")

    def search(self, top_k: int = 40180) -> List[Dict]:
        """使用语义模型从merged_data.jsonl中检索最匹配的文本"""
        if not self.index:
            logger.error("FAISS索引未构建，请先调用 build_index 方法。")
            return []

        logger.info("开始使用语义模型进行检索...")
        
        # 1. 加载所有待检索的数据
        query_items = []
        with open(self.merged_data_path, 'r', encoding='utf-8') as f:
            for line in tqdm(f, desc="加载待检索数据"):
                try:
                    query_items.append(json.loads(line.strip()))
                except json.JSONDecodeError:
                    continue
        
        query_texts = [item.get('text', '') for item in query_items]
        logger.info(f"总共加载了 {len(query_texts)} 条数据用于检索。")

        # 2. 将所有待检索文本编码为向量
        logger.info(f"正在将 {len(query_texts)} 条待检索文本编码为向量 (此步骤可能非常耗时)...")
        query_embeddings = self.model.encode(query_texts, convert_to_numpy=True, show_progress_bar=True, batch_size=64)
        faiss.normalize_L2(query_embeddings)

        # 3. 在FAISS中进行批量搜索
        logger.info("在FAISS索引中进行批量搜索...")
        # 我们为每个查询只找最相似的1个结果
        distances, indices = self.index.search(query_embeddings, 1)

        # 4. 整理结果并排序
        results = []
        for i in range(len(query_texts)):
            best_corpus_idx = indices[i][0]
            score = distances[i][0]
            
            results.append({
                'score': float(score),
                'merged_text': query_texts[i],
                'source_data': query_items[i],
                'best_corpus_idx': int(best_corpus_idx)
            })
            
        # 按语义分数从高到低排序
        results.sort(key=lambda x: x['score'], reverse=True)
        
        # 返回top_k个结果
        final_results = results[:top_k]
        
        logger.info("语义检索完成。")
        return final_results

    def save_results(self, results: List[Dict], output_file: str = "./retrieve/top_semantic_matches.json"):
        """保存语义检索的结果"""
        output_path = Path(output_file)
        output_path.parent.mkdir(exist_ok=True, parents=True)
        logger.info(f"开始保存 {len(results)} 条检索结果到文件: {output_path}")
        
        formatted_results = []
        for i, result in enumerate(results):
            formatted_result = {
                'rank': i + 1,
                'score': result['score'],
                'text': result['merged_text'],
                'source_info': result['source_data']
            }
            # 添加匹配的最佳专家标注信息
            if result['best_corpus_idx'] < len(self.corpus_info):
                matched_info = self.corpus_info[result['best_corpus_idx']]
                formatted_result['matched_annotation'] = {
                    'major_category': matched_info['major_category'],
                    'sub_category': matched_info['sub_category'],
                    'matched_text': matched_info['text']
                }
            formatted_results.append(formatted_result)
            
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(formatted_results, f, ensure_ascii=False, indent=2)
        logger.info("文件保存完成。")

    def save_index(self, index_path: str = "./faiss_index.bin", info_path: str = "./corpus_info.pkl"):
        """保存FAISS索引和语料库信息"""
        logger.info(f"正在保存FAISS索引到 {index_path}")
        faiss.write_index(self.index, index_path)
        with open(info_path, 'wb') as f:
            pickle.dump(self.corpus_info, f)
        logger.info("索引和语料库信息保存完成。")


def main():
    """主函数"""
    start_time = time.time()
    
    searcher = SemanticSearcher()
    
    # 加载标注数据并构建/保存索引
    annotation_items = searcher.load_annotation_data()
    if not annotation_items: return
    
    searcher.build_index(annotation_items)
    searcher.save_index()
    
    # 执行语义检索
    top_results = searcher.search(top_k=140180)
    
    if top_results:
        searcher.save_results(top_results)
        logger.info("\n=== 检索完成！统计信息 ===")
        logger.info(f"总共检索到 {len(top_results)} 条数据")
        logger.info(f"最高得分: {top_results[0]['score']:.4f}")
        logger.info(f"最低得分: {top_results[-1]['score']:.4f}")
    else:
        logger.warning("未找到匹配结果")
        
    total_time = time.time() - start_time
    logger.info(f"\n程序总运行时间: {total_time / 60:.2f} 分钟")


if __name__ == "__main__":

    main()
