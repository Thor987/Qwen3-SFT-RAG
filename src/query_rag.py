import json
import os
import re
import torch
import torch.nn.functional as F
import chromadb
from typing import List, Dict, Any
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM, AutoModelForSequenceClassification
from transformers import TextIteratorStreamer
from threading import Thread


# ============================================================
#  Qwen3-Embedding 封装类
# ============================================================
class Qwen3Embedder:
    def __init__(self, model_path: str, device: str = "cuda"):
        print(f"从 '{model_path}' 加载 Qwen3-Embedding 模型到 {device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, padding_side="left", trust_remote_code=True
        )
        # 修复：为 Qwen3-Embedding 完整设置 padding token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            
        self.model = AutoModel.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map=device,
            trust_remote_code=True,
        )
        self.model.eval()
        self.device = device
        print("✅ Qwen3-Embedding 模型加载完成。")

    @staticmethod
    def _last_token_pool(last_hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        left_padding = attention_mask[:, -1].sum() == attention_mask.shape[0]
        if left_padding:
            return last_hidden_states[:, -1]
        else:
            sequence_lengths = attention_mask.sum(dim=1) - 1
            batch_size = last_hidden_states.shape[0]
            return last_hidden_states[
                torch.arange(batch_size, device=last_hidden_states.device),
                sequence_lengths,
            ]

    def encode(
        self,
        texts,
        normalize_embeddings: bool = True,
        is_query: bool = False,
        batch_size: int = 8,
        task: str = "检索与问题相关的工程规范条文",
    ):
        if isinstance(texts, str):
            texts = [texts]
        if is_query:
            texts = [f"Instruct: {task}\nQuery: {t}" for t in texts]

        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            encoded = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=4096,
                return_tensors="pt",
            ).to(self.model.device)
            with torch.no_grad():
                outputs = self.model(**encoded)
            embeddings = self._last_token_pool(
                outputs.last_hidden_state, encoded["attention_mask"]
            )
            if normalize_embeddings:
                embeddings = F.normalize(embeddings, p=2, dim=1)
            all_embeddings.append(embeddings.cpu().float())
        
        result = torch.cat(all_embeddings, dim=0)
        return result[0].numpy() if result.shape[0] == 1 else result.numpy()


# ============================================================
#  Qwen3-Reranker 封装类
# ============================================================
class Qwen3Reranker:
    def __init__(self, model_path: str, device: str = "cuda"):
        print(f"从 '{model_path}' 加载 Qwen3-Reranker 模型到 {device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True
        )
        # 修复：为 Qwen3-Reranker 完整设置 padding token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map=device,
            trust_remote_code=True,
        )
        self.model.eval()
        print("✅ Qwen3-Reranker 模型加载完成。")

    def predict(self, pairs: List[List[str]], batch_size: int = 1) -> List[float]:
        """对 (查询, 文档) 对进行打分。强制使用 batch_size=1 避免 padding 问题。"""
        all_scores = []
        with torch.no_grad():
            for i in range(0, len(pairs), batch_size):
                batch = pairs[i : i + batch_size]
                encoded = self.tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=4096,
                    return_tensors="pt",
                ).to(self.model.device)
                
                scores = self.model(**encoded, return_dict=True).logits.view(-1).float()
                all_scores.extend(scores.cpu().numpy().tolist())
        return all_scores


# ============================================================
#  主 RAG 查询系统
# ============================================================
class RAGQuerySystem:
    def __init__(self, config: Dict[str, str]):
        self.config = config
        print("初始化RAG查询系统 (Qwen3-4B)...")
        self._load_models()
        self._connect_to_db()
        print("✅ RAG查询系统初始化完成。")

    def _load_models(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.embedder = Qwen3Embedder(self.config["embedding_model_path"], device=device)
        self.reranker = Qwen3Reranker(self.config["rerank_model_path"], device=device)
        
        print(f"从 '{self.config['qwen_model_path']}' 加载Qwen-LLM...")
        self.qwen_tokenizer = AutoTokenizer.from_pretrained(self.config['qwen_model_path'], trust_remote_code=True)
        self.qwen_model = AutoModelForCausalLM.from_pretrained(
            self.config['qwen_model_path'],
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )

    def _connect_to_db(self):
        db_client = chromadb.PersistentClient(path=self.config['db_path'])
        self.collection = db_client.get_collection(name=self.config['collection_name'])
        print(f"✅ 成功连接到集合 '{self.config['collection_name']}'，包含 {self.collection.count()} 个文档。")

    def retrieve_and_rerank(self, query: str, retrieve_top_k: int = 10, rerank_top_k: int = 3) -> List[Dict[str, Any]]:
        clean_query = re.sub(r"你是一位.*?工程师。|请严格依据.*?回答问题。|这是一个单选题，.*答案：X|（\s*）|\n", "", query, flags=re.DOTALL).strip()
        search_query = clean_query if clean_query else query[:100]
        print(f"\n1. 清洗后的检索关键词: '{search_query}'")

        # 检索 - 查询时需附加任务指令 (is_query=True)
        query_embedding = self.embedder.encode(search_query, is_query=True).tolist()
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=retrieve_top_k,
            include=["metadatas", "documents"]
        )
        
        retrieved_docs = [{'id': r_id, 'content': doc, 'metadata': meta} for r_id, doc, meta in zip(results['ids'][0], results['documents'][0], results['metadatas'][0])]
        print(f"检索到 {len(retrieved_docs)} 个候选文档。")

        # 重排序
        pairs = [[query, doc['content']] for doc in retrieved_docs]
        scores = self.reranker.predict(pairs)
        for doc, score in zip(retrieved_docs, scores):
            doc['rerank_score'] = score
            
        reranked_docs = sorted(retrieved_docs, key=lambda x: x['rerank_score'], reverse=True)
        final_docs = reranked_docs[:rerank_top_k]
        print(f"重排序后保留 {len(final_docs)} 个最相关文档。")
        return final_docs

    def build_rag_prompt(self, query: str, reranked_docs: List[Dict[str, Any]]) -> str:
        print("3. 构建RAG prompt...")
        context_parts = [f"参考文档 {i+1} (来源: 文件 {doc['metadata']['source_file']}, 章节 {doc['metadata']['original_section_id']}):\n{doc['content']}" for i, doc in enumerate(reranked_docs)]
        context = "\n\n".join(context_parts)
        return f"请根据以下参考文档来回答问题。请优先使用参考文档中的信息，并清晰、准确地进行回答。\n\n[参考文档开始]\n{context}\n[参考文档结束]\n\n[问题]\n{query}"

    def generate_answer(self, prompt: str) -> str:
        print("4. 使用LoRA模型生成答案...", flush=True)
        messages = [{"role": "user", "content": prompt}]
        text = self.qwen_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        model_inputs = self.qwen_tokenizer([text], return_tensors="pt").to(self.qwen_model.device)
        streamer = TextIteratorStreamer(self.qwen_tokenizer, skip_prompt=True, skip_special_tokens=True)
        generation_kwargs = dict(**model_inputs, streamer=streamer, max_new_tokens=1024, do_sample=False, repetition_penalty=1.1)
        thread = Thread(target=self.qwen_model.generate, kwargs=generation_kwargs)
        thread.start()
        response = "".join([new_text for new_text in streamer])
        thread.join()
        if not response: print("⚠️ 警告: 模型输出了空字符串。")
        return response.strip()

# (辅助函数 load_test_queries, extract_answer_and_analysis, normalize_option_answer 与 success_1/query_rag.py 相同)
def load_test_queries(test_file_path: str) -> List[Dict[str, Any]]:
    with open(test_file_path, 'r', encoding='utf-8') as f: return json.load(f)
def extract_answer_and_analysis(raw_response: str) -> Dict[str, str]:
    match = re.search(r"答案[:：]\s*([A-Da-d\s,，|、\[\]]+)", raw_response)
    extracted_answer = "".join(sorted(list(set(re.findall(r"[A-Da-d]", match.group(1).upper()))))) if match else "未识别"
    analysis_match = re.search(r"解析[:：]\s*(.*?)(?:答案[:：]|$)", raw_response, flags=re.DOTALL)
    extracted_analysis = analysis_match.group(1).strip() if analysis_match else ""
    return {"extracted_answer": extracted_answer, "extracted_analysis": extracted_analysis}
def normalize_option_answer(answer_text: str) -> str:
    return "".join(sorted(set(re.findall(r"[A-Da-d]", str(answer_text).upper()))))

# ============================================================
#  主流程 - 网格搜索超参数优化
# ============================================================
def main():
    # 超参数网格搜索配置
    retrieve_top_k_options = [3, 5, 10, 20, 50, 100]
    rerank_top_k_options = [1, 2, 3, 4, 5]
    
    config = {
        "embedding_model_path": "./models/Qwen/Qwen3-Embedding-4B",
        "rerank_model_path": "./models/Qwen/Qwen3-Reranker-4B",
        "qwen_model_path": "/data/home/aim/aim_lipengfa/aim_lipengfa/cpt_sft/lora_folder/merge_lora/qwen3_view_1_epoch/",
        "db_path": "./chroma_db_qwen3",
        "collection_name": "engineering_specs_qwen3",
        "test_query_file": "./test_new_base.json",
    }
    
    print("="*80)
    print("🚀 开始执行RAG超参数网格搜索 (Qwen3-4B)")
    print(f"Retrieve Top-K 选项: {retrieve_top_k_options}")
    print(f"Rerank Top-K 选项: {rerank_top_k_options}")
    print(f"总计组合数: {len(retrieve_top_k_options)} × {len(rerank_top_k_options)} = {len(retrieve_top_k_options) * len(rerank_top_k_options)}")
    print("="*80)
    
    # 初始化RAG系统（只初始化一次）
    rag_system = RAGQuerySystem(config)
    test_queries = load_test_queries(config['test_query_file'])
    
    # 存储所有组合的结果摘要
    grid_search_summary = []
    
    # 网格搜索循环
    combo_count = 0
    total_combos = len(retrieve_top_k_options) * len(rerank_top_k_options)
    
    for retrieve_k in retrieve_top_k_options:
        for rerank_k in rerank_top_k_options:
            combo_count += 1
            print(f"\n{'='*60}")
            print(f"🔍 组合 {combo_count}/{total_combos}: Retrieve={retrieve_k}, Rerank={rerank_k}")
            print(f"{'='*60}")
            
            # 为当前组合生成输出文件名
            results_output_file = f"./results_retrieve{retrieve_k}_rerank{rerank_k}.json"
            
            all_results, total_count, correct_count = [], 0, 0
            
            # 处理所有查询
            for i, test_item in enumerate(test_queries):
                query = test_item['prompt']
                print(f"处理查询 {i+1}/{len(test_queries)}", end="\r", flush=True)
                
                # 使用当前参数组合进行检索和重排序
                reranked_docs = rag_system.retrieve_and_rerank(query, retrieve_top_k=retrieve_k, rerank_top_k=rerank_k)
                rag_prompt = rag_system.build_rag_prompt(query, reranked_docs)
                answer = rag_system.generate_answer(rag_prompt)
                extracted = extract_answer_and_analysis(answer)
                
                # 判题
                standard_answer = normalize_option_answer(test_item.get('correct_answer', ''))
                pred_answer = normalize_option_answer(extracted["extracted_answer"]) if extracted["extracted_answer"] != "未识别" else ""
                is_correct = bool(standard_answer and pred_answer == standard_answer)

                if standard_answer:
                    total_count += 1
                    if is_correct: correct_count += 1
                
                # 保存结果
                result_data = {**test_item, 'raw_response': answer, **extracted, 'is_correct': is_correct, 
                             'rag_prompt': rag_prompt, 'retrieved_reranked_docs': reranked_docs,
                             'retrieve_top_k': retrieve_k, 'rerank_top_k': rerank_k}
                all_results.append(result_data)
            
            # 计算当前组合的准确率
            final_acc = (correct_count / total_count * 100) if total_count > 0 else 0.0
            
            # 保存当前组合的结果
            with open(results_output_file, 'w', encoding='utf-8') as f:
                json.dump(all_results, f, ensure_ascii=False, indent=2)
            
            # 记录到摘要
            summary_item = {
                "retrieve_top_k": retrieve_k,
                "rerank_top_k": rerank_k,
                "total_samples": total_count,
                "correct_samples": correct_count,
                "accuracy": round(final_acc, 2),
                "output_file": results_output_file
            }
            grid_search_summary.append(summary_item)
            
            # 打印当前组合结果
            print(f"\n📊 组合 {combo_count} 结果:")
            print(f"   参数: Retrieve={retrieve_k}, Rerank={rerank_k}")
            print(f"   样本数: {total_count}")
            print(f"   正确数: {correct_count}")
            print(f"   准确率: {final_acc:.2f}%")
            print(f"   输出文件: {results_output_file}")
    
    # 保存网格搜索摘要
    summary_file = "./grid_search_summary.json"
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(grid_search_summary, f, ensure_ascii=False, indent=2)
    
    # 找出最佳组合
    best_combo = max(grid_search_summary, key=lambda x: x['accuracy'])
    
    print("\n" + "="*80)
    print("🎉 网格搜索完成！")
    print("="*80)
    print("📈 所有组合结果:")
    for item in sorted(grid_search_summary, key=lambda x: x['accuracy'], reverse=True):
        print(f"   Retrieve={item['retrieve_top_k']:3d}, Rerank={item['rerank_top_k']:1d} → 准确率: {item['accuracy']:6.2f}% ({item['correct_samples']}/{item['total_samples']})")
    
    print(f"\n🏆 最佳组合:")
    print(f"   参数: Retrieve={best_combo['retrieve_top_k']}, Rerank={best_combo['rerank_top_k']}")
    print(f"   准确率: {best_combo['accuracy']:.2f}%")
    print(f"   输出文件: {best_combo['output_file']}")
    print(f"\n📋 详细摘要已保存到: {summary_file}")
    print("="*80)

if __name__ == "__main__":
    main()
