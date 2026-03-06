import json
import os
import re
import torch
import torch.nn.functional as F
import chromadb
from transformers import AutoTokenizer, AutoModel
from typing import List, Dict, Any


# ============================================================
#  Qwen3-Embedding 封装类
#  Qwen3-Embedding 基于 Decoder-Only 架构，使用 last-token 池化，
#  查询时需附加任务指令前缀，文档编码时不加前缀。
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
        """使用最后一个有效 token 的隐状态作为句向量（left-padding 情形下直接取最后一列）。"""
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
        """
        对文本列表进行编码，返回 numpy 数组。

        Args:
            texts: 单个字符串或字符串列表。
            normalize_embeddings: 是否对输出做 L2 归一化。
            is_query: 若为 True，则在每条文本前附加任务指令（查询时使用）。
            batch_size: 批处理大小，4B 模型建议设为 4~8。
            task: 任务描述，用于查询端的指令前缀。
        """
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
        # 单条输入时返回 1D 数组，与 SentenceTransformer 行为保持一致
        if result.shape[0] == 1:
            return result[0].numpy()
        return result.numpy()


# ============================================================
#  文档加载与分块
# ============================================================
def load_and_chunk_documents(json_files_dir: str) -> List[Dict[str, Any]]:
    """
    从 JSON 文件加载文档，并按 section 的 id 进行分块。
    （逻辑与 success_1/build_database.py 完全一致）
    """
    documents = []
    id_counts: Dict[str, int] = {}
    print(f"从目录 '{json_files_dir}' 加载 JSON 文件并按 section 分块...")

    for filename in sorted(os.listdir(json_files_dir)):
        if not filename.endswith(".json"):
            continue

        match = re.match(r"^(\d+)", filename)
        if not match:
            print(f"警告: 文件名 '{filename}' 不以数字开头，已跳过。")
            continue
        file_id_prefix = match.group(1)

        file_path = os.path.join(json_files_dir, filename)
        print(f"处理文件: {filename}")

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            title = data.get("title", "")
            sections = data.get("sections", [])

            for section in sections:
                section_id = section.get("id", "").strip()
                content = section.get("content", "").strip()

                if not section_id or not content:
                    continue

                base_chunk_id = f"{file_id_prefix}.{section_id}"
                count = id_counts.get(base_chunk_id, 0)
                new_chunk_id = base_chunk_id if count == 0 else f"{base_chunk_id}.{count}"
                id_counts[base_chunk_id] = count + 1

                documents.append(
                    {
                        "content": content,
                        "chunk_id": new_chunk_id,
                        "metadata": {
                            "source_file": filename,
                            "title": title,
                            "original_section_id": section_id,
                        },
                    }
                )

        except Exception as e:
            print(f"处理文件 {filename} 时出错: {e}")
            continue

    print(f"总共加载并创建了 {len(documents)} 个文档块。")
    return documents


# ============================================================
#  构建向量数据库
# ============================================================
def build_vector_database(
    documents: List[Dict[str, Any]],
    embedding_model_path: str,
    db_path: str,
    collection_name: str,
):
    """使用 Qwen3-Embedding 构建并保存向量数据库。"""
    print("\n开始构建向量数据库...")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 加载 Qwen3-Embedding
    embedder = Qwen3Embedder(embedding_model_path, device=device)

    # 初始化 ChromaDB
    print(f"初始化并保存数据库到 '{db_path}'...")
    db_client = chromadb.PersistentClient(path=db_path)
    collection = db_client.get_or_create_collection(name=collection_name)

    # 批量编码并写入 —— 文档端不加任务前缀（is_query=False）
    batch_size = 8  # Qwen3-Embedding-4B 显存占用较大，建议 batch_size 设小
    for i in range(0, len(documents), batch_size):
        batch = documents[i : i + batch_size]

        contents = [doc["content"] for doc in batch]
        chunk_ids = [doc["chunk_id"] for doc in batch]
        metadatas = [doc["metadata"] for doc in batch]

        embeddings = embedder.encode(
            contents, normalize_embeddings=True, is_query=False
        ).tolist()

        collection.add(
            ids=chunk_ids,
            embeddings=embeddings,
            documents=contents,
            metadatas=metadatas,
        )
        print(f"已处理 {min(i + batch_size, len(documents))}/{len(documents)} 个文档块")

    print("\n向量数据库构建完成！")
    print(f"数据库已保存至: {db_path}")
    print(f"集合名称: {collection_name}")
    print(f"集合中文档总数: {collection.count()}")

    # 打印示例
    print("\n" + "-" * 30)
    print("📋 数据库内容示例 (前3条):")
    results = collection.get(limit=3, include=["documents", "metadatas"])
    for idx in range(len(results["ids"])):
        print(f"\n[示例 {idx+1}]")
        print(f"ID: {results['ids'][idx]}")
        print(f"元数据: {results['metadatas'][idx]}")
        preview = results["documents"][idx]
        print(f"内容预览: {preview[:100]}{'...' if len(preview) > 100 else ''}")
    print("-" * 30)


# ============================================================
#  主流程
# ============================================================
def main():
    DATA_DIR = "./data1"
    EMBEDDING_MODEL_PATH = "./models/Qwen/Qwen3-Embedding-4B"
    DB_SAVE_PATH = "./chroma_db_qwen3"          # 使用独立目录，避免与旧库冲突
    COLLECTION_NAME = "engineering_specs_qwen3"

    print("=" * 60)
    print("🚀 开始执行数据库构建脚本 (Qwen3-Embedding-4B)")
    print("=" * 60)

    docs_to_embed = load_and_chunk_documents(DATA_DIR)

    if not docs_to_embed:
        print("错误: 没有加载到任何文档块，程序终止。")
        return

    build_vector_database(
        documents=docs_to_embed,
        embedding_model_path=EMBEDDING_MODEL_PATH,
        db_path=DB_SAVE_PATH,
        collection_name=COLLECTION_NAME,
    )

    print("\n" + "=" * 60)
    print("✅ 数据库构建任务全部完成。")
    print("=" * 60)


if __name__ == "__main__":
    main()
