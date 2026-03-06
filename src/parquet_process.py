################
from omegaconf import OmegaConf
import re
import pyspark.sql.functions as F
from pyspark.sql.types import ArrayType, StringType

from dataverse.etl import register_etl
@register_etl
def custom___text___my_remove_abs_ref_process(spark, data, subset="Full Text", *args, **kwargs):
    """
    支持Spark DataFrame的ETL函数，对指定列（如'Full Text'）批量删除摘要前和参考文献后的内容。
    自动兼容 RDD 或 DataFrame 输入。
    """
    from pyspark.sql import DataFrame

    # 如果是 RDD，自动转为 DataFrame（假设 RDD 元素是 dict 或 Row）
    if not isinstance(data, DataFrame):
        # 自动推断 schema
        data = spark.createDataFrame(data)
    df = data

    def process_text(text):
        if not text:
            return text
        abstract_headers = [
            r"^Abstract",
            r"^ABSTRACT",
        ]
        ab_pattern = re.compile("|".join(abstract_headers), re.MULTILINE)
        ab_matches = list(ab_pattern.finditer(text))
        start_pos = 0
        if ab_matches:
            first_ab_match = min(ab_matches, key=lambda m: m.start())
            start_pos = first_ab_match.start()
        conclusion_headers = [
            r"^Conclusions\b",
            r"^Summary\b"
        ]
        con_pattern = re.compile("|".join(conclusion_headers), re.MULTILINE | re.IGNORECASE)
        con_matches = list(con_pattern.finditer(text))
        text_len = len(text)
        start_point = 0
        if con_matches:
            last_con_match = max(con_matches, key=lambda m: m.start())
            if last_con_match.start() < text_len * 0.6:
                start_point = 0
            else:
                start_point = last_con_match.end()
        reference_headers = [
            r"^References?\b",
            r"^Bibliography\b",
            r"^Acknowledgements?\b",
            r"^Data availability\b",
            r"^Supplemental material\b"
        ]
        pattern = re.compile("|".join(reference_headers), re.MULTILINE | re.IGNORECASE)
        matches = list(pattern.finditer(text))
        length = len(text) * 0.67
        double_matches = [m for m in matches if m.start() > start_point and m.start() > length]
        if double_matches:
            first_match = min(double_matches, key=lambda m: m.start())
            return text[start_pos:first_match.start()]
        else:
            return text[start_pos:]
    process_udf = F.udf(process_text, returnType=F.StringType())
    return df.withColumn(subset, process_udf(F.col(subset)))

@register_etl
def custom___text___split_by_sentence_block(spark, data, subset="Full Text", min_words=10, *args, **kwargs):
    """
    支持Spark DataFrame的ETL函数，对指定列（如'Full Text'）按结束性标点分块，每块至少min_words个单词。
    分块后每块一行，其他字段保持原值。
    """
    from pyspark.sql import DataFrame

    if not isinstance(data, DataFrame):
        data = spark.createDataFrame(data)
    df = data

    import re
    def split_text(text, min_words=min_words):
        if not text:
            return []
        # 保护网址
        text = re.sub(r'(http[s]?://[^\s]+)', lambda m: m.group(1).replace('.', '[DOT]'), text)
        text = re.sub(r'(doi\.org/[^\s]+)', lambda m: m.group(1).replace('.', '[DOT]'), text)
        # 保护邮箱
        text = re.sub(r'([A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,})', lambda m: m.group(1).replace('.', '[DOT]'), text)
        # 保护小数
        text = re.sub(r'(\d)\.(\d)', r'\1[DOT]\2', text)
        # 保护常见缩写
        abbrs = [
            'e.g', 'i.e', 'S.A', 'U.S.A', 'Dr', 'Mr', 'Ms', 'Prof', 'Inc', 'Ltd', 'Jr', 'Sr', 'vs', 'Fig', 'No', 'Vol', 'pp', 'et al', 'etc'
        ]
        for abbr in abbrs:
            text = re.sub(rf'\b{abbr}\.', f'{abbr}[DOT]', text, flags=re.IGNORECASE)
        # 保护引用编号
        text = re.sub(r'(\[\d+\]\.|\(\d+\)\.|^\d+\.)', lambda m: m.group(0).replace('.', '[DOT]'), text)
        # 分句：句号/问号/感叹号/省略号后跟空格和大写字母才分句
        sentences = re.split(r'(?<=[.!?…])\s+(?=[A-Z])', text)
        # 恢复被保护的点
        sentences = [s.replace('[DOT]', '.') for s in sentences if s.strip()]
        blocks = []
        block = []
        word_count = 0
        for sentence in sentences:
            words = sentence.split()
            block.append(sentence)
            word_count += len(words)
            if word_count >= min_words:
                candidate = ' '.join(block).strip()
                while candidate and not re.match(r'^[A-Za-z]', candidate):
                    block = block[1:]
                    candidate = ' '.join(block).strip()
                if candidate:
                    blocks.append(candidate)
                block = []
                word_count = 0
        if block:
            candidate = ' '.join(block).strip()
            while candidate and not re.match(r'^[A-Za-z]', candidate):
                block = block[1:]
                candidate = ' '.join(block).strip()
            if candidate:
                blocks.append(candidate)
        return blocks

    split_udf = F.udf(split_text, ArrayType(StringType()))
    # 新增分块列
    df = df.withColumn("blocks", split_udf(F.col(subset)))
    # 展开分块，每块一行，其他字段自动复制
    df = df.withColumn(subset, F.explode("blocks")).drop("blocks")
    return df

@register_etl
def custom___text___filter_en(spark, data, subset="Full Text", model_path="f:\\GeoGPT\\filter\\software\\lid.176.bin", threshold=0.65, *args, **kwargs):
    """
    支持Spark DataFrame的ETL函数，对指定列（如'Full Text'）批量提取英文句子。
    自动兼容 RDD 或 DataFrame 输入。
    """
    from pyspark.sql import DataFrame

    if not isinstance(data, DataFrame):
        data = spark.createDataFrame(data)
    df = data

    def filter_en(text):
        import fasttext
        # 模型对象缓存到全局变量，避免每行都加载
        global _fasttext_model
        if '_fasttext_model' not in globals():
            _fasttext_model = fasttext.load_model(model_path)
        model = _fasttext_model

        if not text:
            return []
        sentences = [s.strip() for s in text.split('.') if s.strip()]
        english_sentences = []
        for sentence in sentences:
            label, prob = model.predict(sentence)
            if label[0] == '__label__en' and prob[0] >= threshold:
                english_sentences.append(sentence)
        return english_sentences

    from pyspark.sql.types import ArrayType, StringType
    filter_en_udf = F.udf(filter_en, ArrayType(StringType()))
    return df.withColumn(subset + "_en", filter_en_udf(F.col(subset)))

ETL_config = OmegaConf.create({

    # Set up Spark
    'spark': {
        'appname': 'ETL',
        'driver': {'memory': '32g'},
    },
    'etl': [
        {
          ## 输入数据
          'name': 'data_ingestion___parquet___pq2raw',
          'args': {'path': ['/path/to/data/merage.parquet']}
        },
       {
         # Transform; deduplicate data via minhash
         'name': 'deduplication___minhash___lsh_jaccard', 
         'args': {'threshold': 0.75,
                 'ngram_size': 5,
                 'subset': 'text'}
       },
       {
         ## 数据转换
         # 第一步清洗（去除重音符号）
         'name': 'cleaning___char___remove_accent',
         'args': {'subset': 'Full Text'},
       },
       {
         # 第二步清洗（规范化空格）
         'name': 'cleaning___char___normalize_whitespace',
         'args': {'subset': 'Full Text'},
       },
       {
         # 第三步清洗（删除所有不可打印的字符）
         'name': 'cleaning___char___remove_unprintable',
         'args': {'subset': 'Full Text'},
       },
    #    {
    #     #   第四步清洗（只保留英语）
    #      'name': 'custom___text___filter_en',
    #      'args': {'subset': 'Full Text', 
    #               'model_path': "/path/to/software/lid.176.bin",
    #               'threshold': 0.65},
    #    },
       {
         # 第四步清洗（删除摘要和参考文献）
         'name': 'custom___text___my_remove_abs_ref_process',
         'args': {'subset': 'Full Text'}
       },  
        # 分块需慎重最后再分块
        {
          # 第五步清洗（分块及删除不以标点符号结尾的句子）
          'name': 'cleaning___document___split_by_word',
          'args': {
              'subset': 'Full Text',
              'word_per_chunk': 300,                  
                   }
        },

        {
          # Load; Save the data
          'name': 'data_save___parquet___ufl2parquet',
          'args': {'save_path': '/path/to/data/merage.parquet'}
        }
      ]
  })


from dataverse.etl import ETLPipeline

etl_pipeline = ETLPipeline()

print('XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX')
spark, dataset = etl_pipeline.run(config=ETL_config, verbose=True)


