"""
Milvus + 智谱 Embedding 语义检索测试
流程：文本 → Zhipu embedding-3 (1024维) → 存入 Milvus → 语义搜索
"""
import os
from dotenv import load_dotenv
from pymilvus import MilvusClient, DataType
from zai import ZhipuAiClient
from zhipuai import ZhipuAI

load_dotenv()

ZhipuAI_API_KEY = os.getenv("ZHIPU_API_KEY", "")
MILVUS_HOST = "localhost"
MILVUS_PORT = "19530"
COLLECTION_NAME = "rag_knowledge"
DIM = 2048

KNOWLEDGE_BASE2 = [
    {"id": 1, "text": "Python 是一种面向对象的高级编程语言，由 Guido van Rossum 于 1991 年首次发布。"},
    {"id": 2, "text": "Milvus 是一款开源的向量数据库，用于存储、索引和检索大规模向量数据。"},
    {"id": 3, "text": "RAG 是检索增强生成（Retrieval-Augmented Generation）的缩写，结合了检索系统和大语言模型。"},
    {"id": 4, "text": "北京是中国的首都，拥有超过 2100 万人口，是全国政治、文化中心。"},
    {"id": 5, "text": "机器学习中的 Transformer 架构由 Google 在 2017 年的论文《Attention Is All You Need》中提出。"},
    {"id": 6, "text": "向量嵌入（Embedding）是将文本、图像等非结构化数据转换为高维向量的过程。"},
    {"id": 7, "text": "Docker 是一个开源的容器化平台，使开发者能够将应用及其依赖打包到可移植的容器中。"},
    {"id": 8, "text": "大语言模型（LLM）是基于 Transformer 架构的深度学习模型，通过海量文本预训练获得语言理解和生成能力。"},
    {"id": 9, "text": "中国的四大发明包括造纸术、印刷术、火药和指南针，对世界文明产生了深远影响。"},
    {"id": 10, "text": "余弦相似度是衡量两个向量方向相似程度的指标，广泛应用于语义搜索和推荐系统。"},
]

KNOWLEDGE_BASE = [
    "Python 是一种面向对象的高级编程语言，由 Guido van Rossum 于 1991 年首次发布。",
    "Milvus 是一款开源的向量数据库，用于存储、索引和检索大规模向量数据。",
    "RAG 是检索增强生成（Retrieval-Augmented Generation）的缩写，结合了检索系统和大语言模型。",
    "北京是中国的首都，拥有超过 2100 万人口，是全国政治、文化中心。",
    "机器学习中的 Transformer 架构由 Google 在 2017 年的论文《Attention Is All You Need》中提出。",
    "向量嵌入（Embedding）是将文本、图像等非结构化数据转换为高维向量的过程。",
    "Docker 是一个开源的容器化平台，使开发者能够将应用及其依赖打包到可移植的容器中。",
    "大语言模型（LLM）是基于 Transformer 架构的深度学习模型，通过海量文本预训练获得语言理解和生成能力。",
    "中国的四大发明包括造纸术、印刷术、火药和指南针，对世界文明产生了深远影响。",
    "余弦相似度是衡量两个向量方向相似程度的指标，广泛应用于语义搜索和推荐系统。",
]

def get_embedding(texts: list[str], client: ZhipuAI) -> list[list[float]]:
    """调用智谱 embedding-3 模型生成向量"""
    response = client.embeddings.create(
        model="embedding-3",
        input=texts,
    )
    return [item.embedding for item in response.data]
def main2():
    print("=" * 60)
    print("  Milvus + 智谱 Embedding 语义检索测试")
    print(f"  Milvus: {MILVUS_HOST}:{MILVUS_PORT}")
    print(f"  模型: Zhipu embedding-3 ({DIM}维)")
    print("=" * 60)

    if not ZhipuAI_API_KEY or ZhipuAI_API_KEY == "your_zhipu_api_key_here":
        print("\n⚠️  请先在 .env 中配置 ZHIPU_API_KEY")
        return

    zhipu_client = ZhipuAI(api_key=ZhipuAI_API_KEY)

    milvus_client = None

    try:
        # ── 1. 连接 Milvus ──
        print("\n[1/6] 连接 Milvus ...")
        milvus_client = MilvusClient(uri=f"http://{MILVUS_HOST}:{MILVUS_PORT}")
        print("  ✅ 连接成功")

        # ── 2. 生成向量 ──
        print(f"\n[2/6] 调用智谱 embedding-3 为 {len(KNOWLEDGE_BASE)} 条文本生成向量 ...")
        texts = [item for item in KNOWLEDGE_BASE]
        embeddings = get_embedding(texts, zhipu_client)
        for item in embeddings:
            print(item[:5])
        print(f"  ✅ 向量生成完成，维度: {len(embeddings[0])}")

        # ── 3. 创建集合 ──
        print(f"\n[3/6] 创建集合 '{COLLECTION_NAME}' ...")
        if milvus_client.has_collection(COLLECTION_NAME):
            print("  ⚠️  集合已存在，先删除 ...")
            milvus_client.drop_collection(COLLECTION_NAME)

        schema = milvus_client.create_schema(
            auto_id=False,
            description="RAG 知识库 - 智谱 embedding-3",
        )
        schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True)
        schema.add_field(field_name="text", datatype=DataType.VARCHAR, max_length=512)
        schema.add_field(field_name="embedding", datatype=DataType.FLOAT_VECTOR, dim=DIM)

        index_params = milvus_client.prepare_index_params()
        index_params.add_index(
            field_name="embedding",
            index_type="IVF_FLAT",
            metric_type="COSINE",
            params={"nlist": 128},
        )

        milvus_client.create_collection(
            collection_name=COLLECTION_NAME,
            schema=schema,
            index_params=index_params,
        )
        print("  ✅ 集合 + 索引创建完成 (IVF_FLAT + COSINE)")

        # ── 4. 插入数据 ──
        print(f"\n[4/6] 插入 {len(KNOWLEDGE_BASE)} 条数据到 Milvus ...")
        data = [
            {
                "id": i,
                "text": item[i],
                "embedding": embeddings[i],
            }
            for i, item in enumerate(KNOWLEDGE_BASE)
        ]
        res = milvus_client.upsert(collection_name=COLLECTION_NAME, data=data)
        print(f"  ✅ 插入成功，共 {res['upsert_count']} 条")

        # ── 6. 交互式查询 ──
        print(f"\n[6/6] 进入交互模式（输入 quit 退出）...")
        print("     输入你的问题，系统用语义搜索匹配最相关的知识条目\n")
        while True:
            try:
                user_query = input("  你: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n  👋 退出交互模式")
                break
            if not user_query or user_query.lower() == "quit":
                print("  👋 退出交互模式")
                break

            query_vec = get_embedding([user_query], zhipu_client)[0]
            results = milvus_client.search(
                collection_name=COLLECTION_NAME,
                data=[query_vec],
                anns_field="embedding",
                search_params={"metric_type": "COSINE", "params": {"nprobe": 10}},
                limit=3,
                output_fields=["text"],
            )

            print(f"\n  🤖 最相关的 {len(results[0])} 条结果:")
            for j, hit in enumerate(results[0], 1):
                score = hit["distance"]
                print(f"     [{j}] (相似度={score:.4f})")
                print(f"         {hit['entity']['text']}")
            print()

        print("\n" + "=" * 60)
        print("  🎉 测试完成！")
        print("=" * 60)

    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if milvus_client:
            milvus_client.close()
            print("  ✅ Milvus 连接已关闭")


def main():
    print("=" * 60)
    print("  Milvus + 智谱 Embedding 语义检索测试")
    print(f"  Milvus: {MILVUS_HOST}:{MILVUS_PORT}")
    print(f"  模型: Zhipu embedding-3 ({DIM}维)")
    print("=" * 60)

    if not ZhipuAI_API_KEY or ZhipuAI_API_KEY == "your_zhipu_api_key_here":
        print("\n⚠️  请先在 .env 中配置 ZHIPU_API_KEY")
        return

    zhipu_client = ZhipuAI(api_key=ZhipuAI_API_KEY)
    response = zhipu_client.embeddings.create(
        model="embedding-3",  # 填写需要调用的模型编码
        input=[
            "美食非常美味，服务员也很友好。",
            "这部电影既刺激又令人兴奋。",
            "阅读书籍是扩展知识的好方法。"
        ],
    )
    print(response)

    milvus_client = None

    try:
        # ── 1. 连接 Milvus ──
        print("\n[1/6] 连接 Milvus ...")
        milvus_client = MilvusClient(uri=f"http://{MILVUS_HOST}:{MILVUS_PORT}")
        print("  ✅ 连接成功")

        # ── 2. 生成向量 ──
        print(f"\n[2/6] 调用智谱 embedding-3 为 {len(KNOWLEDGE_BASE)} 条文本生成向量 ...")
        texts = [str(item["text"]) for item in KNOWLEDGE_BASE]
        embeddings = get_embedding(texts, zhipu_client)
        print(f"  ✅ 向量生成完成，维度: {len(embeddings[0])}")

        # ── 3. 创建集合 ──
        print(f"\n[3/6] 创建集合 '{COLLECTION_NAME}' ...")
        if milvus_client.has_collection(COLLECTION_NAME):
            print("  ⚠️  集合已存在，先删除 ...")
            milvus_client.drop_collection(COLLECTION_NAME)

        schema = milvus_client.create_schema(
            auto_id=False,
            description="RAG 知识库 - 智谱 embedding-3",
        )
        schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True)
        schema.add_field(field_name="text", datatype=DataType.VARCHAR, max_length=512)
        schema.add_field(field_name="embedding", datatype=DataType.FLOAT_VECTOR, dim=DIM)

        index_params = milvus_client.prepare_index_params()
        index_params.add_index(
            field_name="embedding",
            index_type="IVF_FLAT",
            metric_type="COSINE",
            params={"nlist": 128},
        )

        milvus_client.create_collection(
            collection_name=COLLECTION_NAME,
            schema=schema,
            index_params=index_params,
        )
        print("  ✅ 集合 + 索引创建完成 (IVF_FLAT + COSINE)")

        # ── 4. 插入数据 ──
        print(f"\n[4/6] 插入 {len(KNOWLEDGE_BASE)} 条数据到 Milvus ...")
        data = [
            {
                "id": item["id"],
                "text": item["text"],
                "embedding": embeddings[i],
            }
            for i, item in enumerate(KNOWLEDGE_BASE)
        ]
        res = milvus_client.insert(collection_name=COLLECTION_NAME, data=data)
        print(f"  ✅ 插入成功，共 {res['insert_count']} 条")

        # ── 5. 语义搜索 ──
        print(f"\n[5/6] 语义搜索测试 ...")
        queries = [
            "什么是向量数据库？",
            "Python 是谁发明的？",
            "embedding 是什么？",
        ]

        for query in queries:
            print(f"\n  🔍 查询: \"{query}\"")
            query_vec = get_embedding([query], zhipu_client)[0]

            results = milvus_client.search(
                collection_name=COLLECTION_NAME,
                data=[query_vec],
                anns_field="embedding",
                search_params={"metric_type": "COSINE", "params": {"nprobe": 10}},
                limit=3,
                output_fields=["text"],
            )

            for j, hit in enumerate(results[0], 1):
                score = hit["distance"]
                text_preview = hit["entity"]["text"][:60]
                print(f"     [{j}] 相似度={score:.4f}  \"{text_preview}...\"")

        # ── 6. 交互式查询 ──
        print(f"\n[6/6] 进入交互模式（输入 quit 退出）...")
        print("     输入你的问题，系统用语义搜索匹配最相关的知识条目\n")
        while True:
            try:
                user_query = input("  你: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n  👋 退出交互模式")
                break
            if not user_query or user_query.lower() == "quit":
                print("  👋 退出交互模式")
                break

            query_vec = get_embedding([user_query], zhipu_client)[0]
            results = milvus_client.search(
                collection_name=COLLECTION_NAME,
                data=[query_vec],
                anns_field="embedding",
                search_params={"metric_type": "COSINE", "params": {"nprobe": 10}},
                limit=3,
                output_fields=["text"],
            )

            print(f"\n  🤖 最相关的 {len(results[0])} 条结果:")
            for j, hit in enumerate(results[0], 1):
                score = hit["distance"]
                print(f"     [{j}] (相似度={score:.4f})")
                print(f"         {hit['entity']['text']}")
            print()

        print("\n" + "=" * 60)
        print("  🎉 测试完成！")
        print("=" * 60)

    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if milvus_client:
            print("\n🧹 清理测试数据 ...")
            try:
                if milvus_client.has_collection(COLLECTION_NAME):
                    milvus_client.drop_collection(COLLECTION_NAME)
                    print(f"  ✅ 集合 '{COLLECTION_NAME}' 已删除")
            except Exception as e:
                print(f"  ⚠️  清理时出错: {e}")
            milvus_client.close()
            print("  ✅ Milvus 连接已关闭")


if __name__ == "__main__":
    main2()
