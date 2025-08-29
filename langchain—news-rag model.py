from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv, find_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.prompts import ChatPromptTemplate
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_openai import ChatOpenAI  
import os
import pickle

# 加载环境变量
load_dotenv(find_dotenv())

## 初始化Dashscope Embeddings
embeddings = DashScopeEmbeddings(
    model="text-embedding-v3",
    dashscope_api_key="sk-11d21951a0ad4c6599138de1cb50ac24",
)
llm = ChatOpenAI(
    model="qwen2.5-32b-instruct",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    api_key="sk-11d21951a0ad4c6599138de1cb50ac24"
)

# ####下面是离线部分####
# try:
#     # 检查数据集文件是否存在
#     if not os.path.exists('homework/新闻数据集.txt'):
#         raise FileNotFoundError("数据集文件 'homework/新闻数据集.txt' 不存在")
    
#     print("开始加载文档...")
#     loader = TextLoader('homework/新闻数据集.txt', encoding='utf-8')
#     documents = loader.load()
#     print(f"文档加载完成，共加载 {len(documents)} 个文档")
    
#     print("开始切分文档...")
#     text_splitter = RecursiveCharacterTextSplitter.from_language(
#         language="markdown",
#         chunk_size=200,
#         chunk_overlap=20
#     )
#     texts = text_splitter.split_documents(documents)
#     print(f"文档切分完成，共生成 {len(texts)} 个文本片段")
    
#     print("开始构建向量数据库...")
#     db = FAISS.from_documents(texts, embeddings)
#     print("向量数据库构建完成")
    
#     print("正在保存向量数据库...")
#     with open("homework/my_db", "wb") as f:
#         pickle.dump(db, f)
#     print("向量数据库已保存至 'homework/my_db'")
    
# except Exception as e:
#     print(f"处理过程中发生错误: {str(e)}")
#     import traceback
#     traceback.print_exc()


####下面是在线部分####
#load数据库
with open("my_db", "rb") as f:
    db = pickle.load(f)

#获取检索器，k为相关系数，选择前两个最相关的
retriever = db.as_retriever(search_kwarg={"k": 2})

#创建问题回答模板(带有system消息)
# {context}为检索信息，{question}为用户问题
prompt_template = ChatPromptTemplate.from_messages([
    ("system","""你是一个对接问题排查机器人。
     你的任务是根据下述给定的已知信息回答用户问题。
     确保你的回复完全基于下述已知信息，不要编造答案。
     请用中文回答。
     已知信息：
     {context}
     """

    ),("user","{question}")
])
chain_type_kwargs = {
"prompt": prompt_template,
}
#构建Retrieva1QA链
#使用stuff类型的链，这意味所有检索到的上下文会被组合为一个输入，然后传给大模型llm
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type = "stuff",
    chain_type_kwargs = chain_type_kwargs,
    retriever = retriever,
    return_source_documents = True
)
response = qa_chain.invoke({"query":"什么是中药零食"
""})

print(response["result"])
print("-"*100)
#输出回答来源
print (response["source_documents"])