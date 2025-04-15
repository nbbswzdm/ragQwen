import os
from dotenv import load_dotenv
from typing import List, Dict, Tuple, Any
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings.dashscope import DashScopeEmbeddings
from langchain_community.chat_models.tongyi import ChatTongyi
from langchain_core.documents import Document
from langchain_core.runnables import RunnablePassthrough
from sentence_transformers.util import cos_sim
from langsmith import Client
from langchain_core.tracers.context import tracing_v2_enabled
from langchain.callbacks.tracers import LangChainTracer

# 加载环境变量
load_dotenv()

# 配置LangSmith
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_PROJECT"] = "Qwen-RAG-FAISS"  # 更新项目名称

class QwenRAGSystem:
    def __init__(self):
        # 初始化LangSmith客户端
        self.client = Client()
        
        # 初始化模型和组件
        self.llm = ChatTongyi(
            model_name="qwen-plus",
            temperature=0.7,
            dashscope_api_key=os.getenv("DASHSCOPE_API_KEY")
        )
        self.embedding = DashScopeEmbeddings(
            model="text-embedding-v2",
            dashscope_api_key=os.getenv("DASHSCOPE_API_KEY")
        )
        
        # 初始化FAISS向量库
        self.vectorstore_path = "faiss_vectorstore"
        self._initialize_vectorstore()
        
        self.history = []
        
        # 定义提示模板
        self.qa_prompt = ChatPromptTemplate.from_messages([
            ("system", """你是一个专业的问答助手，请根据以下信息回答问题。
            
检索内容:
{context}

历史问答:
{history}

用户提问:
{question}"""),
        ])
        
        self.query_expansion_prompt = ChatPromptTemplate.from_messages([
            ("system", """基于以下用户问题和历史对话，生成3个相关的完整独立回答的扩展问题。
历史对话:
{history}

用户问题:
{question}

请直接输出3个相关问题，每行一个，不要编号。"""),
        ])
        
        # 创建LangChain追踪器
        self.tracer = LangChainTracer()
    
    def _initialize_vectorstore(self):
        """初始化FAISS向量库"""
        try:
            # 尝试加载已有向量库
            self.vectorstore = FAISS.load_local(
                self.vectorstore_path, 
                self.embedding,
                allow_dangerous_deserialization=True
            )
            print("成功加载已有FAISS向量库")
        except:
            # 创建新向量库
            self.vectorstore = FAISS.from_texts(
                ["默认初始文本"], 
                embedding=self.embedding
            )
            self.vectorstore.save_local(self.vectorstore_path)
            print("创建新的FAISS向量库")
    
    def expand_queries(self, question: str) -> List[str]:
        """生成扩展问题（带追踪）"""
        chain = (
            {"question": RunnablePassthrough(), 
             "history": lambda x: "\n".join([f"Q: {q}\nA: {a}" for q, a in self.history])}
            | self.query_expansion_prompt
            | self.llm
            | StrOutputParser()
        )
        
        with tracing_v2_enabled():
            expanded = chain.invoke(question, config={"callbacks": [self.tracer]})
            return [question] + [q.strip() for q in expanded.split("\n") if q.strip()][:3]
    
    def compute_similarity(self, query: str, queries: List[str]) -> List[float]:
        """计算余弦相似度（带追踪）"""
        with tracing_v2_enabled():
            query_embed = self.embedding.embed_query(query)
            query_embeds = self.embedding.embed_documents(queries)
            similarities = cos_sim(query_embed, query_embeds)[0].tolist()
            return similarities
    
    def retrieve_documents(self, queries: List[str], similarities: List[float]) -> List[Tuple[str, float]]:
        """检索文档并计算综合得分（带追踪）"""
        doc_scores = {}
        
        with tracing_v2_enabled():
            for query, sim in zip(queries, similarities):
                docs = self.vectorstore.similarity_search_with_score(
                    query, 
                    k=3,
                )
                for doc, doc_score in docs:
                    if doc.page_content not in doc_scores:
                        doc_scores[doc.page_content] = 0
                    doc_scores[doc.page_content] += sim * (1 - doc_score)  # FAISS返回的是距离，需要转换为相似度
        
        # 按得分排序并取前三
        sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_docs[:3]
    
    def generate_response(self, question: str, docs: List[Tuple[str, float]]) -> str:
        """生成最终回答（带追踪）"""
        context = "\n".join([f"[相关度: {score:.2f}] {doc}" 
                            for doc, score in docs])
        history = "\n".join([f"Q: {q}\nA: {a}" for q, a in self.history[-3:]])  # 只保留最近3轮
        
        chain = (
            {"context": lambda x: context, 
             "history": lambda x: history,
             "question": RunnablePassthrough()}
            | self.qa_prompt
            | self.llm
            | StrOutputParser()
        )
        
        with tracing_v2_enabled():
            return chain.invoke(question, config={"callbacks": [self.tracer]})
    
    def add_to_knowledge_base(self, documents: List[str]):
        """向知识库添加文档（带追踪）"""
        docs = [Document(page_content=text) for text in documents]
        with tracing_v2_enabled():
            self.vectorstore.add_documents(docs)
            self.vectorstore.save_local(self.vectorstore_path)  # 保存变更
    
    def log_feedback(self, question: str, answer: str, feedback: Dict):
        """记录用户反馈到LangSmith"""
        runs = self.client.list_runs(
            project_name=os.environ["LANGCHAIN_PROJECT"],
            execution_order=1,
            error=False,
        )
        
        # 找到最近的对应运行记录
        for run in runs:
            if run.inputs.get("question") == question and run.outputs.get("output") == answer:
                self.client.create_feedback(
                    run.id,
                    key="user_rating",
                    score=feedback.get("rating", 0),
                    comment=feedback.get("comment", ""),
                )
                break
    
    def __call__(self, question: str) -> str:
        """处理用户问题（整个流程追踪）"""
        with tracing_v2_enabled():
            # 1. 生成扩展问题
            expanded_queries = self.expand_queries(question)
            print(f"[DEBUG] 扩展问题: {expanded_queries}")
            
            # 2. 计算相似度
            similarities = self.compute_similarity(question, expanded_queries)
            print(f"[DEBUG] 相似度: {list(zip(expanded_queries, similarities))}")
            
            # 3. 检索文档
            retrieved_docs = self.retrieve_documents(expanded_queries, similarities)
            print(f"[DEBUG] 检索到 {len(retrieved_docs)} 个文档")
            
            # 4. 生成回答
            answer = self.generate_response(question, retrieved_docs)
            
            # 5. 更新历史
            self.history.append((question, answer))
            
            return answer

# 示例用法
if __name__ == "__main__":
    # 初始化系统
    rag = QwenRAGSystem()
    
    # 向知识库添加测试数据
    test_knowledge = [
        "阿里千问(Qwen)是阿里巴巴开发的大语言模型，支持多种自然语言处理任务",
        "Qwen模型的特点包括强大的中文理解能力、长文本处理和支持多种插件",
        "使用LangChain构建RAG应用通常需要向量数据库、检索器和生成模型",
        "FAISS是Meta开发的向量相似度搜索库，适合大规模向量检索",
        "RAG(检索增强生成)技术通过结合检索和生成提高回答准确性"
    ]
    rag.add_to_knowledge_base(test_knowledge)
    
    # 测试问题
    test_questions = [
        "介绍一下阿里千问",
        "它有什么特点?",
        "如何用LangChain实现RAG?"
    ]
    
    for q in test_questions:
        print(f"\n用户提问: {q}")
        answer = rag(q)
        print(f"系统回答: {answer}")
        
        # 模拟用户反馈
        feedback = {
            "rating": 5 if "不知道" not in answer else 1,
            "comment": "回答准确" if "不知道" not in answer else "回答不完整"
        }
        rag.log_feedback(q, answer, feedback)
    
    # 保存向量库
    rag.vectorstore.save_local(rag.vectorstore_path)