# backend_rag.py - 智能客服 RAG 版
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com' #通过代码设置环境变量，配置国内镜像站 (hf-mirror.com)
import uuid
import json
import logging
from typing import List, Dict, Optional
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from openai import AsyncOpenAI
import redis.asyncio as redis
from urllib.parse import urlparse

# ---- RAG 相关导入 ----
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings   # 免费本地Embedding
from langchain_community.vectorstores import Chroma
from langchain_classic.chains import RetrievalQA

# 也可以改为 OpenAI / Cohere 的 Embedding（需要额外配置API Key）
# from langchain_openai import OpenAIEmbeddings

# ------------------------------
# 1. 日志 & 环境变量
# ------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv('deepseek.env')
api_key = os.getenv("DEEPSEEK_API_KEY")
if not api_key:
    raise RuntimeError("请设置 DEEPSEEK_API_KEY")

# ------------------------------
# 2. Redis 配置（会话管理，与之前相同）
# ------------------------------
#REDIS_URL = os.getenv("REDIS_URL")
#REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
#REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
#REDIS_DB = int(os.getenv("REDIS_DB", 0))
#SESSION_TTL = int(os.getenv("SESSION_TTL", 1800))

REDIS_URL = os.getenv("REDIS_URL")
REDIS_DB = int(os.getenv("REDIS_DB", 0))
SESSION_TTL = int(os.getenv("SESSION_TTL", 1800)) # 30分钟过期

if REDIS_URL:
    parsed = urlparse(REDIS_URL)
    REDIS_HOST = parsed.hostname
    REDIS_PORT = parsed.port
    REDIS_PASSWORD = parsed.password
else:
    # 本地开发 fallback
    REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
    REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
    REDIS_PASSWORD = os.getenv("REDIS_PASSWORD", None)

# 电商客服 System Prompt（精简版，只保留角色和规则，具体知识由RAG提供）
SYSTEM_PROMPT = """
你是一个专业、热情、高效的电商客服，名叫“小智”。
你的工作是为用户提供商品咨询、订单查询、物流跟踪、售后处理等服务。

【重要规则】
1. 只回答与购物相关的问题。
2. 回答简洁清晰，每句话不超过30字，使用敬语。
3. 当你被问到产品信息或售后问题时，请优先使用下面【参考知识】中的内容回答。
4. 如果知识库中没有相关信息，请礼貌告知用户并建议联系人工客服。
5. 每条回复结尾加上“😊”或“🌟”。
"""

# ------------------------------
# 3. 向量数据库与 RAG 初始化（全局单例）
# ------------------------------
vectorstore = None
retriever = None

def init_rag():
    """加载本地知识库，构建 Chroma 向量存储"""
    global vectorstore, retriever
    logger.info("开始加载知识库...")

    # 3.1 加载文档（支持多个 .txt）
    docs_dir = "./knowledge_base"
    if not os.path.exists(docs_dir):
        os.makedirs(docs_dir)
        logger.warning(f"{docs_dir} 不存在，已创建空目录。请放入您的产品FAQ文本文件。")
        return

    loader = DirectoryLoader(docs_dir, glob="**/*.txt", loader_cls=TextLoader, loader_kwargs={'encoding': 'utf-8'})
    documents = loader.load()
    if not documents:
        logger.warning("知识库中没有找到任何 .txt 文件")
        return

    # 3.2 文本分块
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_documents(documents)
    logger.info(f"文档已分割为 {len(chunks)} 个文本块")

    # 3.3 使用免费本地 Embedding 模型（可替换为 OpenAI/Cohere）
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # 3.4 存入 Chroma（持久化到本地目录）
    persist_dir = "./chroma_db"
    vectorstore = Chroma.from_documents(documents=chunks, embedding=embeddings, persist_directory=persist_dir)
    vectorstore.persist()
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    logger.info("知识库向量化完成，RAG 检索器已就绪")

# ------------------------------
# 4. 会话管理器（与之前相同，略作简化）
# ------------------------------
class SessionManager:
    def __init__(self, redis_client: redis.Redis):
        self.redis = redis_client
        self.ttl = SESSION_TTL

    def _get_key(self, session_id: str) -> str:
        return f"chat:session:{session_id}"

    async def create_session(self, session_id: str = None) -> str:
        if not session_id:
            session_id = str(uuid.uuid4())
        key = self._get_key(session_id)
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        await self.redis.setex(key, self.ttl, json.dumps(messages, ensure_ascii=False))
        return session_id

    async def add_message(self, session_id: str, role: str, content: str) -> bool:
        key = self._get_key(session_id)
        data = await self.redis.get(key)
        if not data:
            return False
        messages = json.loads(data)
        messages.append({"role": role, "content": content})
        await self.redis.setex(key, self.ttl, json.dumps(messages, ensure_ascii=False))
        return True

    async def get_messages(self, session_id: str, max_turns: int = 10) -> Optional[List[Dict]]:
        key = self._get_key(session_id)
        data = await self.redis.get(key)
        if not data:
            return None
        messages = json.loads(data)
        if len(messages) > max_turns * 2 + 1:
            return [messages[0]] + messages[-(max_turns * 2):]
        return messages

    async def session_exists(self, session_id: str) -> bool:
        key = self._get_key(session_id)
        return await self.redis.exists(key) > 0

# ------------------------------
# 5. FastAPI 生命周期
# ------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    # 启动时：连接 Redis + 初始化 RAG
    if REDIS_URL:
        parsed = __import__('urllib.parse').urlparse(REDIS_URL)
        redis_client = redis.Redis(host=parsed.hostname, port=parsed.port, password=parsed.password, db=REDIS_DB, decode_responses=True)
    else:
        redis_client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB, decode_responses=True)
    app.state.redis = redis_client
    app.state.session_manager = SessionManager(redis_client)
    logger.info("Redis连接已建立")

    # 初始化 RAG
    init_rag()

    yield

    await redis_client.close()
    logger.info("Redis连接已关闭")

# ------------------------------
# 6. FastAPI 实例 & CORS
# ------------------------------
app = FastAPI(title="智能客服 - RAG增强版", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # 生产环境可改成具体前端域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

deepseek_client = AsyncOpenAI(api_key=api_key, base_url="https://api.deepseek.com")

# ------------------------------
# 7. Pydantic 模型
# ------------------------------
class ChatRequest(BaseModel):
    session_id: Optional[str] = None
    message: str

class ChatResponse(BaseModel):
    session_id: str
    reply: str

class CreateSessionResponse(BaseModel):
    session_id: str

# ------------------------------
# 8. 辅助函数：检索 + 构造增强消息
# ------------------------------
async def retrieve_knowledge(query: str) -> str:
    """根据用户问题检索相关知识片段，拼成字符串"""
    if retriever is None:
        return ""
    try:
        docs = retriever.invoke(query)
        if not docs:
            return ""
        # 将检索到的内容拼接，并加上【参考知识】标记
        context = "\n\n【参考知识】\n" + "\n---\n".join([doc.page_content for doc in docs])
        return context
    except Exception as e:
        logger.error(f"检索失败: {e}")
        return ""

async def generate_rag_reply(user_query: str, history_messages: List[Dict]) -> str:
    """生成带知识增强的回答"""
    # 1. 检索相关知识
    retrieved_knowledge = await retrieve_knowledge(user_query)
    
    # 2. 构建新的消息列表（保留原有系统prompt和历史对话）
    #    注意：我们不会把整段知识塞进 system，而是插入一条临时的 assistant 或 user 消息
    #    这里采用在用户问题前显式添加知识的方式，避免干扰对话历史格式
    enhanced_query = user_query
    if retrieved_knowledge:
        enhanced_query = f"{retrieved_knowledge}\n\n用户问题：{user_query}\n请基于上述【参考知识】回答用户问题。如果参考知识与问题无关，请忽略它。"
    
    # 3. 构造新的 messages（历史 + 增强后的用户消息）
    new_messages = history_messages.copy()   # 包含 system 和历史对话
    # 移除最后一条 user 消息（因为我们要用增强版替换），这里假设 history_messages 的最后一条是 user
    if new_messages and new_messages[-1]["role"] == "user":
        new_messages.pop()
    new_messages.append({"role": "user", "content": enhanced_query})
    
    # 4. 调用 DeepSeek
    try:
        response = await deepseek_client.chat.completions.create(
            model="deepseek-v4-pro",
            messages=new_messages,
            temperature=0.7,
            max_tokens=512,
        )
        return response.choices[0].message.content
    except Exception as e:
        logger.error(f"AI调用失败: {e}")
        raise HTTPException(status_code=500, detail="AI服务暂时不可用")

# ------------------------------
# 9. API 端点（与之前类似，但 /chat 使用新的 RAG 生成函数）
# ------------------------------
@app.post("/sessions", response_model=CreateSessionResponse)
async def new_session():
    session_id = await app.state.session_manager.create_session()
    return CreateSessionResponse(session_id=session_id)

@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    session_id = req.session_id
    if not session_id or not await app.state.session_manager.session_exists(session_id):
        session_id = await app.state.session_manager.create_session()
        logger.info(f"自动创建会话: {session_id}")

    # 保存用户消息
    await app.state.session_manager.add_message(session_id, "user", req.message)

    # 获取历史消息（不含本次用户消息，因为上面已添加）
    messages = await app.state.session_manager.get_messages(session_id, max_turns=10)

    # 使用 RAG 生成回复（内部会检索知识库并调用 AI）
    try:
        reply = await generate_rag_reply(req.message, messages)
    except HTTPException:
        # 确保失败时回滚用户消息
        raise
    except Exception as e:
        logger.error(f"未知错误: {e}")
        raise HTTPException(status_code=500, detail="服务内部错误")

    # 保存 AI 回复
    await app.state.session_manager.add_message(session_id, "assistant", reply)
    return ChatResponse(session_id=session_id, reply=reply)

@app.delete("/sessions/{session_id}")
async def delete_session(session_id: str):
    if await app.state.session_manager.delete_session(session_id):
        return {"status": "success", "message": "会话已删除"}
    raise HTTPException(status_code=404, detail="会话不存在")

@app.get("/")
async def root():
    return {"message": "智能客服 RAG 版已启动，请先 POST /sessions 创建会话"}
