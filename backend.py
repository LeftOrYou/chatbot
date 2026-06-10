# backend.py - Redis存储版智能客服
import os
import uuid
import json
import logging
from typing import List, Dict, Optional
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware   # 导入 CORS 中间件
from pydantic import BaseModel
from dotenv import load_dotenv
from openai import AsyncOpenAI
import redis.asyncio as redis

# ------------------------------
# 1. 配置日志
# ------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ------------------------------
# 2. 加载配置
# ------------------------------
load_dotenv('deepseek.env')
api_key = os.getenv("DEEPSEEK_API_KEY")
if not api_key:
    raise RuntimeError("请设置 DEEPSEEK_API_KEY")

# ------------------------------
# 3. Redis配置
# ------------------------------
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
REDIS_DB = int(os.getenv("REDIS_DB", 0))
SESSION_TTL = int(os.getenv("SESSION_TTL", 1800))  # 30分钟过期

# 电商客服System Prompt
SYSTEM_PROMPT = """
你是一个专业、热情、高效的电商客服，名叫“小智”。
你的工作是为用户提供商品咨询、订单查询、物流跟踪、售后处理等服务。

【重要规则】
1. 只回答与购物相关的问题。对于无关问题，礼貌地表示无法回答并引导回购物主题。
2. 回答必须简洁、清晰，每句话不超过30字，避免冗长。
3. 遇到不确定的信息（如具体订单状态），引导用户提供订单号或联系人工客服。
4. 始终使用“您”、“请”、“谢谢”等敬语，语气亲切。
5. 严禁编造促销活动、价格、库存等任何未提供的数据。

【产品知识库 & FAQ】
以下是已知信息，只基于这些内容回答；超出范围请让用户提供详情或转人工。

1. 商品:
   - 本店主营：手机壳、数据线、充电宝、耳机。
   - 手机壳材质：硅胶、TPU、玻璃，价格 19-59 元。
   - 数据线：苹果/Type-C，1米/2米，价格 15-35 元。
   - 充电宝：10000mAh/20000mAh，价格 69-129 元。
   - 耳机：有线/蓝牙，价格 49-199 元。

2. 物流:
   - 默认中通快递，下单后24小时内发货（周末及节假日顺延）。
   - 满99元包邮，不满99元收运费8元。
   - 提供物流单号后，可在“我的订单”中跟踪。

3. 售后:
   - 支持七天无理由退货（不影响二次销售，买家承担运费）。
   - 质量问题：请在签收后48小时内联系客服并提供照片，商家承担退换货运费。
   - 退款到账时间：原路返回，1-7个工作日。

4. 支付:
   - 支持微信支付、支付宝、银行卡。
   - 不支持货到付款。

5. 优惠:
   - 新用户关注店铺可领5元无门槛券。
   - 全场满200减20（特价商品除外）。

【输出格式要求】
- 每条回复结尾加上“😊”或“🌟”。
- 如果用户表达感谢，回复“不客气，祝您购物愉快！”

请记住以上所有规则，现在开始服务用户。
"""

# ------------------------------
# 4. Redis操作封装
# ------------------------------
class SessionManager:
    """会话管理器（Redis存储）"""
    
    def __init__(self, redis_client: redis.Redis):
        self.redis = redis_client
        self.ttl = SESSION_TTL
    
    def _get_key(self, session_id: str) -> str:
        """生成Redis键名"""
        return f"chat:session:{session_id}"
    
    async def create_session(self, session_id: str = None) -> str:
        """创建新会话，返回session_id"""
        if not session_id:
            session_id = str(uuid.uuid4())
        key = self._get_key(session_id)
        # 存储消息列表：初始化时放入system prompt
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        await self.redis.setex(
            key,
            self.ttl,
            json.dumps(messages, ensure_ascii=False)
        )
        logger.info(f"创建会话: {session_id}")
        return session_id
    
    async def add_message(self, session_id: str, role: str, content: str) -> bool:
        """添加消息到会话历史"""
        key = self._get_key(session_id)
        # 获取当前消息列表
        data = await self.redis.get(key)
        if not data:
            return False
        messages = json.loads(data)
        messages.append({"role": role, "content": content})
        # 更新并重置过期时间
        await self.redis.setex(key, self.ttl, json.dumps(messages, ensure_ascii=False))
        return True
    
    async def get_messages(self, session_id: str, max_turns: int = 10) -> Optional[List[Dict]]:
        """获取会话消息列表（支持滑动窗口）"""
        key = self._get_key(session_id)
        data = await self.redis.get(key)
        if not data:
            return None
        messages = json.loads(data)
        # 滑动窗口：保留system prompt + 最近max_turns轮对话
        if len(messages) > max_turns * 2 + 1:
            return [messages[0]] + messages[-(max_turns * 2):]
        return messages
    
    async def session_exists(self, session_id: str) -> bool:
        """检查会话是否存在"""
        key = self._get_key(session_id)
        return await self.redis.exists(key) > 0
    
    async def delete_session(self, session_id: str) -> bool:
        """删除会话"""
        key = self._get_key(session_id)
        result = await self.redis.delete(key)
        if result:
            logger.info(f"删除会话: {session_id}")
        return result > 0

# ------------------------------
# 5. FastAPI生命周期管理
# ------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    # 启动时：创建Redis连接池
    redis_client = redis.Redis(
        host=REDIS_HOST,
        port=REDIS_PORT,
        db=REDIS_DB,
        decode_responses=True  # 自动解码为字符串
    )
    app.state.redis = redis_client
    app.state.session_manager = SessionManager(redis_client)
    logger.info("Redis连接已建立")
    
    yield
    
    # 关闭时：清理连接
    await redis_client.close()
    logger.info("Redis连接已关闭")

# ------------------------------
# 6. FastAPI应用
# ------------------------------
app = FastAPI(
    title="智能客服 - Redis记忆版",
    lifespan=lifespan
)

# ------------------------------
# CORS 配置（重要！）
# ------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501"],  # 允许 Streamlit 前端地址访问，关键点：allow_origins 设置允许来自 Streamlit 默认地址（http://localhost:8501）的跨域请求。
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 初始化DeepSeek客户端
deepseek_client = AsyncOpenAI(
    api_key=api_key,
    base_url="https://api.deepseek.com",
)

# Pydantic模型
class ChatRequest(BaseModel):
    session_id: Optional[str] = None
    message: str

class ChatResponse(BaseModel):
    session_id: str
    reply: str

class CreateSessionResponse(BaseModel):
    session_id: str

# API端点
@app.post("/sessions", response_model=CreateSessionResponse)
async def new_session():
    """创建新会话"""
    session_id = await app.state.session_manager.create_session()
    return CreateSessionResponse(session_id=session_id)

@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    """发送消息（支持自动创建会话）"""
    session_id = req.session_id
    
    # 如果没有session_id或会话不存在，自动创建新会话
    if not session_id or not await app.state.session_manager.session_exists(session_id):
        session_id = await app.state.session_manager.create_session()
        logger.info(f"自动创建会话: {session_id}")
    
    # 保存用户消息
    await app.state.session_manager.add_message(session_id, "user", req.message)
    
    # 获取历史消息（限制最近10轮）
    messages = await app.state.session_manager.get_messages(session_id, max_turns=10)
    
    try:
        response = await deepseek_client.chat.completions.create(
            model="deepseek-v4-pro",
            messages=messages,
            temperature=0.7,
            max_tokens=256,
        )
        reply = response.choices[0].message.content
    except Exception as e:
        logger.error(f"AI调用失败: {e}")
        raise HTTPException(status_code=500, detail="AI服务暂时不可用")
    
    # 保存AI回复
    await app.state.session_manager.add_message(session_id, "assistant", reply)
    
    return ChatResponse(session_id=session_id, reply=reply)

@app.delete("/sessions/{session_id}")
async def delete_session(session_id: str):
    """删除会话"""
    if await app.state.session_manager.delete_session(session_id):
        return {"status": "success", "message": "会话已删除"}
    raise HTTPException(status_code=404, detail="会话不存在")

@app.get("/")
async def root():
    return {"message": "智能客服已启动，请先 POST /sessions 创建会话"}
