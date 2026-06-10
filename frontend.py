# frontend.py
import streamlit as st
import requests
import json
import time

# ---------- 1. 页面配置 ----------
st.set_page_config(page_title="智能客服", page_icon="🛒", layout="centered")
st.title("🛒 智能客服小智")
st.caption("我是您的购物助手，有什么可以帮您的吗？")

# ---------- 2. 后端服务配置 ----------
# 确保这里填写的端口和你的backend.py启动端口一致
API_BASE_URL = "http://localhost:8000"

# ---------- 3. 初始化会话状态 ----------
if "session_id" not in st.session_state:
    # 首次访问时，创建一个新的后端会话
    try:
        response = requests.post(f"{API_BASE_URL}/sessions")
        if response.status_code == 200:
            st.session_state.session_id = response.json()["session_id"]
            # 初始化本地消息历史
            st.session_state.messages = []
        else:
            st.error("❌ 无法连接到后端服务，请确保后端已启动")
            st.stop()
    except Exception as e:
        st.error(f"❌ 连接后端失败: {e}")
        st.stop()

# ---------- 4. 展示历史消息 ----------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ---------- 5. 处理用户输入 ----------
if prompt := st.chat_input("请输入您的购物问题..."):
    # 在界面上立即显示用户的消息
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # 显示“助手正在输入”的状态提示
    with st.chat_message("assistant"):
        # 创建一个占位符，用于动态更新内容
        message_placeholder = st.empty()
        with st.spinner("小智正在思考..."):
            try:
                # 调用后端API
                response = requests.post(
                    f"{API_BASE_URL}/chat",
                    json={"session_id": st.session_state.session_id, "message": prompt},
                    timeout=30
                )
                
                if response.status_code == 200:
                    reply = response.json()["reply"]
                    # 用打字机效果逐字显示回复
                    full_response = ""
                    for chunk in reply.split():
                        full_response += chunk + " "
                        message_placeholder.markdown(full_response + "▌")
                        time.sleep(0.05)
                    message_placeholder.markdown(full_response)
                else:
                    error_msg = f"❌ 后端错误: {response.status_code} - {response.text}"
                    message_placeholder.markdown(error_msg)
                    reply = error_msg
            except requests.exceptions.Timeout:
                error_msg = "⏰ 请求超时，请稍后再试"
                message_placeholder.markdown(error_msg)
                reply = error_msg
            except Exception as e:
                error_msg = f"❌ 发生错误: {str(e)}"
                message_placeholder.markdown(error_msg)
                reply = error_msg

    # 将助手的回复也存入历史记录
    st.session_state.messages.append({"role": "assistant", "content": reply})