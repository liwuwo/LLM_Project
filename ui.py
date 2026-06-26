import gradio as gr
import os
import base64
from llm.llms import deepseek_llm
from agents.text_SQL_agent import TextSQLAgent
from utils.constants import USE_LOCAL_LLM
currentLLM = deepseek_llm

# 初始化 TextSQLAgent（懒加载，首次使用时才创建）
_sql_agent = None


def get_sql_agent():
    """获取 TextSQLAgent 实例（单例懒加载）。"""
    global _sql_agent



    if _sql_agent is None:
        _sql_agent = TextSQLAgent(use_local_llm=USE_LOCAL_LLM, max_iterations=15, verbose=False)
    return _sql_agent


with gr.Blocks(title="LLM助手") as demo:

    gr.Markdown("# LLM 聊天助手")

    with gr.Tabs() as tabs:
        # ============ 标签页 1：多模态聊天 ============
        with gr.TabItem("多模态聊天", id="chat_tab"):
            chatbot = gr.Chatbot(
                label="多模态的对话",
                height=500,
            )

            with gr.Row():
                user_input = gr.Textbox(
                    placeholder="请输入你的问题...",
                    show_label=False,
                    scale=8,
                    lines=5,
                )
                send_btn = gr.Button("发送", variant="primary", scale=1)

            with gr.Accordion("附件（可选）", open=False):
                with gr.Row():
                    image_input = gr.File(
                        label="图片",
                        file_count="multiple",
                        file_types=["image"],
                        height=120,
                    )
                    audio_input = gr.File(
                        label="音频",
                        file_count="multiple",
                        file_types=["audio"],
                        height=120,
                    )
                    video_input = gr.File(
                        label="视频",
                        file_count="multiple",
                        file_types=["video"],
                        height=120,
                    )

            with gr.Row():
                clear_btn = gr.Button("清空对话", variant="secondary")

            def respond(message, chat_history, images, audios, videos):
                if not message.strip():
                    yield "", chat_history, images, audios, videos
                    return
                chat_history = chat_history or []
                chat_history.append({"role": "user", "content": message})
                chat_history.append({"role": "assistant", "content": ""})
                yield "", chat_history, None, None, None

                # 工具函数：读取文件并编码为 base64 data URI
                def encode_file(file_path, mime_type):
                    with open(file_path, "rb") as fp:
                        b64 = base64.b64encode(fp.read()).decode("utf-8")
                    return f"data:{mime_type};base64,{b64}"

                # 根据扩展名判断类型并编码
                def get_media_entry(file_path):
                    ext = os.path.splitext(file_path)[1].lower()
                    image_exts = {".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp"}
                    audio_exts = {".mp3", ".wav", ".flac", ".ogg", ".m4a"}
                    video_exts = {".mp4", ".webm", ".mov", ".avi", ".mkv"}
                    if ext in image_exts:
                        return {"type": "image_url", "image_url": {"url": encode_file(file_path, "image/jpeg")}}
                    elif ext in audio_exts:
                        return {"type": "input_audio", "input_audio": {"data": encode_file(file_path, f"audio/{ext[1:]}").split(",", 1)[1], "format": ext[1:]}}
                    elif ext in video_exts:
                        return {"type": "video_url", "video_url": {"url": encode_file(file_path, f"video/{ext[1:]}")}}
                    return None

                # 构建多模态消息内容
                content = []
                for f in images or []:
                    entry = get_media_entry(f.path)
                    if entry:
                        content.append(entry)
                for f in audios or []:
                    entry = get_media_entry(f.path)
                    if entry:
                        content.append(entry)
                for f in videos or []:
                    entry = get_media_entry(f.path)
                    if entry:
                        content.append(entry)
                content.append({"type": "text", "text": message})

                # 历史消息 + 当前多模态消息
                messages = [{"role": m["role"], "content": m["content"]} for m in chat_history[:-2]]
                messages.append({"role": "user", "content": content})

                for chunk in currentLLM.stream(messages):
                    chat_history[-1]["content"] += chunk.content
                    yield "", chat_history, None, None, None

            send_btn.click(
                respond,
                inputs=[user_input, chatbot, image_input, audio_input, video_input],
                outputs=[user_input, chatbot, image_input, audio_input, video_input],
            )
            user_input.submit(
                respond,
                inputs=[user_input, chatbot, image_input, audio_input, video_input],
                outputs=[user_input, chatbot, image_input, audio_input, video_input],
            )
            clear_btn.click(lambda: ("", []), outputs=[user_input, chatbot])

        # ============ 标签页 2：文本转 SQL 查询 ============
        with gr.TabItem("文本转 SQL 查询", id="sql_tab"):
            gr.Markdown(
                "### 💬 自然语言转数据库查询\n"
                "用中文描述你想查询的内容，系统将自动转换为 SQL 并查询数据库返回结果。"
            )

            sql_chatbot = gr.Chatbot(
                label="Text-to-SQL 对话",
                height=500,
            )

            with gr.Row():
                sql_user_input = gr.Textbox(
                    placeholder="示例：查询哪些会员购买了商品，具体的商品单价和总消费金额是多少",
                    show_label=False,
                    scale=8,
                    lines=5,
                )
                sql_send_btn = gr.Button("查询", variant="primary", scale=1)

            with gr.Row():
                sql_clear_btn = gr.Button("清空对话", variant="secondary")
                show_thoughts = gr.Checkbox(label="显示思考推理过程", value=False, scale=1)
                sql_status = gr.Markdown("*等待输入...*")

            def _strip_react_format(text):
                """彻底清理文本中残留的 ReAct 格式标记（Thought/Action/Action Input/Final Answer 等）。"""
                if not text:
                    return text
                import re
                # 移除开头出现的格式标记行
                lines = str(text).split('\n')
                filtered = []
                for line in lines:
                    stripped = line.strip()
                    # 移除纯格式标记行
                    if re.match(r'^(Thought|Action|Action Input|Observation|Final Answer)\s*[:：]', stripped):
                        continue
                    filtered.append(line)
                cleaned = '\n'.join(filtered).strip()
                return cleaned

            def format_intermediate_steps(intermediate_steps):
                """将 Agent 的中间步骤格式化为 Markdown 文本，便于展示工具调用过程。"""
                if not intermediate_steps:
                    return ""
                lines = ["\n---\n**🧭 推理过程："]
                for step_idx, (action, observation) in enumerate(intermediate_steps, 1):
                    tool_name = getattr(action, 'tool', 'unknown')
                    tool_input = getattr(action, 'tool_input', '')
                    lines.append(f"\n**步骤 {step_idx}**：调用工具 `{tool_name}`")
                    if tool_input:
                        input_str = str(tool_input)
                        # 对于 SQL 类型，直接高亮显示
                        lines.append(f"> 输入：`{input_str}`")
                    obs_str = str(observation)
                    if len(obs_str) > 600:
                        obs_str = obs_str[:600] + "...（已截断）"
                    lines.append(f"> 输出：\n```\n{obs_str}\n```")
                lines.append("")
                return "\n".join(lines)

            def sql_respond(message, chat_history, show_thoughts_flag):
                if not message or not message.strip():
                    yield "", chat_history, "*请输入要查询的问题...*"
                    return

                chat_history = chat_history or []
                chat_history.append({"role": "user", "content": message})
                chat_history.append({"role": "assistant", "content": "⏳ 正在分析问题并准备查询..."})
                yield "", chat_history, "🚀 Agent 正在执行查询，请稍候..."

                try:
                    agent = get_sql_agent()
                    result = agent.query(message)

                    answer = result.get('answer', '查询失败')
                    steps = result.get('intermediate_steps', [])
                    success = result.get('success', False)

                    # 最终清理：确保答案中不包含任何 ReAct 格式标记
                    answer = _strip_react_format(answer)

                    # 根据开关决定是否显示推理过程
                    display_text = answer
                    if show_thoughts_flag:
                        steps_text = format_intermediate_steps(steps)
                        if steps_text:
                            display_text += steps_text

                    chat_history[-1]["content"] = display_text
                    if show_thoughts_flag:
                        status_md = f"✅ 查询完成，共执行 **{len(steps)}** 步工具调用。" if success else "⚠️ 查询过程中出现问题。"
                    else:
                        status_md = "✅ 查询完成。" if success else "⚠️ 查询过程中出现问题。"
                    yield "", chat_history, status_md

                except Exception as e:
                    chat_history[-1]["content"] = f"❌ 查询执行失败：{str(e)}"
                    yield "", chat_history, "*查询出错，请检查数据库连接或问题表述。*"

            sql_send_btn.click(
                sql_respond,
                inputs=[sql_user_input, sql_chatbot, show_thoughts],
                outputs=[sql_user_input, sql_chatbot, sql_status],
            )
            sql_user_input.submit(
                sql_respond,
                inputs=[sql_user_input, sql_chatbot, show_thoughts],
                outputs=[sql_user_input, sql_chatbot, sql_status],
            )
            sql_clear_btn.click(
                lambda: ("", [], "*已清空，等待新的查询...*"),
                outputs=[sql_user_input, sql_chatbot, sql_status],
            )


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",  # 允许外部访问（AutoDL 部署必须）
        server_port=7860,
        theme=gr.themes.Soft(),
        share=False,  # 如需临时公网分享可改为 True
    )