import gradio as gr
import os
import base64
from llm.llms import local_llm

currentLLM = local_llm

# MIME类型映射
MIME_MAP = {
    ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png",
    ".gif": "image/gif", ".webp": "image/webp", ".bmp": "image/bmp",
    ".mp3": "audio/mpeg", ".wav": "audio/wav", ".flac": "audio/flac",
    ".ogg": "audio/ogg", ".m4a": "audio/mp4",
    ".mp4": "video/mp4", ".webm": "video/webm", ".mov": "video/quicktime"
}

with gr.Blocks(title="LLM助手") as demo:
    gr.Markdown("# LLM 聊天助手")
    chatbot = gr.Chatbot(label="多模态的对话", height=500)

    with gr.Tabs():
        with gr.TabItem("图片"):
            image_input = gr.File(height=100,label="上传图片（可多选）", file_count="multiple", file_types=["image"])
        with gr.TabItem("音频"):
            audio_input = gr.File(height=100,label="上传音频（可多选）", file_count="multiple", file_types=["audio"])
        with gr.TabItem("视频"):
            video_input = gr.File(height=100,label="上传视频（可多选）", file_count="multiple", file_types=["video"])

    with gr.Row():
        user_input = gr.Textbox(placeholder="请输入你的问题...", show_label=False, scale=8, lines=5)
        send_btn = gr.Button("发送", variant="primary", scale=1)
    with gr.Row():
        clear_btn = gr.Button("清空对话", variant="secondary")


    def respond(message, chat_history, images, audios, videos):
        if not message.strip():
            yield "", chat_history, images, audios, videos
            return

        chat_history = chat_history or []
        # UI 展示只用文本，多模态内容不直接暴露给 Chatbot 组件
        chat_history.append({"role": "user", "content": message})
        chat_history.append({"role": "assistant", "content": ""})
        yield "", chat_history, None, None, None

        def encode_file(file_path, mime_type):
            with open(file_path, "rb") as fp:
                b64 = base64.b64encode(fp.read()).decode("utf-8")
            return f"data:{mime_type};base64,{b64}"

        def get_media_entry(file_path):
            if not file_path or not os.path.exists(file_path):
                return None
            ext = os.path.splitext(file_path)[1].lower()
            mime = MIME_MAP.get(ext)
            if not mime:
                return None

            if mime.startswith("image"):
                return {"type": "image_url", "image_url": {"url": encode_file(file_path, mime)}}
            elif mime.startswith("audio"):
                data_uri = encode_file(file_path, mime)
                base64_data = data_uri.split(",", 1)[1]
                return {"type": "input_audio", "input_audio": {"data": base64_data, "format": ext[1:]}}
            elif mime.startswith("video"):
                return {"type": "video_url", "video_url": {"url": encode_file(file_path, mime)}}
            return None

        # 构建多模态内容列表
        content = []
        for f in (images or []):
            entry = get_media_entry(f.path if hasattr(f, "path") else f)
            if entry: content.append(entry)
        for f in (audios or []):
            entry = get_media_entry(f.path if hasattr(f, "path") else f)
            if entry: content.append(entry)
        for f in (videos or []):
            entry = get_media_entry(f.path if hasattr(f, "path") else f)
            if entry: content.append(entry)
        content.append({"type": "text", "text": message})

        # 分离 UI 历史与 LLM 上下文
        llm_messages = []
        for m in chat_history[:-2]:
            # 历史消息统一转为字符串格式（LLM 通常兼容）
            llm_messages.append({"role": m["role"], "content": m["content"]})
        llm_messages.append({"role": "user", "content": content})

        try:
            for chunk in currentLLM.stream(llm_messages):
                # 兼容不同 LLM 返回结构
                delta = chunk.content if hasattr(chunk, "content") else chunk
                chat_history[-1]["content"] += delta
                yield "", chat_history, None, None, None
        except Exception as e:
            chat_history[-1]["content"] = f"❌ 模型调用失败: {str(e)}"
            yield "", chat_history, None, None, None


    send_btn.click(respond, inputs=[user_input, chatbot, image_input, audio_input, video_input],
                   outputs=[user_input, chatbot, image_input, audio_input, video_input])
    user_input.submit(respond, inputs=[user_input, chatbot, image_input, audio_input, video_input],
                      outputs=[user_input, chatbot, image_input, audio_input, video_input])
    clear_btn.click(lambda: ("", [], None, None, None),
                    outputs=[user_input, chatbot, image_input, audio_input, video_input])

if __name__ == "__main__":
    demo.launch(theme=gr.themes.Soft())
