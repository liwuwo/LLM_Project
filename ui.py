import gradio as gr
import os
import base64
from llms import deepseek_llm,local_llm

currentLLM = deepseek_llm

with gr.Blocks(title="LLM助手") as demo:

    gr.Markdown("# LLM 聊天助手")

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


if __name__ == "__main__":
    demo.launch(theme=gr.themes.Soft())