import gc
import os
import shutil
import signal
import sys
from pathlib import Path

import gradio as gr
import torch
from dotenv import load_dotenv

from src.llm_client import LLMClient
from src.model_manager import DEFAULT_POOL_SIZE, MODELS, model_pool
from src.transcriber import LANGUAGE_OPTIONS, transcribe_audio
from src.utils import cleanup_temp_root

load_dotenv()
AUTH_USER = os.getenv("WHISPER_AUTH_USERNAME", "admin")
AUTH_PASS = os.getenv("WHISPER_AUTH_PASSWORD", "password123")
llm_client = LLMClient()

if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


def get_device_info() -> tuple[str, str]:
    if torch.cuda.is_available():
        return "cuda", f"🟢 CUDA aktiv ({torch.cuda.get_device_name(0)})"
    return "cpu", "🟡 CPU Modus"


def cleanup_temp() -> None:
    cleanup_temp_root()
    for temp_dir in [Path("/tmp/gradio"), Path(os.getenv("TMPDIR", "/tmp")) / "gradio"]:
        if temp_dir.exists():
            shutil.rmtree(temp_dir, ignore_errors=True)


def remote_restart() -> str:
    os.kill(os.getpid(), signal.SIGTERM)
    return "Neustart ausgelöst. systemd startet den Dienst neu."


def clear_gpu_memory() -> str:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        return "CUDA-Cache geleert."
    return "Kein CUDA-Backend aktiv."


def kill_python_zombies() -> str:
    return "systemd verwaltet den Prozess. Für harte Neustarts nutze systemctl restart whisperx."


def get_pool_status() -> str:
    available = model_pool.get_available_count()
    total = model_pool.pool_size or DEFAULT_POOL_SIZE
    queue_size = model_pool.get_queue_size()
    llm_status = "✅ Verbunden" if llm_client.is_available() else "❌ Nicht verfügbar"
    return f"Worker: {available}/{total} verfügbar | Queue: {queue_size} wartend | LLM: {llm_status}"


def transcribe_with_llm_review(
    file_path,
    language,
    model_name,
    device,
    initial_prompt_user="",
    speed_mode=False,
    llm_review=False,
):
    result = transcribe_audio(
        file_path=file_path,
        language=language,
        model_name=model_name,
        device=device,
        initial_prompt_user=initial_prompt_user,
        speed_mode=speed_mode,
    )

    if llm_review and llm_client.is_available() and result[0].strip():
        corrected = llm_client.review_transcription(result[0], language=language)
        if corrected.strip():
            result[0] = corrected.strip()
            result[2] = f"{result[2]} | LLM-Korrektur angewendet"
    return result


def build_interface():
    default_device, device_label = get_device_info()

    with gr.Blocks(theme=gr.themes.Soft(), title="WhisperX Backend – DGX Spark") as interface:
        gr.Markdown("# 🎙️ WhisperX Schreibdienst Backend – DGX Spark Edition")
        gr.Markdown(
            f"Status: {device_label} | Pool: {model_pool.pool_size or DEFAULT_POOL_SIZE} Worker | "
            f"Max Queue: {(model_pool.pool_size or DEFAULT_POOL_SIZE) * 5}"
        )

        with gr.Tabs():
            with gr.TabItem("Transkription"):
                with gr.Row():
                    with gr.Column(variant="panel"):
                        gr.Markdown("### ⚙️ Konfiguration")
                        file_input = gr.File(label="Audio/Video Datei")
                        language_dropdown = gr.Dropdown(
                            choices=list(LANGUAGE_OPTIONS.keys()),
                            label="Sprache",
                            value="Identify",
                        )
                        model_dropdown = gr.Dropdown(
                            choices=MODELS,
                            label="Modell",
                            value=os.getenv("WHISPERX_MODEL", "large-v3"),
                            allow_custom_value=True,
                        )
                        device_dropdown = gr.Dropdown(
                            choices=["cuda", "cpu"] if torch.cuda.is_available() else ["cpu"],
                            label="Gerät",
                            value=os.getenv("WHISPERX_DEVICE", default_device),
                        )
                        initial_prompt_input = gr.Textbox(label="Initial Prompt", lines=3)
                        speed_mode_input = gr.Checkbox(label="Speed Mode", value=False, visible=False)
                        llm_review_input = gr.Checkbox(
                            label="🤖 Korrektur-LLM anwenden",
                            value=False,
                            info="Nutzt eine OpenAI-kompatible LLM-API oder Ollama für minimale Nachkorrekturen.",
                        )
                        transcribe_button = gr.Button("▶️ Transkription Starten", variant="primary")

                    with gr.Column(variant="panel"):
                        gr.Markdown("### 📝 Auswertung")
                        time_output = gr.Textbox(label="Bearbeitungszeit / Status")
                        text_output = gr.TextArea(label="Text-Ergebnis", lines=12)
                        json_output = gr.TextArea(label="JSON Daten", visible=False)

            with gr.TabItem("🔧 System Admin"):
                gr.Markdown("### 🛠️ Hardware-Management (DGX Spark / CUDA)")
                with gr.Row():
                    btn_vram = gr.Button("🧹 VRAM leeren")
                    btn_zombie = gr.Button("💀 Prozess-Hinweis")
                    btn_reboot = gr.Button("🔄 Server Neustart")
                    btn_status = gr.Button("📊 Pool Status")

                admin_status = gr.Textbox(label="Aktion-Ergebnis", interactive=False)
                btn_vram.click(fn=clear_gpu_memory, outputs=admin_status, api_name="system_cleanup")
                btn_zombie.click(fn=kill_python_zombies, outputs=admin_status, api_name="system_kill_zombies")
                btn_reboot.click(fn=remote_restart, outputs=admin_status, api_name="system_reboot")
                btn_status.click(fn=get_pool_status, outputs=admin_status, api_name="system_pool_status")

        transcribe_button.click(
            fn=transcribe_with_llm_review,
            inputs=[
                file_input,
                language_dropdown,
                model_dropdown,
                device_dropdown,
                initial_prompt_input,
                speed_mode_input,
                llm_review_input,
            ],
            outputs=[text_output, json_output, time_output],
            api_name="start_process",
            concurrency_limit=(model_pool.pool_size or DEFAULT_POOL_SIZE) * 5,
        )

    return interface


if __name__ == "__main__":
    cleanup_temp()
    device, label = get_device_info()
    print(f"--- SYSTEM: {label} ---")
    print(f"--- SYSTEM: Worker Pool: {model_pool.pool_size or DEFAULT_POOL_SIZE} Instanzen ---")
    print(f"--- SYSTEM: LLM Client: {'Aktiv' if llm_client.is_available() else 'Inaktiv'} ---")

    if torch.cuda.is_available():
        torch.cuda.init()

    interface = build_interface()
    interface.launch(
        server_name=os.getenv("WHISPERX_HOST", "0.0.0.0"),
        server_port=int(os.getenv("WHISPERX_PORT", "7860")),
        auth=(AUTH_USER, AUTH_PASS),
        auth_message="WhisperX Backend Login",
        max_threads=(model_pool.pool_size or DEFAULT_POOL_SIZE) * 7,
    )
