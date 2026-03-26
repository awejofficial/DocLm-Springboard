import reflex as rx
import os
from typing import List

from DOCU_AI.components.navbar import navbar
from DOCU_AI.components.footer import footer
from DOCU_AI.components.log_panel import log_panel
from DOCU_AI.states.rag_state import ChatState
from DOCU_AI.states.upload_state import UploadState

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UPLOAD_DIR = os.path.join(BASE_DIR, "documents")
os.makedirs(UPLOAD_DIR, exist_ok=True)


def file_badge(name: str):
    return rx.hstack(
        rx.text("📄", font_size="14px"),
        rx.text(
            name,
            color="#a0aec0",
            font_size="13px",
            font_family="'Courier New', monospace",
            flex="1",
        ),
        rx.button(
            "✕",
            on_click=UploadState.delete_file(name),
            bg="transparent",
            color="#4a5568",
            font_size="12px",
            padding="2px 6px",
            border_radius="4px",
            _hover={"color": "#fc8181", "bg": "#1f0d0d"},
            cursor="pointer",
        ),
        padding_x="12px",
        padding_y="6px",
        bg="#0d0f14",
        border="1px solid #1e2535",
        border_radius="6px",
        width="100%",
        align="center",
    )


def upload():
    return rx.box(
        navbar(),

        rx.box(
            rx.vstack(

                # Page title
                rx.vstack(
                    rx.text(
                        "DOCUMENT INGESTION",
                        color="#00ffe7",
                        font_size="11px",
                        font_weight="700",
                        letter_spacing="0.2em",
                        font_family="'Courier New', monospace",
                    ),
                    rx.heading("Upload Documents", size="7", color="#e2e8f0", font_weight="800"),
                    rx.text(
                        "Upload PDF or TXT files. They'll be indexed into the vector store automatically.",
                        color="#4a5568",
                        font_size="14px",
                    ),
                    spacing="2",
                    align="start",
                    margin_bottom="32px",
                ),

                rx.hstack(
                    # Upload zone
                    rx.vstack(
                        rx.upload(
                            rx.vstack(
                                rx.text("⬆", font_size="36px"),
                                rx.text(
                                    "Drop files here or click to browse",
                                    color="#4a5568",
                                    font_size="14px",
                                    font_weight="500",
                                ),
                                rx.text(
                                    "Supported: .pdf  .txt",
                                    color="#2d3748",
                                    font_size="12px",
                                    font_family="'Courier New', monospace",
                                ),
                                spacing="3",
                                align="center",
                                padding="40px",
                            ),
                            id="upload1",
                            border="2px dashed #1e2535",
                            border_radius="12px",
                            bg="#0d0f14",
                            _hover={"border_color": "#00ffe7", "bg": "#00ffe705"},
                            transition="all 0.25s",
                            cursor="pointer",
                            width="100%",
                        ),

                        # Selected files preview
                        rx.foreach(
                            rx.selected_files("upload1"),
                            lambda f: rx.hstack(
                                rx.text("📎", font_size="13px"),
                                rx.text(f, color="#a0aec0", font_size="12px"),
                                padding_x="10px",
                                padding_y="4px",
                                bg="#060810",
                                border_radius="4px",
                            ),
                        ),

                        # Upload button
                        rx.hstack(
                            rx.button(
                                rx.cond(
                                    UploadState.uploading,
                                    rx.hstack(rx.spinner(size="1"), rx.text("Uploading..."), spacing="2"),
                                    rx.text("Upload & Index →"),
                                ),
                                on_click=UploadState.handle_upload(
                                    rx.upload_files(upload_id="upload1")
                                ),
                                bg="#00ffe7",
                                color="#0d0f14",
                                font_weight="700",
                                font_size="14px",
                                padding_x="24px",
                                padding_y="10px",
                                border_radius="8px",
                                _hover={"bg": "#00d4c5"},
                                transition="all 0.2s",
                                cursor="pointer",
                                disabled=UploadState.uploading,
                            ),
                            rx.button(
                                "Clear",
                                on_click=rx.clear_selected_files("upload1"),
                                bg="transparent",
                                color="#4a5568",
                                font_size="13px",
                                border="1px solid #1e2535",
                                padding_x="16px",
                                padding_y="10px",
                                border_radius="8px",
                                _hover={"border_color": "#4a5568", "color": "#a0aec0"},
                                cursor="pointer",
                            ),
                            spacing="3",
                        ),

                        # Success/Error messages
                        rx.cond(
                            UploadState.upload_done,
                            rx.box(
                                rx.hstack(
                                    rx.text("✅", font_size="14px"),
                                    rx.text(
                                        "Files uploaded successfully! Vector store will rebuild on next query.",
                                        color="#68d391",
                                        font_size="13px",
                                    ),
                                    spacing="2",
                                ),
                                padding="12px 16px",
                                bg="#0d1f14",
                                border="1px solid #1a4731",
                                border_radius="8px",
                            ),
                        ),

                        rx.cond(
                            UploadState.error_msg != "",
                            rx.box(
                                rx.text(
                                    "⚠️ " + UploadState.error_msg,
                                    color="#fc8181",
                                    font_size="13px",
                                ),
                                padding="12px 16px",
                                bg="#1f0d0d",
                                border="1px solid #4a1515",
                                border_radius="8px",
                            ),
                        ),

                        # Indexed files list
                        rx.box(
                            rx.vstack(
                                rx.hstack(
                                    rx.text(
                                        "INDEXED FILES",
                                        color="#00ffe7",
                                        font_size="10px",
                                        font_weight="700",
                                        letter_spacing="0.15em",
                                        font_family="'Courier New', monospace",
                                    ),
                                    rx.spacer(),
                                    rx.button(
                                        "🗑 Clear All",
                                        on_click=UploadState.clear_all_files,
                                        bg="transparent",
                                        color="#fc8181",
                                        font_size="11px",
                                        border="1px solid #4a1515",
                                        padding_x="10px",
                                        padding_y="4px",
                                        border_radius="6px",
                                        _hover={"bg": "#1f0d0d", "border_color": "#fc8181"},
                                        cursor="pointer",
                                    ),
                                    width="100%",
                                    align="center",
                                    margin_bottom="8px",
                                ),
                                rx.cond(
                                    ChatState.uploaded_files == [],
                                    rx.text(
                                        "No files indexed yet",
                                        color="#2d3748",
                                        font_size="13px",
                                        font_family="'Courier New', monospace",
                                    ),
                                    rx.vstack(
                                        rx.foreach(
                                            ChatState.uploaded_files,
                                            file_badge,
                                        ),
                                        spacing="2",
                                        align="start",
                                    ),
                                ),
                                align="start",
                                width="100%",
                            ),
                            padding="16px",
                            bg="#0a0c10",
                            border="1px solid #1e2535",
                            border_radius="10px",
                            width="100%",
                        ),

                        spacing="4",
                        align="start",
                        width="100%",
                        on_mount=ChatState.load_uploaded_files,
                    ),

                    # Log panel
                    rx.box(
                        log_panel(),
                        width="380px",
                        flex_shrink="0",
                    ),

                    spacing="6",
                    align="start",
                    width="100%",
                    wrap="wrap",
                ),

                max_width="1100px",
                margin="0 auto",
                padding="48px 40px",
                width="100%",
            ),
        ),

        footer(),
        bg="#080a0f",
        min_height="100vh",
        display="flex",
        flex_direction="column",
    )
