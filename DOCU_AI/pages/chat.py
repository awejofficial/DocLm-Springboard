import reflex as rx
from DOCU_AI.components.navbar import navbar
from DOCU_AI.components.footer import footer
from DOCU_AI.components.log_panel import log_panel
from DOCU_AI.states.rag_state import ChatState


def user_bubble(item):
    return rx.hstack(
        rx.spacer(),
        rx.box(
            rx.text(
                item.question,
                color="#0d0f14",
                font_size="14px",
                line_height="1.6",
            ),
            bg="#00ffe7",
            padding="12px 16px",
            border_radius="14px 14px 4px 14px",
            max_width="65%",
        ),
    )


def ai_bubble(item):
    return rx.vstack(
        rx.hstack(
            rx.box(
                rx.text(
                    item.answer,
                    color="#e2e8f0",
                    font_size="14px",
                    line_height="1.7",
                    white_space="pre-wrap",
                ),
                bg="#0d1828",
                padding="12px 16px",
                border_radius="14px 14px 14px 4px",
                max_width="65%",
                border="1px solid #1e3a5a",
            ),
            rx.spacer(),
        ),
        rx.hstack(
            rx.text(
                "📂 " + item.sources,
                font_size="11px",
                color="#2d5a7a",
                font_family="'Courier New', monospace",
            ),
            rx.spacer(),
            padding_left="4px",
        ),
        spacing="1",
        width="100%",
        align="start",
    )


def chat_item(item):
    return rx.vstack(
        user_bubble(item),
        ai_bubble(item),
        spacing="3",
        width="100%",
    )


def chat():
    return rx.box(
        navbar(),

        rx.box(
            rx.hstack(

                # Main chat area
                rx.vstack(
                    # Header
                    rx.hstack(
                        rx.vstack(
                            rx.text(
                                "DOCUMENT CHAT",
                                color="#00ffe7",
                                font_size="11px",
                                font_weight="700",
                                letter_spacing="0.2em",
                                font_family="'Courier New', monospace",
                            ),
                            rx.heading("Chat with your Docs", size="6", color="#e2e8f0", font_weight="800"),
                            spacing="1",
                            align="start",
                        ),
                        rx.spacer(),
                        rx.button(
                            "Clear Chat",
                            on_click=ChatState.clear_history,
                            bg="transparent",
                            color="#4a5568",
                            font_size="12px",
                            border="1px solid #1e2535",
                            padding_x="14px",
                            padding_y="7px",
                            border_radius="6px",
                            _hover={"border_color": "#fc8181", "color": "#fc8181"},
                            transition="all 0.2s",
                            cursor="pointer",
                        ),
                        width="100%",
                        align="center",
                        margin_bottom="16px",
                    ),

                    # Messages area
                    rx.box(
                        rx.cond(
                            ChatState.history == [],
                            rx.vstack(
                                rx.text("💬", font_size="40px"),
                                rx.text(
                                    "No messages yet",
                                    color="#2d3748",
                                    font_size="15px",
                                    font_weight="600",
                                ),
                                rx.text(
                                    "Ask a question about your uploaded documents",
                                    color="#1e2535",
                                    font_size="13px",
                                ),
                                spacing="2",
                                align="center",
                                padding_top="60px",
                            ),
                            rx.vstack(
                                rx.foreach(ChatState.history, chat_item),
                                spacing="5",
                                width="100%",
                                padding="8px",
                            ),
                        ),
                        flex="1",
                        overflow_y="auto",
                        padding="16px",
                        bg="#0a0c10",
                        border="1px solid #1e2535",
                        border_radius="12px",
                        min_height="380px",
                    ),

                    # Loading indicator
                    rx.cond(
                        ChatState.is_loading,
                        rx.hstack(
                            rx.spinner(color="#00ffe7", size="2"),
                            rx.text(
                                "Thinking...",
                                color="#00ffe7",
                                font_size="13px",
                                font_family="'Courier New', monospace",
                            ),
                            spacing="2",
                            align="center",
                            padding_y="8px",
                        ),
                    ),

                    # Input area
                    rx.hstack(
                        rx.input(
                            placeholder="Ask something about your documents...",
                            value=ChatState.question,
                            on_change=ChatState.set_question,
                            on_key_down=lambda e: rx.cond(
                                e == "Enter",
                                ChatState.ask_question(),
                                rx.noop(),
                            ),
                            bg="#0d0f14",
                            color="#e2e8f0",
                            border="1px solid #1e2535",
                            border_radius="8px",
                            padding_x="16px",
                            padding_y="10px",
                            font_size="14px",
                            _placeholder={"color": "#2d3748"},
                            _focus={"border_color": "#00ffe7", "outline": "none"},
                            flex="1",
                        ),
                        rx.button(
                            "Ask →",
                            on_click=ChatState.ask_question,
                            bg="#00ffe7",
                            color="#0d0f14",
                            font_weight="700",
                            font_size="14px",
                            padding_x="20px",
                            padding_y="10px",
                            border_radius="8px",
                            _hover={"bg": "#00d4c5"},
                            transition="all 0.2s",
                            cursor="pointer",
                            disabled=ChatState.is_loading,
                        ),
                        spacing="3",
                        width="100%",
                    ),

                    spacing="4",
                    flex="1",
                    min_width="0",
                ),

                # Sidebar: log panel
                rx.vstack(
                    log_panel(),
                    # Quick tips
                    rx.box(
                        rx.vstack(
                            rx.text(
                                "TIPS",
                                color="#00ffe7",
                                font_size="10px",
                                font_weight="700",
                                letter_spacing="0.15em",
                                font_family="'Courier New', monospace",
                                margin_bottom="8px",
                            ),
                            rx.text("• Ask specific questions about the content", color="#2d3748", font_size="12px"),
                            rx.text("• \"Summarize this document\"", color="#2d3748", font_size="12px"),
                            rx.text("• \"What does it say about X?\"", color="#2d3748", font_size="12px"),
                            rx.text("• Upload docs first at /upload", color="#2d3748", font_size="12px"),
                            spacing="2",
                            align="start",
                        ),
                        padding="16px",
                        bg="#0d0f14",
                        border="1px solid #1e2535",
                        border_radius="10px",
                        width="100%",
                    ),
                    spacing="4",
                    width="340px",
                    flex_shrink="0",
                ),

                spacing="6",
                align="start",
                width="100%",
            ),
            max_width="1200px",
            margin="0 auto",
            padding="40px",
            width="100%",
        ),

        footer(),
        bg="#080a0f",
        min_height="100vh",
        display="flex",
        flex_direction="column",
    )
