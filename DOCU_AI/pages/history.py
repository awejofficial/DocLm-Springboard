import reflex as rx
from DOCU_AI.components.navbar import navbar
from DOCU_AI.components.footer import footer
from DOCU_AI.states.rag_state import ChatState


def history_card(item):
    return rx.box(
        rx.vstack(
            # Question
            rx.hstack(
                rx.box(
                    rx.text("Q", color="#0d0f14", font_size="10px", font_weight="800"),
                    bg="#00ffe7",
                    width="20px",
                    height="20px",
                    border_radius="50%",
                    display="flex",
                    align_items="center",
                    justify_content="center",
                    flex_shrink="0",
                ),
                rx.text(
                    item.question,
                    color="#e2e8f0",
                    font_size="14px",
                    font_weight="600",
                    line_height="1.5",
                ),
                spacing="3",
                align="start",
                width="100%",
            ),
            # Answer
            rx.box(
                rx.text(
                    item.answer,
                    color="#a0aec0",
                    font_size="13px",
                    line_height="1.7",
                    white_space="pre-wrap",
                ),
                padding="12px 16px",
                bg="#0a0c10",
                border_radius="8px",
                border_left="3px solid #00ffe740",
                width="100%",
            ),
            # Sources
            rx.hstack(
                rx.text(
                    "📂 " + item.sources,
                    color="#2d5a7a",
                    font_size="11px",
                    font_family="'Courier New', monospace",
                ),
                spacing="2",
            ),
            spacing="3",
            align="start",
            width="100%",
        ),
        padding="20px 24px",
        bg="#0d0f14",
        border="1px solid #1e2535",
        border_radius="12px",
        width="100%",
        _hover={"border_color": "#00ffe730"},
        transition="border-color 0.2s",
    )


def history():
    return rx.box(
        navbar(),

        rx.box(
            rx.vstack(
                # Header
                rx.hstack(
                    rx.vstack(
                        rx.text(
                            "CONVERSATION LOG",
                            color="#00ffe7",
                            font_size="11px",
                            font_weight="700",
                            letter_spacing="0.2em",
                            font_family="'Courier New', monospace",
                        ),
                        rx.heading("Chat History", size="7", color="#e2e8f0", font_weight="800"),
                        spacing="1",
                        align="start",
                    ),
                    rx.spacer(),
                    rx.hstack(
                        rx.button(
                            "⬇ Download PDF",
                            on_click=ChatState.download_chat,
                            bg="#00ffe7",
                            color="#0d0f14",
                            font_weight="700",
                            font_size="13px",
                            padding_x="18px",
                            padding_y="9px",
                            border_radius="8px",
                            _hover={"bg": "#00d4c5"},
                            cursor="pointer",
                        ),
                        rx.button(
                            "🗑 Clear All",
                            on_click=ChatState.clear_history,
                            bg="transparent",
                            color="#fc8181",
                            font_size="13px",
                            border="1px solid #4a1515",
                            padding_x="18px",
                            padding_y="9px",
                            border_radius="8px",
                            _hover={"bg": "#1f0d0d", "border_color": "#fc8181"},
                            cursor="pointer",
                        ),
                        spacing="3",
                    ),
                    width="100%",
                    align="center",
                    margin_bottom="32px",
                ),

                # Empty state
                rx.cond(
                    ChatState.history == [],
                    rx.box(
                        rx.vstack(
                            rx.text("📭", font_size="48px"),
                            rx.text(
                                "No chat history yet",
                                color="#2d3748",
                                font_size="16px",
                                font_weight="600",
                            ),
                            rx.link(
                                rx.text("Start a conversation →", color="#00ffe7", font_size="14px"),
                                href="/chat",
                            ),
                            spacing="3",
                            align="center",
                        ),
                        padding="80px",
                        display="flex",
                        justify_content="center",
                    ),
                    rx.vstack(
                        rx.foreach(ChatState.history, history_card),
                        spacing="4",
                        width="100%",
                    ),
                ),

                max_width="860px",
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
