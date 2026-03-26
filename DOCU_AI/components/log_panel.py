import reflex as rx
from DOCU_AI.states.rag_state import ChatState


def log_panel():
    return rx.box(
        # Header
        rx.hstack(
            rx.hstack(
                rx.box(width="8px", height="8px", border_radius="50%", bg="#00ffe7"),
                rx.text(
                    "BACKEND CONSOLE",
                    color="#00ffe7",
                    font_size="11px",
                    font_weight="700",
                    letter_spacing="0.15em",
                    font_family="'Courier New', monospace",
                ),
                spacing="2",
                align="center",
            ),
            rx.spacer(),
            rx.button(
                "REBUILD INDEX",
                on_click=ChatState.rebuild_vectorstore,
                font_size="10px",
                font_weight="700",
                letter_spacing="0.1em",
                color="#0d0f14",
                bg="#00ffe7",
                border_radius="4px",
                padding_x="10px",
                padding_y="4px",
                _hover={"bg": "#00d4c5"},
                cursor="pointer",
            ),
            width="100%",
            align="center",
            margin_bottom="10px",
        ),

        # Log lines
        rx.box(
            rx.cond(
                ChatState.backend_logs == [],
                rx.text(
                    "> waiting for activity...",
                    color="#2d3748",
                    font_size="12px",
                    font_family="'Courier New', monospace",
                ),
                rx.vstack(
                    rx.foreach(
                        ChatState.backend_logs,
                        lambda log: rx.text(
                            "> " + log,
                            color="#68d391",
                            font_size="12px",
                            font_family="'Courier New', monospace",
                            white_space="pre-wrap",
                            word_break="break-word",
                        ),
                    ),
                    spacing="1",
                    align="start",
                    width="100%",
                ),
            ),
            height="220px",
            overflow_y="auto",
            padding="12px",
            bg="#060810",
            border_radius="6px",
            border="1px solid #1a2035",
        ),

        padding="16px",
        bg="#0d0f14",
        border="1px solid #1e2535",
        border_radius="10px",
        width="100%",
    )
