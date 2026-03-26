import reflex as rx


def navbar():
    return rx.box(
        rx.hstack(
            # Logo
            rx.hstack(
                rx.box(
                    rx.text("⬡", font_size="22px", color="#00ffe7"),
                    margin_right="8px",
                ),
                rx.text(
                    "Doc",
                    font_size="20px",
                    font_weight="800",
                    color="#ffffff",
                    letter_spacing="0.12em",
                ),
                rx.text(
                    "LM",
                    font_size="20px",
                    font_weight="800",
                    color="#00ffe7",
                    letter_spacing="0.12em",
                ),
                align="center",
                spacing="0",
            ),

            rx.spacer(),

            # Nav links
            rx.hstack(
                rx.link(
                    "Home",
                    href="/",
                    color="#a0aec0",
                    font_size="13px",
                    font_weight="500",
                    letter_spacing="0.08em",
                    text_transform="uppercase",
                    _hover={"color": "#00ffe7"},
                    transition="color 0.2s",
                ),
                rx.link(
                    "Upload",
                    href="/upload",
                    color="#a0aec0",
                    font_size="13px",
                    font_weight="500",
                    letter_spacing="0.08em",
                    text_transform="uppercase",
                    _hover={"color": "#00ffe7"},
                    transition="color 0.2s",
                ),
                rx.link(
                    "Chat",
                    href="/chat",
                    color="#a0aec0",
                    font_size="13px",
                    font_weight="500",
                    letter_spacing="0.08em",
                    text_transform="uppercase",
                    _hover={"color": "#00ffe7"},
                    transition="color 0.2s",
                ),
                rx.link(
                    "History",
                    href="/history",
                    color="#a0aec0",
                    font_size="13px",
                    font_weight="500",
                    letter_spacing="0.08em",
                    text_transform="uppercase",
                    _hover={"color": "#00ffe7"},
                    transition="color 0.2s",
                ),
                rx.link(
                    "About",
                    href="/about",
                    color="#a0aec0",
                    font_size="13px",
                    font_weight="500",
                    letter_spacing="0.08em",
                    text_transform="uppercase",
                    _hover={"color": "#00ffe7"},
                    transition="color 0.2s",
                ),
                spacing="7",
            ),

            width="100%",
            padding_x="32px",
            padding_y="16px",
            align="center",
        ),
        width="100%",
        bg="#0d0f14",
        border_bottom="1px solid #1e2535",
        position="sticky",
        top="0",
        z_index="100",
        backdrop_filter="blur(12px)",
    )
