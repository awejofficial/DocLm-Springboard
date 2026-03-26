import reflex as rx

def hero():
    return rx.hstack(
            # LEFT SIDE TEXT
            rx.vstack(
                rx.text("DocuSearch AI", color="gray", font_size="18px"),

                rx.heading(
                    "AI Based Document\nSearch & Knowledge Retrieval",
                    size="9",
                    weight="bold"
                ),

                rx.text(
                    "Chat with your documents using advanced AI",
                    color="gray"
                ),

                rx.button(
                    "Get Started",
                    #quote(),
                    color_scheme="blue",
                    size="3",
                    margin_top="10px"
                ),

                align="start",
                spacing="4",
                width="50%",
            ),

            # RIGHT SIDE IMAGE
            rx.image(
                src="/logo.jpeg",  # put image in assets folder
                width="300px"
            ),
            
            justify="between",
            align="center",
            padding="60px",
        )

# def quote():
#     return rx.box(
#         rx.text("Qoute..."),
#         rx.text("Author...")
#     )