import reflex as rx
from DOCU_AI.components.navbar import navbar
from DOCU_AI.components.footer import footer


def tech_badge(label: str, color: str = "#1e2535"):
    return rx.box(
        rx.text(label, color="#a0aec0", font_size="12px", font_family="'Courier New', monospace"),
        padding_x="12px",
        padding_y="5px",
        bg=color,
        border="1px solid #2d3748",
        border_radius="4px",
    )


def arch_step(num: str, title: str, desc: str):
    return rx.hstack(
        rx.vstack(
            rx.box(
                rx.text(num, color="#0d0f14", font_size="11px", font_weight="800"),
                bg="#00ffe7",
                width="28px",
                height="28px",
                border_radius="50%",
                display="flex",
                align_items="center",
                justify_content="center",
                flex_shrink="0",
            ),
            rx.box(
                width="1px",
                height="32px",
                bg="#1e2535",
                margin_left="13px",
            ),
            spacing="0",
            align="center",
        ),
        rx.vstack(
            rx.text(title, color="#e2e8f0", font_size="14px", font_weight="600"),
            rx.text(desc, color="#4a5568", font_size="12px", line_height="1.5"),
            spacing="1",
            align="start",
            padding_bottom="16px",
        ),
        spacing="4",
        align="start",
        width="100%",
    )


def about():
    return rx.box(
        navbar(),

        rx.box(
            rx.vstack(

                # Hero
                rx.vstack(
                    rx.text(
                        "ABOUT THE SYSTEM",
                        color="#00ffe7",
                        font_size="11px",
                        font_weight="700",
                        letter_spacing="0.2em",
                        font_family="'Courier New', monospace",
                    ),
                    rx.heading("DocuAI", size="8", color="#e2e8f0", font_weight="800"),
                    rx.text(
                        "An AI-powered document intelligence system using Corrective RAG "
                        "to provide precise, context-grounded answers from your documents.",
                        color="#4a5568",
                        font_size="15px",
                        max_width="600px",
                        line_height="1.7",
                    ),
                    spacing="3",
                    align="start",
                    margin_bottom="48px",
                ),

                rx.hstack(
                    # Architecture
                    rx.vstack(
                        rx.text(
                            "SYSTEM ARCHITECTURE",
                            color="#00ffe7",
                            font_size="10px",
                            font_weight="700",
                            letter_spacing="0.15em",
                            font_family="'Courier New', monospace",
                            margin_bottom="16px",
                        ),
                        arch_step("1", "Document Upload", "PDF and TXT files saved to the documents/ folder"),
                        arch_step("2", "Text Extraction", "PyPDFLoader and TextLoader parse raw content"),
                        arch_step("3", "Chunking", "RecursiveCharacterTextSplitter (size=300, overlap=30)"),
                        arch_step("4", "Embedding", "HuggingFace all-MiniLM-L6-v2 generates vectors"),
                        arch_step("5", "Vector Store", "ChromaDB stores and indexes all embeddings"),
                        arch_step("6", "Retrieval", "MMR search finds top-k relevant chunks"),
                        arch_step("7", "Corrective RAG", "Relevance check + query rewriting if needed"),
                        arch_step("8", "Generation", "LLaMA 3.1-8B on Groq produces the final answer"),
                        padding="24px",
                        bg="#0d0f14",
                        border="1px solid #1e2535",
                        border_radius="12px",
                        flex="1",
                        align="start",
                    ),

                    # Tech stack
                    rx.vstack(
                        rx.vstack(
                            rx.text(
                                "TECH STACK",
                                color="#00ffe7",
                                font_size="10px",
                                font_weight="700",
                                letter_spacing="0.15em",
                                font_family="'Courier New', monospace",
                                margin_bottom="12px",
                            ),
                            rx.flex(
                                tech_badge("Python 3.12"),
                                tech_badge("Reflex 0.8"),
                                tech_badge("LangChain"),
                                tech_badge("Groq API"),
                                tech_badge("LLaMA 3.1-8B"),
                                tech_badge("ChromaDB"),
                                tech_badge("HuggingFace"),
                                tech_badge("all-MiniLM-L6-v2"),
                                tech_badge("PyPDF"),
                                tech_badge("ReportLab"),
                                wrap="wrap",
                                gap="2",
                            ),
                            padding="20px",
                            bg="#0d0f14",
                            border="1px solid #1e2535",
                            border_radius="12px",
                            width="100%",
                            align="start",
                        ),

                        rx.vstack(
                            rx.text(
                                "WHAT IS CORRECTIVE RAG?",
                                color="#00ffe7",
                                font_size="10px",
                                font_weight="700",
                                letter_spacing="0.15em",
                                font_family="'Courier New', monospace",
                                margin_bottom="12px",
                            ),
                            rx.text(
                                "Standard RAG retrieves chunks and passes them directly to the LLM. "
                                "Corrective RAG adds a relevance evaluation step — if the retrieved "
                                "context is not relevant enough, the query is automatically rewritten "
                                "and retrieval is repeated before generating the final answer.",
                                color="#4a5568",
                                font_size="13px",
                                line_height="1.7",
                            ),
                            padding="20px",
                            bg="#0d0f14",
                            border="1px solid #1e2535",
                            border_radius="12px",
                            width="100%",
                            align="start",
                        ),

                        rx.vstack(
                            rx.text(
                                "USE CASES",
                                color="#00ffe7",
                                font_size="10px",
                                font_weight="700",
                                letter_spacing="0.15em",
                                font_family="'Courier New', monospace",
                                margin_bottom="12px",
                            ),
                            rx.text("📚  Academic research & paper analysis", color="#4a5568", font_size="13px"),
                            rx.text("📋  Resume and HR document review", color="#4a5568", font_size="13px"),
                            rx.text("📊  Business report summarization", color="#4a5568", font_size="13px"),
                            rx.text("⚖️  Legal document Q&A", color="#4a5568", font_size="13px"),
                            rx.text("🔬  Technical documentation search", color="#4a5568", font_size="13px"),
                            padding="20px",
                            bg="#0d0f14",
                            border="1px solid #1e2535",
                            border_radius="12px",
                            spacing="2",
                            align="start",
                            width="100%",
                        ),

                        spacing="4",
                        width="340px",
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
