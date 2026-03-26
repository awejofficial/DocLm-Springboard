import reflex as rx
from DOCU_AI.components.navbar import navbar
from DOCU_AI.components.footer import footer


def stat_card(value: str, label: str, sublabel: str):
    return rx.box(
        rx.vstack(
            rx.text(
                value,
                color="#00ffe7",
                font_size="32px",
                font_weight="800",
                line_height="1",
                font_family="'Courier New', monospace",
            ),
            rx.text(label, color="#e2e8f0", font_size="14px", font_weight="600"),
            rx.text(sublabel, color="#2d3748", font_size="11px"),
            spacing="1",
            align="start",
        ),
        padding="24px",
        bg="#0d0f14",
        border="1px solid #1e2535",
        border_radius="12px",
        flex="1",
        min_width="140px",
        _hover={"border_color": "#00ffe730"},
        transition="border-color 0.2s",
    )


def feature_card(icon: str, title: str, desc: str, tag: str):
    return rx.box(
        rx.vstack(
            rx.hstack(
                rx.text(icon, font_size="22px"),
                rx.box(
                    rx.text(
                        tag,
                        color="#00ffe7",
                        font_size="9px",
                        font_weight="700",
                        letter_spacing="0.15em",
                        font_family="'Courier New', monospace",
                    ),
                    padding_x="8px",
                    padding_y="2px",
                    border="1px solid #00ffe733",
                    border_radius="20px",
                    bg="#00ffe708",
                ),
                spacing="3",
                align="center",
            ),
            rx.text(title, color="#e2e8f0", font_size="15px", font_weight="700"),
            rx.text(desc, color="#4a5568", font_size="13px", line_height="1.6"),
            spacing="3",
            align="start",
        ),
        padding="24px",
        bg="#0d0f14",
        border="1px solid #1e2535",
        border_radius="12px",
        flex="1",
        min_width="220px",
        _hover={"border_color": "#00ffe730", "transform": "translateY(-2px)"},
        transition="all 0.2s ease",
    )


def pipeline_step(num: str, title: str, desc: str, is_last: bool = False):
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
            rx.cond(
                not is_last,
                rx.box(width="1px", height="36px", bg="#1e2535", margin_left="13px"),
                rx.box(),
            ),
            spacing="0",
            align="center",
        ),
        rx.vstack(
            rx.text(title, color="#e2e8f0", font_size="14px", font_weight="600"),
            rx.text(desc, color="#4a5568", font_size="12px", line_height="1.5"),
            spacing="1",
            align="start",
            padding_bottom="20px",
        ),
        spacing="4",
        align="start",
        width="100%",
    )


def tech_pill(label: str):
    return rx.box(
        rx.text(
            label,
            color="#a0aec0",
            font_size="11px",
            font_family="'Courier New', monospace",
        ),
        padding_x="10px",
        padding_y="4px",
        bg="#0a0c10",
        border="1px solid #1e2535",
        border_radius="4px",
    )


def home():
    return rx.box(
        navbar(),

        # ── HERO SECTION ──────────────────────────────────────────────
        rx.box(
            rx.vstack(
                # Badge
                rx.box(
                    rx.hstack(
                        rx.box(
                            width="6px", height="6px",
                            bg="#00ffe7", border_radius="50%",
                        ),
                        rx.text(
                            "CORRECTIVE RAG · GROQ · LLAMA 3.1",
                            color="#00ffe7",
                            font_size="10px",
                            font_weight="700",
                            letter_spacing="0.2em",
                            font_family="'Courier New', monospace",
                        ),
                        spacing="2",
                        align="center",
                    ),
                    padding_x="14px",
                    padding_y="6px",
                    border="1px solid #00ffe733",
                    border_radius="20px",
                    bg="#00ffe708",
                ),

                # Headline
                rx.vstack(
                    rx.heading(
                        "Your Documents.",
                        size="9",
                        color="#e2e8f0",
                        font_weight="800",
                        line_height="1.05",
                        text_align="center",
                    ),
                    rx.heading(
                        "Your Answers.",
                        size="9",
                        color="#00ffe7",
                        font_weight="800",
                        line_height="1.05",
                        text_align="center",
                    ),
                    spacing="0",
                    align="center",
                ),

                rx.text(
                    "Upload PDFs or TXTs. Ask questions in plain English. "
                    "DocuAI retrieves the exact context, evaluates relevance, "
                    "rewrites queries if needed — and answers with precision.",
                    color="#4a5568",
                    font_size="16px",
                    text_align="center",
                    max_width="540px",
                    line_height="1.75",
                ),

                # CTAs
                rx.hstack(
                    rx.link(
                        rx.button(
                            rx.hstack(
                                rx.text("Upload Documents"),
                                rx.text("→", font_size="16px"),
                                spacing="2",
                                align="center",
                            ),
                            bg="#00ffe7",
                            color="#0d0f14",
                            font_weight="700",
                            font_size="14px",
                            padding_x="28px",
                            padding_y="13px",
                            border_radius="8px",
                            _hover={"bg": "#00d4c5", "transform": "translateY(-1px)"},
                            transition="all 0.2s",
                            cursor="pointer",
                        ),
                        href="/upload",
                    ),
                    rx.link(
                        rx.button(
                            "Open Chat",
                            bg="transparent",
                            color="#e2e8f0",
                            font_weight="600",
                            font_size="14px",
                            border="1px solid #1e2535",
                            padding_x="28px",
                            padding_y="13px",
                            border_radius="8px",
                            _hover={"border_color": "#00ffe740", "color": "#00ffe7"},
                            transition="all 0.2s",
                            cursor="pointer",
                        ),
                        href="/chat",
                    ),
                    spacing="4",
                ),

                spacing="7",
                align="center",
            ),
            padding_y="110px",
            padding_x="40px",
            display="flex",
            justify_content="center",
            # Subtle radial glow behind hero
            background="radial-gradient(ellipse 70% 40% at 50% 0%, #00ffe710 0%, transparent 70%)",
        ),

        # ── STATS STRIP ───────────────────────────────────────────────
        rx.box(
            rx.hstack(
                stat_card("RAG", "Architecture", "Corrective RAG"),
                stat_card("8B", "LLaMA Model", "Groq inference"),
                stat_card("MMR", "Retrieval", "Chroma vector DB"),
                stat_card("∞", "Documents", "PDF + TXT support"),
                spacing="4",
                wrap="wrap",
            ),
            max_width="860px",
            margin="0 auto",
            padding_x="40px",
            padding_bottom="72px",
        ),

        # ── FEATURES ──────────────────────────────────────────────────
        rx.box(
            rx.vstack(
                rx.text(
                    "CAPABILITIES",
                    color="#00ffe7",
                    font_size="10px",
                    font_weight="700",
                    letter_spacing="0.2em",
                    font_family="'Courier New', monospace",
                    margin_bottom="4px",
                ),
                rx.heading(
                    "Everything you need to query your docs",
                    size="6",
                    color="#e2e8f0",
                    font_weight="700",
                    margin_bottom="32px",
                    text_align="center",
                ),
                rx.flex(
                    feature_card("📤", "Instant Ingestion", "Drop PDFs or TXTs. Files are chunked, embedded, and indexed into ChromaDB automatically.", "UPLOAD"),
                    feature_card("🔍", "Smart Retrieval", "MMR-based search finds semantically relevant chunks — not just keyword matches.", "CHROMA"),
                    feature_card("🧠", "Corrective RAG", "Relevance is evaluated after retrieval. If context is weak, the query is auto-rewritten and re-retrieved.", "ADVANCED"),
                    feature_card("⚡", "Groq · LLaMA 3.1", "Ultra-fast inference via Groq. Answers grounded in your documents, not hallucinations.", "LLM"),
                    gap="4",
                    wrap="wrap",
                ),
                align="center",
                width="100%",
            ),
            max_width="1000px",
            margin="0 auto",
            padding_x="40px",
            padding_bottom="80px",
        ),

        # ── HOW IT WORKS + TECH STACK ─────────────────────────────────
        rx.box(
            rx.hstack(
                # Pipeline steps
                rx.vstack(
                    rx.text(
                        "RAG PIPELINE",
                        color="#00ffe7",
                        font_size="10px",
                        font_weight="700",
                        letter_spacing="0.2em",
                        font_family="'Courier New', monospace",
                        margin_bottom="20px",
                    ),
                    pipeline_step("1", "Document Upload", "PDF and TXT files saved to the documents/ folder"),
                    pipeline_step("2", "Text Extraction", "PyPDFLoader & TextLoader parse raw content"),
                    pipeline_step("3", "Chunking", "RecursiveCharacterTextSplitter (size=300, overlap=30)"),
                    pipeline_step("4", "Embedding", "HuggingFace all-MiniLM-L6-v2 generates vectors"),
                    pipeline_step("5", "Vector Store", "ChromaDB stores and indexes all embeddings"),
                    pipeline_step("6", "Retrieval", "MMR search surfaces top-k relevant chunks"),
                    pipeline_step("7", "Corrective Check", "LLM evaluates relevance — rewrites query if needed"),
                    pipeline_step("8", "Generation", "LLaMA 3.1 on Groq produces the final grounded answer", is_last=True),
                    padding="28px",
                    bg="#0d0f14",
                    border="1px solid #1e2535",
                    border_radius="14px",
                    align="start",
                    flex="1",
                ),

                # Tech stack + CTA
                rx.vstack(
                    rx.vstack(
                        rx.text(
                            "TECH STACK",
                            color="#00ffe7",
                            font_size="10px",
                            font_weight="700",
                            letter_spacing="0.2em",
                            font_family="'Courier New', monospace",
                            margin_bottom="12px",
                        ),
                        rx.flex(
                            tech_pill("Python 3.12"),
                            tech_pill("Reflex"),
                            tech_pill("LangChain"),
                            tech_pill("ChromaDB"),
                            tech_pill("Groq API"),
                            tech_pill("LLaMA 3.1-8B"),
                            tech_pill("HuggingFace"),
                            tech_pill("all-MiniLM-L6-v2"),
                            tech_pill("PyPDF"),
                            wrap="wrap",
                            gap="2",
                        ),
                        padding="24px",
                        bg="#0d0f14",
                        border="1px solid #1e2535",
                        border_radius="12px",
                        align="start",
                        width="100%",
                    ),

                    rx.vstack(
                        rx.text(
                            "FULL-STACK GENAI",
                            color="#00ffe7",
                            font_size="10px",
                            font_weight="700",
                            letter_spacing="0.2em",
                            font_family="'Courier New', monospace",
                        ),
                        rx.heading(
                            "Built end-to-end with modern AI primitives",
                            size="5",
                            color="#e2e8f0",
                            font_weight="700",
                            line_height="1.3",
                        ),
                        rx.text(
                            "DocuAI demonstrates a complete production GenAI stack — "
                            "from file ingestion to vector search to LLM generation — "
                            "all wired together with Reflex state management.",
                            color="#4a5568",
                            font_size="13px",
                            line_height="1.7",
                        ),
                        rx.hstack(
                            rx.link(
                                rx.button(
                                    "Read the docs →",
                                    bg="transparent",
                                    color="#00ffe7",
                                    font_size="13px",
                                    font_weight="600",
                                    border="1px solid #00ffe733",
                                    padding_x="16px",
                                    padding_y="8px",
                                    border_radius="6px",
                                    _hover={"bg": "#00ffe710"},
                                    cursor="pointer",
                                ),
                                href="/about",
                            ),
                        ),
                        padding="24px",
                        bg="#0d0f14",
                        border="1px solid #1e2535",
                        border_radius="12px",
                        align="start",
                        spacing="4",
                        width="100%",
                    ),

                    spacing="4",
                    width="320px",
                    flex_shrink="0",
                ),

                spacing="6",
                align="start",
                wrap="wrap",
                width="100%",
            ),
            max_width="1000px",
            margin="0 auto",
            padding_x="40px",
            padding_bottom="100px",
        ),

        # ── BOTTOM CTA ────────────────────────────────────────────────
        rx.box(
            rx.vstack(
                rx.text(
                    "GET STARTED",
                    color="#00ffe7",
                    font_size="10px",
                    font_weight="700",
                    letter_spacing="0.2em",
                    font_family="'Courier New', monospace",
                ),
                rx.heading(
                    "Ready to query your documents?",
                    size="7",
                    color="#e2e8f0",
                    font_weight="800",
                    text_align="center",
                ),
                rx.text(
                    "Upload a PDF or TXT file and start asking questions in seconds.",
                    color="#4a5568",
                    font_size="14px",
                    text_align="center",
                ),
                rx.link(
                    rx.button(
                        "Upload your first document →",
                        bg="#00ffe7",
                        color="#0d0f14",
                        font_weight="700",
                        font_size="14px",
                        padding_x="32px",
                        padding_y="14px",
                        border_radius="8px",
                        _hover={"bg": "#00d4c5", "transform": "translateY(-1px)"},
                        transition="all 0.2s",
                        cursor="pointer",
                    ),
                    href="/upload",
                ),
                spacing="5",
                align="center",
            ),
            padding="72px 40px",
            border_top="1px solid #1e2535",
            display="flex",
            justify_content="center",
            background="radial-gradient(ellipse 50% 80% at 50% 100%, #00ffe708 0%, transparent 70%)",
        ),

        footer(),
        bg="#080a0f",
        min_height="100vh",
        display="flex",
        flex_direction="column",
    )
