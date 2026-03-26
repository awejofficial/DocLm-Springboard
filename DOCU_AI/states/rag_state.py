import reflex as rx
from typing import List
from pydantic import BaseModel
import os
import io

from DOCU_AI.backend.rag import get_answer, set_status_callback, build_vectorstore


class ChatItem(BaseModel):
    question: str
    answer: str
    sources: str


class ChatState(rx.State):
    question: str = ""
    history: List[ChatItem] = []
    is_loading: bool = False
    backend_logs: List[str] = []
    uploaded_files: List[str] = []

    def _push_log(self, msg: str):
        self.backend_logs = self.backend_logs + [msg]

    def set_question(self, value: str):
        self.question = value

    def ask_question(self):
        if not self.question.strip():
            return

        self.is_loading = True
        self.backend_logs = []
        set_status_callback(self._push_log)

        answer, sources = get_answer(self.question)

        if isinstance(sources, list):
            sources_str = ", ".join([os.path.basename(s) for s in sources])
        else:
            sources_str = os.path.basename(sources) if sources else ""

        new_item = ChatItem(
            question=self.question,
            answer=answer,
            sources=sources_str
        )
        self.history = self.history + [new_item]
        self.question = ""
        self.is_loading = False

    def clear_history(self):
        self.history = []

    def download_chat(self):
        if not self.history:
            return
        lines = []
        for i, item in enumerate(self.history, 1):
            lines.append(f"--- Q&A #{i} ---")
            lines.append(f"Q: {item.question}")
            lines.append(f"A: {item.answer}")
            if item.sources:
                lines.append(f"Sources: {item.sources}")
            lines.append("")
        content = "\n".join(lines)
        return rx.download(data=content.encode("utf-8"), filename="chat_history.txt")

    def rebuild_vectorstore(self):
        self.backend_logs = []
        set_status_callback(self._push_log)
        build_vectorstore()

    def load_uploaded_files(self):
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        doc_dir = os.path.join(base_dir, "documents")
        os.makedirs(doc_dir, exist_ok=True)
        try:
            files = os.listdir(doc_dir)
            self.uploaded_files = [f for f in files if f.endswith((".pdf", ".txt"))]
        except Exception:
            self.uploaded_files = []
