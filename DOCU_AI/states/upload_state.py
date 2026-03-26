import reflex as rx
import os
import shutil
from typing import List
from DOCU_AI.backend.rag import build_vectorstore, set_status_callback
from DOCU_AI.states.rag_state import ChatState

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UPLOAD_DIR = os.path.join(BASE_DIR, "documents")
CHROMA_PATH = os.path.join(BASE_DIR, "chroma_db")
os.makedirs(UPLOAD_DIR, exist_ok=True)


def _clear_chroma():
    """Delete the persisted vector store so it rebuilds fresh on next query."""
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)
    import DOCU_AI.backend.rag as rag_module
    rag_module.vectorstore = None
    rag_module.retriever = None


class UploadState(rx.State):
    uploading: bool = False
    upload_done: bool = False
    uploaded_names: List[str] = []
    error_msg: str = ""

    async def handle_upload(self, files: list[rx.UploadFile]):
        self.uploading = True
        self.upload_done = False
        self.error_msg = ""
        names = []
        try:
            for file in files:
                save_path = os.path.join(UPLOAD_DIR, file.filename)
                content = await file.read()
                with open(save_path, "wb") as f:
                    f.write(content)
                names.append(file.filename)
            self.uploaded_names = names
            self.upload_done = True
            _clear_chroma()
        except Exception as e:
            self.error_msg = str(e)
        self.uploading = False
        yield ChatState.load_uploaded_files()

    def delete_file(self, filename: str):
        """Delete a single document and invalidate the vector store."""
        file_path = os.path.join(UPLOAD_DIR, filename)
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
            _clear_chroma()
        except Exception as e:
            self.error_msg = str(e)
        yield ChatState.load_uploaded_files()

    def clear_all_files(self):
        """Delete all documents and the vector store."""
        try:
            for f in os.listdir(UPLOAD_DIR):
                if f.endswith((".pdf", ".txt")):
                    os.remove(os.path.join(UPLOAD_DIR, f))
            _clear_chroma()
            self.upload_done = False
            self.uploaded_names = []
        except Exception as e:
            self.error_msg = str(e)
        yield ChatState.load_uploaded_files()
