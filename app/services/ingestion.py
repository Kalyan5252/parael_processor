import uuid
from typing import List
from pypdf import PdfReader
from io import BytesIO
import pandas as pd
import docx
from app.core.clients import collection, embed_text

class IngestionService:

    def load_pdf_from_bytes(self, file_content: bytes) -> str:
        reader = PdfReader(BytesIO(file_content))
        pages = []
        for page in reader.pages:
            text = page.extract_text()
            if text:
                pages.append(text)
        return "\n".join(pages)

    def load_docx_from_bytes(self, file_content: bytes) -> str:
        doc = docx.Document(BytesIO(file_content))
        return "\n".join([para.text for para in doc.paragraphs])

    def load_excel_from_bytes(self, file_content: bytes) -> str:
        df = pd.read_excel(BytesIO(file_content))
        return df.to_string()

    def load_csv_from_bytes(self, file_content: bytes) -> str:
        df = pd.read_csv(BytesIO(file_content))
        return df.to_string()

    def chunk_text(self, text, chunk_size=500, overlap=100):
        words = text.split()
        chunks = []
        start = 0
        while start < len(words):
            end = start + chunk_size
            chunk = " ".join(words[start:end])
            chunks.append(chunk)
            start += chunk_size - overlap
        return chunks

    def build_contextual_chunks(self, text: str, user_id: str, file_id: str):
        chunks = self.chunk_text(text)
        contextual_chunks = []

        for i, chunk in enumerate(chunks):
            prev_chunk = chunks[i - 1] if i > 0 else ""
            next_chunk = chunks[i + 1] if i < len(chunks) - 1 else ""

            context_text = f"""
            PREVIOUS CONTEXT:
            {prev_chunk}

            MAIN CHUNK:
            {chunk}

            NEXT CONTEXT:
            {next_chunk}
            """

            contextual_chunks.append({
                "id": str(uuid.uuid4()),
                "text": chunk,
                "context_embedding_text": context_text.strip(),
                "metadata": {
                    "chunk_index": i,
                    "source": "custom_document",
                    "userId": user_id,
                    "fileId": file_id
                }
            })
        return contextual_chunks


    def ingest_text(self, text: str, user_id: str, file_id: str) -> List[str]:
        chunks = self.build_contextual_chunks(text, user_id, file_id)
        
        embeddings = embed_text(
            [c["context_embedding_text"] for c in chunks]
        )

        collection.add(
            ids=[c["id"] for c in chunks],
            documents=[c["text"] for c in chunks],
            embeddings=embeddings,
            metadatas=[c["metadata"] for c in chunks],
        )
        return [c["id"] for c in chunks]
    
    def ingest_file(self, file_content: bytes, filename: str, user_id: str, file_id: str) -> str:
        try:
            filename_lower = filename.lower()
            text = ""
            msg = ""

            if filename_lower.endswith('.pdf'):
                text = self.load_pdf_from_bytes(file_content)
                msg = "PDF Ingested"
            elif filename_lower.endswith('.docx'):
                text = self.load_docx_from_bytes(file_content)
                msg = "Docx Ingested"
            elif filename_lower.endswith(('.xlsx', '.xls')):
                text = self.load_excel_from_bytes(file_content)
                msg = "Excel Ingested"
            elif filename_lower.endswith('.csv'):
                text = self.load_csv_from_bytes(file_content)
                msg = "CSV Ingested"
            elif filename_lower.endswith('.txt'):
                text = file_content.decode('utf-8', errors='ignore')
                msg = "Text File Ingested"
            else:
                return "ingestion not supports file type"

            self.ingest_text(text, user_id, file_id)
            return msg

        except Exception as e:
            print(f"Error ingesting file {filename}: {e}")
            return "unknown problem at ingestion"


ingestion_service = IngestionService()
