# 📚 Sistema de RAG Local com DeepSeek + Ollama

Este projeto implementa um sistema completo de **Retrieval-Augmented Generation (RAG)** executado localmente utilizando:

- DeepSeek (via Ollama)
- LangChain
- FAISS (Vector Store)
- HuggingFace Embeddings
- Streamlit (Interface Web)

O sistema permite que o usuário faça upload de um PDF e realize perguntas baseadas exclusivamente no conteúdo do documento.

---

## 🧠 Arquitetura do Sistema

1. Upload do PDF
2. Extração de texto com PDFPlumber
3. Divisão semântica com SemanticChunker
4. Geração de embeddings com HuggingFace
5. Armazenamento vetorial com FAISS
6. Recuperação por similaridade (Top-K)
7. Geração de resposta com DeepSeek (via Ollama)

Fluxo:

Usuário → PDF → Chunking → Embeddings → FAISS → Retriever → DeepSeek → Resposta

---

## ⚙️ Tecnologias Utilizadas

- Python
- Streamlit
- LangChain
- FAISS
- HuggingFace Embeddings
- Ollama
- DeepSeek-r1 (1.5b)

---

## 💻 Como Executar o Projeto

### 1️⃣ Instalar Dependências

```bash
pip install streamlit langchain langchain-community langchain-huggingface \
langchain-experimental langchain-ollama faiss-cpu pdfplumber
