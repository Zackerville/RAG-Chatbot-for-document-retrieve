
# RAG Chatbot for Document Retrieval

![Streamlit](https://img.shields.io/badge/Made%20with-Streamlit-red?style=flat&logo=streamlit)
![License](https://img.shields.io/badge/license-MIT-blue.svg)

## 🚀 Overview

This is a **Simple** **Retrieval-Augmented Generation (RAG) Chatbot**  application built with **Streamlit**.
It allows users to ask questions and get accurate answers from uploaded documents using modern LLM Vicuna that also suport Vietnamese.

![1753781622692](https://file+.vscode-resource.vscode-cdn.net/c%3A/Users/Zackerville/Desktop/RAG-Chatbot-for-document-retrieve/image/README/1753781622692.png)

## 🧠 Features

- **Chatbot UI** built with Streamlit for easy interaction
- **RAG pipeline**: combines retrieval (search) and generation (LLM) for better answers
- **Document upload**: users can upload PDF as a knowledge base
- **Supports multiple models** (customizable in code)
- **Easy to deploy and run locally**

## 📂 Project Structure

.
├── RAG.py
├── RAG_chatbot.py
├── RAG_chatbot_v1.1.py
├── README.md
├── .gitignore

## ⚡️ Getting Started

1. **Clone the repo**

```bash
git clone https://github.com/Zackerville/RAG-Chatbot-for-document-retrieve.git
cd RAG-Chatbot-for-document-retrieve
```

2. **Setup Python environment**

   ```
   python3 -m venv venv
   source venv/bin/activate
   pip install --upgrade pip
   pip install streamlit transformers sentence-transformers langchain
   ```
   3. **Run Streamlit app**
      ```
      streamlit run RAG_chatbot_v1.1.py
      ```
