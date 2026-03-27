# Scientific Research Assistant Bot

A lightweight RAG assistant for researching scientific papers, powered by **OpenAI GPT & Embeddings** and **Gradio**.

## 🚀 Quick Start

**1. Add files for your research**
in pdf format to data folder
```bash
mkdir ./data
```
```bash
cp /full/path/to/article1.pdf ./data/
```

**2. Install Dependencies**

```bash
pip install openai gradio langchain-chroma langchain-openai langchain-community pypdf langchain-text-splitters
```

**3. Set OpenAI API Key**
You need a [OpenAI Platform](https://platform.openai.com/api-keys/) key.

  * **Mac/Linux:** `export OPENAI_API_KEY="your_openai_api_key"`
  * **Windows:** `$env:OPENAI_API_KEY="your_openai_api_key"`

**4. Run db_uploader**
and wait until it completes ChromaDB setup.
```bash
python db_uploader.py
```

**5. Run the App**

```bash
python main.py
```

Open the URL displayed in your terminal (usually `http://127.0.0.1:7860`).

## 💡 What it does

  * **Expert Assistance:** Answers questions about AWS architecture (EC2, Lambda, S3, etc.), troubleshooting, and CI/CD.
  * **Contextual:** Remembers your chat history for debugging sessions.
  * **Focused:** It acts strictly as an AWS support agent and will decline unrelated topics (like cooking or general news).

## 🔧 Requirements

  * Python 3.9+
  * OpenAI GPT models and Embedding models
