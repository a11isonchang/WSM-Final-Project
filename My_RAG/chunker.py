from langchain_text_splitters import RecursiveCharacterTextSplitter


def get_text_splitter(language: str, chunk_size: int, chunk_overlap: int):
    """
    Language-aware RecursiveCharacterTextSplitter
    """

    # ===== 中文（zh）=====
    if language.startswith("zh"):
        return RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=[
                r"\n{2,}",        # ✅ 多空行（段落）
                "\n",             # 單行
                "。", "！", "？",  # 中文句尾
                "；",             # 分號
                "，",             # ✅ 最後才用逗號
                ".", "!", "?",    # fallback
                ""
            ],
            is_separator_regex=True
        )

    # ===== 英文（句子邊界 + word-based）=====
    return RecursiveCharacterTextSplitter(
        chunk_size=120,
        chunk_overlap=20,
        length_function=lambda x: len(x.split()),
        separators=[
            r"\n\n+",                                   # 段落
            r"\n",
            r"(?<=[.!?])\s+(?=[A-Z])",                  # 英文句子邊界
            r"\s+",
            ""
        ],
        is_separator_regex=True
    )


def chunk_documents(docs, language, chunk_size=384, chunk_overlap=64):
    """
    Chunk documents by language
    """
    chunks = []

    text_splitter = get_text_splitter(
        language=language,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    for doc in docs:
        if (
            isinstance(doc, dict)
            and isinstance(doc.get("content"), str)
            and doc.get("language") == language
        ):
            text = doc["content"]

            base_metadata = {k: v for k, v in doc.items() if k != "content"}

            split_docs = text_splitter.create_documents(
                texts=[text],
                metadatas=[base_metadata]
            )

            for i, split_doc in enumerate(split_docs):
                split_doc.metadata["chunk_index"] = i
                chunks.append({
                    "page_content": split_doc.page_content,
                    "metadata": split_doc.metadata
                })

    return chunks