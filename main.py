from openai import OpenAI
import gradio as gr

import db_setup

client = OpenAI()
chats = {}

model = "gpt-5.4"
system_instruction = """
    You are a scientific research assistant. Your task is to answer questions using the scientific articles provided in the context.

    ## Core Rules

    1. **Source citation is mandatory.** Every claim, fact, or conclusion must be followed by an inline citation in the format: [Author et al., Year, p. X] or [Article Title, Section Name].

    2. **Distinguish between sources and your own knowledge.**
    - Information from the provided articles → cite with [Author, Year] notation
    - General background knowledge not covered in the articles → explicitly mark as [General knowledge] and use sparingly

    3. **If the answer is not in the provided articles**, say so directly:
    > "The provided articles do not contain sufficient information to answer this question. Based on general knowledge: ..."

    4. **Do not hallucinate citations.** Never fabricate author names, page numbers, or sections. If you are unsure of an exact location, write [Author, Year — approximate location].

    5. **Quoting vs. paraphrasing.** Prefer paraphrasing with citation. Use direct quotes only when the exact wording matters — always mark them with quotation marks and cite the source.

    ## Response Format

    - Start with a direct answer to the question
    - Support each claim with a citation immediately after the sentence
    - If multiple sources agree or contradict each other — note this explicitly
    - End with a **Sources Used** section listing all cited articles

    ## Example

    **Question:** What methods are used for skill gap analysis?

    **Answer:**
    Skill gap analysis typically involves comparing a user's current skill profile against the requirements of a target occupation [Smith et al., 2023, Section 3.2]. One common approach uses vector similarity between skill embeddings derived from job postings and user profiles [Johnson & Lee, 2022, p. 47]. Some systems additionally leverage taxonomy-based matching, such as alignment with the ESCO framework [Brown et al., 2021, p. 12].

    **Sources Used:**
    - Smith et al. (2023). *Career Recommendation Using NLP*. Section 3.2
    - Johnson & Lee (2022). *Skill Embedding Models*. p. 47
    - Brown et al. (2021). *ESCO-Based Job Matching*. p. 12
"""

vector_store = db_setup.get_db()
retriever = vector_store.as_retriever(search_type="similarity_score_threshold",
                                 search_kwargs={"score_threshold": .5,
                                                "k": 10})

def get_openai_response(question, history, request: gr.Request):
    chat_id = request.session_hash
    chat_is_not_created = chat_id not in chats
    chat_was_cleared = chat_id in chats and len(history) == 0 and len(chats[chat_id]) != 0
    if chat_is_not_created or chat_was_cleared:
        chats[chat_id] = []

    docs = retriever.invoke(question)
    docs_input_parts = []
    for doc in docs:
        title = doc.metadata.get('title', "file: " + doc.metadata.get('source', 'unknown'))
        page = doc.metadata.get('page_label', doc.metadata.get('page', 'N/A'))
        docs_input_parts.append(f"[source: {title}; page: {page}]\n{doc.page_content}")
    docs_joined = '\n\n'.join(docs_input_parts)
    full_question = f"User question:\n{question}\n\n\nRetrieved context:\n{docs_joined}"

    chats[chat_id].append({"role": "user", "content": full_question})

    messages = [{"role": "system", "content": system_instruction}] + chats[chat_id]

    stream = client.chat.completions.create(
        model=model,
        messages=messages,
        stream=True,
    )

    parts = []
    for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            parts.append(delta)
            yield "".join(parts)

    assistant_text = "".join(parts)
    chats[chat_id].append({"role": "assistant", "content": assistant_text})

    if len(docs) > 0:
        parts.append("\n\n\n**Sources:**")
        yield "".join(parts)

    for doc in docs:
        title = doc.metadata.get('title', 'Unknown Title')
        source = "file: " + doc.metadata.get('source', 'unknown')
        page = doc.metadata.get('page_label', doc.metadata.get('page', 'N/A'))
        
        content = doc.page_content
        if len(content) > 350:
            content = f"{content[:150]} ... {content[-150:]}"
        content = content.replace('\n', '')
        
        parts.append(f"\n**[source: {title} ({source}); page: {page}]**\n{content}")
        yield "".join(parts)

gr.ChatInterface(
    get_openai_response,
    chatbot=gr.Chatbot(height=800),
    textbox=gr.Textbox(placeholder="Ask me any question about your research", container=False),
    title="Scientific Research Assistant",
    description="Ask me any question about your research",
    examples=["Hello", "What is ESCO?", "What is machine learning?"]
).launch(theme="ocean")
