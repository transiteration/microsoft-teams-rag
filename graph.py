import os
import html
import logging
from dotenv import load_dotenv
from pydantic import BaseModel
from typing import List, TypedDict

from google import genai
from google.genai.types import EmbedContentConfig
from qdrant_client import QdrantClient

from langgraph.graph import StateGraph, END
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.output_parsers.string import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

GOOGLE_CLOUD_PROJECT = os.getenv("GOOGLE_CLOUD_PROJECT")
GOOGLE_CLOUD_LOCATION = os.getenv("GOOGLE_CLOUD_LOCATION")
GEMINI_MODEL = os.getenv("GEMINI_MODEL")
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_COLLECTION = os.getenv("COLLECTION_NAME")
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME")
RETRIEVER_K = int(os.getenv("RETRIEVER_K"))
MAX_TRANSFORM_ATTEMPTS = int(os.getenv("MAX_TRANSFORM_ATTEMPTS"))


class GraphState(TypedDict):
    question: str
    generation: str
    documents: List[Document]
    query_type: str
    transform_attempts: int
    chat_history: List[BaseMessage]


class GraderOutput(BaseModel):
    score: str


class QueryTypeOutput(BaseModel):
    query_type: str


llm = ChatGoogleGenerativeAI(model=GEMINI_MODEL, temperature=0)

logging.info("Initializing Google GenAI client for embeddings...")
if not GOOGLE_CLOUD_PROJECT:
    raise ValueError("GOOGLE_CLOUD_PROJECT environment variable not set.")
genai_client = genai.Client(
    project=GOOGLE_CLOUD_PROJECT,
    location=GOOGLE_CLOUD_LOCATION,
    vertexai=True,
)

logging.info("Initializing native Qdrant client...")
qdrant_client = QdrantClient(url=QDRANT_URL)


def format_chat_history(chat_history: List[BaseMessage]) -> str:
    """Formats chat history into a readable string for the model."""
    if not chat_history:
        return "No history yet."
    return "\n".join(
        f"{'User' if isinstance(msg, HumanMessage) else 'Assistant'}: {msg.content}"
        for msg in chat_history
    )


def classify_query(state: GraphState) -> dict:
    logging.info("Classifying query")
    question = state["question"]
    chat_history = state["chat_history"]

    prompt_template = ChatPromptTemplate.from_template(
        """You are a query classifier. The user can ask questions in any language.
        Classify the user's input into one of the following categories:
        - "informational": The user is asking a question that requires information retrieval (e.g., about company policies, projects, history).
        - "greeting": The user is saying hello or a similar greeting.
        - "chit_chat": The user is making a general conversational statement or asking a question not related to company information (e.g., "how are you?", "what's the weather?").

        Use the following chat history to help you understand the context of the user's latest message.

        Chat History:
        {chat_history}

        Provide the classification for the latest user input as a JSON with a single key 'query_type' and no preamble or explanation.
        User input: {question}"""
    )
    structured_llm_classifier = llm.with_structured_output(
        QueryTypeOutput, method="json_mode", include_raw=False
    )
    classifier_chain = prompt_template | structured_llm_classifier
    classification_result = classifier_chain.invoke(
        {"question": question, "chat_history": format_chat_history(chat_history)}
    )
    query_type = classification_result.query_type
    logging.info(f"Classification: {query_type}")
    return {
        "query_type": query_type,
        "question": question,
        "transform_attempts": state.get("transform_attempts", 0),
        "documents": [],
        "chat_history": chat_history,
    }


def handle_greeting_or_chit_chat(state: GraphState) -> dict:
    """
    Generates a polite, multilingual response for greetings or chit-chat,
    guiding the user back to the bot's main purpose.
    """
    logging.info("Handling greeting or chit-chat")
    question = state["question"]
    query_type = state["query_type"]
    chat_history = state["chat_history"]

    if query_type == "greeting":
        prompt_text = """You are a helpful company assistant. The user has just greeted you.
        Respond politely in the same language as the user's message.
        Briefly state that you are doing well and that your main purpose is to answer questions about the company.
        Ask them how you can help with company-related matters.

        User's message: {question}
        Your response:"""
    else:
        prompt_text = """You are a helpful company assistant. The user has made a conversational statement (chit-chat) that is not a direct question about the company.
        Respond politely in the same language as the user's message.
        Gently guide the conversation back to your purpose, which is to provide information about the company's policies, projects, or history.
        Ask if they have any questions on those topics.

        User's message: {question}
        Your response:"""

    prompt = ChatPromptTemplate.from_template(prompt_text)
    chain = prompt | llm | StrOutputParser()
    response_text = chain.invoke({"question": question})

    return {
        "generation": response_text,
        "question": state["question"],
        "chat_history": state["chat_history"],
    }


def retrieve_documents(state: GraphState) -> dict:
    """
    Retrieves documents from Qdrant using a direct embedding and search approach.
    """
    logging.info("Retrieving documents using direct embedding client")
    question = state["question"]

    try:
        logging.info(f"Embedding query: '{question}'")
        response = genai_client.models.embed_content(
            model=EMBEDDING_MODEL_NAME,
            contents=question,
            config=EmbedContentConfig(task_type="RETRIEVAL_QUERY"),
        )
        query_vector = response.embeddings[0].values
    except Exception as e:
        logging.error(f"Failed to embed query: {e}")
        return {
            "documents": [],
            "question": question,
            "chat_history": state["chat_history"],
        }

    try:
        logging.info("Performing vector search in Qdrant...")
        search_results = qdrant_client.search(
            collection_name=QDRANT_COLLECTION,
            query_vector=query_vector,
            limit=RETRIEVER_K,
            with_payload=True,
        )
    except Exception as e:
        logging.error(f"Failed to search Qdrant: {e}")
        return {
            "documents": [],
            "question": question,
            "chat_history": state["chat_history"],
        }

    documents = []
    for hit in search_results:
        payload = hit.payload
        if payload:
            documents.append(
                Document(
                    page_content=payload.get("page_content", ""),
                    metadata={k: v for k, v in payload.items() if k != "page_content"},
                )
            )

    logging.info(f"Retrieved {len(documents)} documents.")
    return {
        "documents": documents,
        "question": question,
        "chat_history": state["chat_history"],
    }


def grade_documents(state: GraphState) -> dict:
    logging.info("Grading documents")
    question = state["question"]
    documents = state["documents"]

    prompt = ChatPromptTemplate.from_template(
        """You are a grader assessing relevance of a retrieved document to a user question.
        The user's question can be in any language.
        The document's content will start with its file name or be a conversation thread.
        If the document contains keywords, a file name, or conversation topics related to the user question, grade it as relevant.
        It does not need to be a stringent test. The goal is to filter out erroneous retrievals.

        Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question.
        Provide the binary score as a JSON with a single key 'score' and no preamble or explanation.

        Retrieved document:
        \n\n {document} \n\n
        User question: {question}"""
    )

    structured_llm_grader = llm.with_structured_output(
        GraderOutput, method="json_mode", include_raw=False
    )

    grader_chain = prompt | structured_llm_grader

    filtered_docs = []
    for d in documents:
        response = grader_chain.invoke(
            {"question": question, "document": d.page_content}
        )
        grade = response.score
        if grade.lower() == "yes":
            logging.info("Document relevant")
            filtered_docs.append(d)
        else:
            logging.info("Document not relevant")
    return {
        "documents": filtered_docs,
        "question": question,
        "chat_history": state["chat_history"],
    }


def generate(state: GraphState) -> dict:
    """
    Generates a RAG answer, ensuring the response is in the same
    language as the user's question.
    """
    logging.info("Generating answer")
    question = state["question"]
    documents = state["documents"]
    chat_history = state["chat_history"]

    if not documents:
        return {
            "documents": [],
            "question": question,
            "generation": "I couldn't find any relevant documents to answer your question. Please try asking in a different way.",
            "chat_history": chat_history,
        }

    prompt = ChatPromptTemplate.from_template(
        """You are an assistant for question-answering tasks for a company.
        Use the following pieces of retrieved context to answer the question.
        Also, consider the chat history to make the answer conversational and understand context.
        IMPORTANT: You must identify the language of the user's question and provide your answer in that same language.

        For files, the context begins with "File Name: [the file's name]". If the user is asking for a specific document by its name or topic, explicitly mention the file name in your answer.
        If you don't know the answer, just say that you don't know (in the user's language).
        Use three sentences maximum and keep the answer concise.

        Chat History:
        {chat_history}

        Question: {question}
        Context: {context}
        Answer:"""
    )

    context = "\n".join([d.page_content for d in documents])
    rag_chain = prompt | llm | StrOutputParser()
    generation = rag_chain.invoke(
        {
            "context": context,
            "question": question,
            "chat_history": format_chat_history(chat_history),
        }
    )

    if documents:
        unique_sources = {}
        for d in documents:
            if d.metadata and d.metadata.get("source"):
                source_url = d.metadata.get("source")
                if source_url not in unique_sources:
                    file_name = d.metadata.get("file_name", "Source Link")
                    link_text = html.escape(file_name)
                    unique_sources[source_url] = link_text

        if unique_sources:
            sorted_sources = sorted(unique_sources.items(), key=lambda item: item[1])
            source_list_items = [
                f'{i}. <a href="{url}">{text}</a>'
                for i, (url, text) in enumerate(sorted_sources, 1)
            ]
            source_list_str = "\n".join(source_list_items)
            generation += f"\n\n<b>Source(s):</b>\n{source_list_str}"

    return {
        "documents": documents,
        "question": question,
        "generation": generation,
        "chat_history": chat_history,
    }


def transform_query(state: GraphState) -> dict:
    """
    Rewrites the user's question to be more effective for retrieval,
    maintaining the original language.
    """
    logging.info("Transforming query")
    question = state["question"]
    chat_history = state["chat_history"]

    prompt = ChatPromptTemplate.from_template(
        """You are a query transformation expert. Your task is to rewrite a user's question to be more effective for a vector database search, taking into account the conversation history.
        If the user's question is a follow-up (e.g., "what about them?", "tell me more"), use the history to resolve the ambiguity and create a self-contained question.
        Reformulate it into a concise and specific question that is more likely to retrieve relevant documents.
        IMPORTANT: The rewritten question must be in the same language as the original user's question.

        Example (English):
        User: "Tell me about Project Phoenix"
        Assistant: "Project Phoenix is a new initiative..."
        User's new question: "Who is the project manager?"
        Rewritten self-contained question: "Who is the project manager for Project Phoenix?"

        Example (Spanish):
        User: "Háblame del Proyecto Fénix"
        Assistant: "El Proyecto Fénix es una nueva iniciativa..."
        User's new question: "¿quién es el gerente del proyecto?"
        Rewritten self-contained question: "¿Quién es el gerente del Proyecto Fénix?"

        Chat History:
        {chat_history}

        Original question: {question}
        Rewritten question:"""
    )

    rewriter = prompt | llm | StrOutputParser()
    better_question = rewriter.invoke(
        {"question": question, "chat_history": format_chat_history(chat_history)}
    )
    logging.info(f"Rewritten question: {better_question}")
    new_attempts = state.get("transform_attempts", 0) + 1
    return {
        "documents": [],
        "question": better_question,
        "transform_attempts": new_attempts,
        "chat_history": chat_history,
    }


def decide_to_generate(state: GraphState) -> str:
    logging.info("Deciding to generate or transform query")
    documents = state["documents"]
    current_attempts = state.get("transform_attempts", 0)

    if not documents:
        if current_attempts < MAX_TRANSFORM_ATTEMPTS:
            logging.info(
                f"Rewrite query (Attempt {current_attempts + 1} of {MAX_TRANSFORM_ATTEMPTS})"
            )
            return "transform_query"
        else:
            logging.info(
                f"Max rewrite attempts reached ({current_attempts}), no documents found."
            )
            return "handle_no_answer"
    else:
        logging.info("Proceed to generate")
        return "generate"


def handle_no_answer(state: GraphState) -> dict:
    """
    Generates a message in the user's language indicating that no
    answer could be found after all attempts.
    """
    logging.info("Handling no answer")
    question = state["question"]
    chat_history = state["chat_history"]

    prompt = ChatPromptTemplate.from_template(
        """You are a helpful company assistant.
        You were unable to find any relevant documents to answer the user's question, even after trying to rephrase it.
        Inform the user about this politely.
        Suggest that they try rephrasing their question with more specific terms.
        IMPORTANT: You must respond in the same language as the user's original question.

        User's original question: {question}
        Your response:"""
    )
    chain = prompt | llm | StrOutputParser()
    generation_message = chain.invoke({"question": question})

    return {
        "documents": [],
        "question": question,
        "generation": generation_message,
        "chat_history": chat_history,
    }


def decide_query_type(state: GraphState) -> str:
    logging.info(f"Routing query type ({state['query_type']})")
    if state["query_type"] == "informational":
        return "retrieve"
    else:
        return "handle_greeting_or_chit_chat"


workflow = StateGraph(GraphState)
workflow.add_node("classify_query", classify_query)
workflow.add_node("handle_greeting_or_chit_chat", handle_greeting_or_chit_chat)
workflow.add_node("retrieve", retrieve_documents)
workflow.add_node("grade_documents", grade_documents)
workflow.add_node("generate", generate)
workflow.add_node("transform_query", transform_query)
workflow.add_node("handle_no_answer", handle_no_answer)

workflow.set_entry_point("classify_query")

workflow.add_conditional_edges(
    "classify_query",
    decide_query_type,
    {
        "retrieve": "retrieve",
        "handle_greeting_or_chit_chat": "handle_greeting_or_chit_chat",
    },
)
workflow.add_edge("handle_greeting_or_chit_chat", END)
workflow.add_edge("retrieve", "grade_documents")
workflow.add_conditional_edges(
    "grade_documents",
    decide_to_generate,
    {
        "transform_query": "transform_query",
        "generate": "generate",
        "handle_no_answer": "handle_no_answer",
    },
)
workflow.add_edge("transform_query", "retrieve")
workflow.add_edge("generate", END)
workflow.add_edge("handle_no_answer", END)

app = workflow.compile()
logging.info("Graph compiled successfully")
