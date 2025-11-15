import os
import gradio as gr
from ingest import load_vectorstore
from langchain.prompts import PromptTemplate
from langchain_community.llms import HuggingFacePipeline
from langchain_core.runnables import RunnablePassthrough
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch

# --- Configuration ---
DB_FAISS_PATH = "vectorstore/db_faiss"
LLM_MODEL_ID = "google/gemma-2b-it" 

# Global variable to hold the initialized RAG chain
RAG_CHAIN = None
FINAL_LLM = None

def setup_llm(model_id):
    """Initializes the LLM pipeline for the RAG chain."""
    print(f"Loading LLM: {model_id}. This should load much faster...")
    try:
        # Load in 4-bit, but ensure torch dtype is used
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16, # Use bfloat16 for Gemma stability
            device_map="auto",
            load_in_4bit=True
        )
        
        # Ensure the model is in evaluation mode
        model.eval()

        text_pipeline = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=512,
            temperature=0.1,
            do_sample=True,
            return_full_text=False
        )
        
        llm = HuggingFacePipeline(pipeline=text_pipeline)
        return llm
    except Exception as e:
        print(f"Failed to load LLM ({model_id}): {e}")
        print("Please ensure you have a T4 GPU selected in the Colab Runtime settings.")
        return None

def format_docs(docs):
    """Formats the retrieved documents into a string with clear source citations."""
    formatted_context = ""
    for i, doc in enumerate(docs):
        source = os.path.basename(doc.metadata.get('source', 'Unknown'))
        page = doc.metadata.get('page', 'N/A')
        
        formatted_context += (
            f"--- LEGAL DOCUMENT SNIPPET {i+1} ---\n"
            f"Source: {source} (Page: {page})\n"
            f"Content: {doc.page_content}\n"
        )
        
    return formatted_context

def initialize_rag_chain():
    """Initializes the entire RAG pipeline."""
    global RAG_CHAIN, FINAL_LLM

    print("Loading Vector Store...")
    vector_db = load_vectorstore()
    if vector_db is None:
        return False
        
    FINAL_LLM = setup_llm(LLM_MODEL_ID)
    if FINAL_LLM is None:
        return False

    rag_template = """
    You are a meticulous Legal Retrieval Assistant. Your task is to generate a comprehensive, 
    raw answer to the QUESTION based ONLY on the provided CONTEXT. 
    
    You MUST cite the source of every fact you use by including the SNIPPET number (e.g., [Snippet 1], [Snippet 3]) 
    at the end of the sentence or clause. Do not add any introductory or concluding remarks.
    
    CONTEXT:
    {context}
    
    QUESTION: {question}
    
    RAW, CITED ANSWER:
    """
    
    rag_prompt = PromptTemplate(template=rag_template, input_variables=["context", "question"])
    retriever = vector_db.as_retriever(search_kwargs={"k": 5})

    RAG_CHAIN = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | rag_prompt
        | FINAL_LLM
    )
    
    print("✅ RAG Chain and LLMs initialized successfully.")
    return True

def chat_function(question, history):
    """The main function for the Gradio interface."""
    if RAG_CHAIN is None:
        return "System Error: The RAG pipeline failed to initialize. Please check the console."

    try:
        raw_answer = RAG_CHAIN.invoke(question)

        # Retrieve documents separately to get citation information
        docs = RAG_CHAIN.get_input_schema().steps[0].runnable.retriever.invoke(question)

        source_references = []
        for doc in docs:
            source = os.path.basename(doc.metadata.get('source', 'Unknown'))
            page = doc.metadata.get('page', 'N/A')
            source_references.append(f"- {source} (Page: {page})")
        
        final_response = (
            f"**Legal Expert Response:**\n\n{raw_answer}\n\n"
            f"**📚 Sources Used:**\n"
            f"{'\\n'.join(sorted(list(set(source_references))))}"
        )
        
        return final_response

    except Exception as e:
        return f"An error occurred during chat processing: {e}"

if initialize_rag_chain():
    iface = gr.ChatInterface(
        fn=chat_function,
        chatbot=gr.Chatbot(height=500),
        textbox=gr.Textbox(placeholder="Ask your legal question here...", container=False, scale=7),
        title="🇮🇳 Multi-Agent Legal RAG System (Gemma-2B)",
        description=(
            "Ask a question about the Indian Contract Act, Company Law, or related judgements/regulations. "
            "This system uses the efficient **Gemma-2B** model, BGE Embeddings, and a RAG pipeline for cited answers."
        ),
        submit_btn="Analyze Law",
        # REMOVED: The problematic 'clear_btn' argument
    )
    iface.launch(share=True)
