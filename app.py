import os
from flask import Flask, render_template, request
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate

# ==============================================================================
# 1. CONFIGURAÇÕES INICIAIS
# ==============================================================================
# ATENÇÃO: Cole a sua chave do Groq aqui (começa com gsk_)
os.environ["GROQ_API_KEY"] = "SUA_CHAVE_AQUI"

app = Flask(__name__)

PATH_MANUAIS = "data/"
PATH_CHROMA_DB = "chroma_db/"

print("Inicializando Motor RAG... Aguarde.")

# ==============================================================================
# 2. INGESTÃO E INDEXAÇÃO (Fase Estática Local)
# ==============================================================================
loader = PyPDFDirectoryLoader(PATH_MANUAIS)
documents = loader.load()

# Geração de vetores local (sem custo e sem limite)
embeddings_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

if len(documents) > 0:
    print(f"Indexando {len(documents)} páginas de PDFs localmente...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=102)
    splits = text_splitter.split_documents(documents)
    
    vectorstore = Chroma.from_documents(
        documents=splits, 
        embedding=embeddings_model, 
        persist_directory=PATH_CHROMA_DB
    )
else:
    print("Nenhum PDF novo encontrado. Carregando banco de vetores existente...")
    vectorstore = Chroma(persist_directory=PATH_CHROMA_DB, embedding_function=embeddings_model)

# ==============================================================================
# 3. NÚCLEO RAG TRANSPARENTE
# ==============================================================================
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

# O LLM agora usa o Llama 3.3 via Groq (Super rápido, sem erros de API de nomenclatura)
llm = ChatGroq(model_name="llama-3.3-70b-versatile", temperature=0)

system_prompt = (
    "Você é um assistente de suporte técnico Nível 1.\n"
    "Responda à dúvida do analista utilizando estritamente o contexto abaixo.\n"
    "Se a resposta não estiver no contexto recuperado, diga apenas que não localizou a instrução no manual.\n\n"
    "Contexto recuperado:\n{context}"
)
prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt), 
    ("human", "{input}")
])

def executar_rag(pergunta_usuario):
    documentos_recuperados = retriever.invoke(pergunta_usuario)
    contexto_texto = "\n\n".join([doc.page_content for doc in documentos_recuperados])
    prompt_pronto = prompt.invoke({"context": contexto_texto, "input": pergunta_usuario})
    
    # Chama o Groq para gerar a resposta
    resposta_llm = llm.invoke(prompt_pronto)
    
    return {
        "answer": resposta_llm.content,
        "context": documentos_recuperados
    }

print("Servidor RAG Pronto e Operacional!")

# ==============================================================================
# 4. ROTAS DO FLASK
# ==============================================================================
@app.route('/', methods=['GET', 'POST'])
def index():
    user_question = ""
    bot_response = ""
    sources = []

    if request.method == 'POST':
        user_question = request.form.get('question', '')
        response = executar_rag(user_question)
        bot_response = response["answer"]
        
        for doc in response["context"]:
            sources.append({
                "source": os.path.basename(doc.metadata.get('source', 'Desconhecido')),
                "page": doc.metadata.get('page', 0)
            })

    return render_template('index.html', user_question=user_question, bot_response=bot_response, sources=sources)

if __name__ == '__main__':
    app.run(debug=True, port=5000)