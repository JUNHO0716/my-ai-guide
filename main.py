import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_chroma import Chroma

# === 1) 환경 변수 / 기본 설정 ===
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("❌ OPENAI_API_KEY가 설정되지 않았습니다. Render 환경변수에 추가하세요.")

# 캐시 디렉토리 설정 (무료 플랜에서도 빠르게)
PERSIST_DIR = "./chroma_cache"
os.makedirs(PERSIST_DIR, exist_ok=True)

# === 2) 지식 데이터 로드 & 캐싱 ===
print("[INFO] Initializing knowledge base...")
loader = TextLoader("내홈페이지정보.txt", encoding="utf-8")
docs = CharacterTextSplitter(chunk_size=600, chunk_overlap=80).split_documents(loader.load())

embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

# 캐시 폴더에 임베딩 저장 (속도 향상)
db = Chroma.from_documents(docs, embeddings, persist_directory=PERSIST_DIR)
db.persist()

retriever = db.as_retriever()
print("[INFO] Knowledge base ready ✅")

# === 3) LLM 설정 (응답 시간 제한 20초) ===
llm = ChatOpenAI(
    model_name="gpt-4o-mini",
    temperature=0,
    openai_api_key=OPENAI_API_KEY,
    request_timeout=20
)

qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)

# === 4) Flask 서버 ===
app = Flask(__name__)
app.config["JSON_AS_ASCII"] = False  # ✅ 한글 깨짐 방지
CORS(app, resources={r"/ask": {"origins": ["https://mathpb.com", "http://localhost:5173", "http://localhost:3000"]}})

@app.route("/health", methods=["GET"])
def health():
    return jsonify(status="ok", version="v1")

@app.route("/ask", methods=["POST"])
def ask():
    try:
        data = request.get_json(silent=True) or {}
        question = data.get("question", "").strip()
        if not question:
            return jsonify({"error": "No question provided"}), 400

        print(f"[INFO] New question: {question}", flush=True)
        resp = qa_chain.invoke({"query": question})
        answer = resp.get("result", "").strip() or "답변을 찾을 수 없습니다."
        print(f"[INFO] Answer generated: {answer}", flush=True)
        return jsonify({"answer": answer})

    except Exception as e:
        import traceback
        print("[ERROR]", traceback.format_exc(), flush=True)
        return jsonify({"error": f"[server] {e}"}), 500

# === 5) 관리자용 지식 리로드 엔드포인트 ===
@app.route("/reload", methods=["POST"])
def reload_knowledge():
    admin_token = os.environ.get("ADMIN_TOKEN", "")
    provided = (request.headers.get("X-Admin-Token") or "").strip()
    if not admin_token or provided != admin_token:
        return jsonify({"error": "Unauthorized"}), 401
    global qa_chain
    qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)
    return jsonify({"status": "reloaded"})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8080"))
    app.run(host="0.0.0.0", port=port)
