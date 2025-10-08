import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_chroma import Chroma

# === 1) 환경변수/기본 설정 ===
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY가 설정되지 않았습니다. Render 환경변수에 추가하세요.")

# Render 무료 플랜은 컨테이너 재시작 시 디스크가 초기화될 수 있음
# → 작은 지식 파일은 매번 부팅 시 임베딩 생성(수 초~수십 초)으로 운용하거나,
#   유료 Persistent Disk를 쓰면 persist_directory로 캐싱 가능.
PERSIST_DIR = os.environ.get("CHROMA_DIR", None)  # 예: "/data/chroma" (Render Persistent Disk 장착 시)

def build_qa_chain():
    loader = TextLoader("내홈페이지정보.txt", encoding="utf-8")
    documents = loader.load()

    text_splitter = CharacterTextSplitter(chunk_size=600, chunk_overlap=80)
    docs = text_splitter.split_documents(documents)

    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

    if PERSIST_DIR:
        db = Chroma.from_documents(docs, embeddings, persist_directory=PERSIST_DIR)
        db.persist()
    else:
        db = Chroma.from_documents(docs, embeddings)

    retriever = db.as_retriever()
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0, openai_api_key=OPENAI_API_KEY)
    return RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)

qa_chain = build_qa_chain()

# === 2) Flask 앱/CORS ===
app = Flask(__name__)
app.config["JSON_AS_ASCII"] = False
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

        print(f"[INFO] New question: {question}", flush=True)  # ✅ 요청 로그 추가
        resp = qa_chain.invoke({"query": question})
        answer = resp.get("result", "").strip() or "No answer found."
        print(f"[INFO] Answer generated: {answer}", flush=True)  # ✅ 응답 로그 추가
        return jsonify({"answer": answer})

    except Exception as e:
        import traceback
        print("[ERROR]", traceback.format_exc(), flush=True)  # ✅ 전체 에러로그 출력
        return jsonify({"error": f"[server] {e}"}), 500

# 지식 파일 갱신 시 핫리로드(간단 토큰)
@app.route("/reload", methods=["POST"])
def reload_knowledge():
    admin_token = os.environ.get("ADMIN_TOKEN", "")
    provided = (request.headers.get("X-Admin-Token") or "").strip()
    if not admin_token or provided != admin_token:
        return jsonify({"error": "Unauthorized"}), 401
    global qa_chain
    qa_chain = build_qa_chain()
    return jsonify({"status": "reloaded"})

if __name__ == "__main__":
    # Render는 PORT 환경변수를 제공하는 경우가 많음
    port = int(os.environ.get("PORT", "8080"))
    app.run(host="0.0.0.0", port=port)
