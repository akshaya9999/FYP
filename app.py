import os
import hashlib
import json
from datetime import datetime, timedelta
from flask import Flask, render_template, request, jsonify, session
from werkzeug.utils import secure_filename
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq
from chromadb import PersistentClient
import redis
import bcrypt
from dotenv import load_dotenv
from prompts import QA_PROMPT, SUMMARY_PROMPT
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import re
from faster_whisper import WhisperModel
import soundfile as sf
import numpy as np
import os
import subprocess
import soundfile as sf
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2ForSequenceClassification
import torch
from rlhf_inference import RLHFRanker
import uuid


whisper_model = WhisperModel("small", device="cpu")

SER_MODEL_PATH = "ser_model"
SER_CONFIDENCE_THRESHOLD = 0.55
# SER_ID_TO_LABEL = {0: "neutral", 1: "mild_distress", 2: "low_energy"}

SER_ID_TO_LABEL = {
    0: "neutral",
    1: "happy",
    2: "sad",
    3: "angry",
    4: "fearful",
}

rlhf_ranker = RLHFRanker()
 
def load_ser_model(model_path: str):
    if not os.path.exists(model_path):
        print(f"[SER] Model not found at {model_path}. Will use energy heuristic only.")
        return None, None

    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_path)
    model = Wav2Vec2ForSequenceClassification.from_pretrained(model_path)
    model.eval()
    print(f"[SER] Loaded fine-tuned model from {model_path}")
    return feature_extractor, model

ser_feature_extractor, ser_model = load_ser_model(SER_MODEL_PATH)

analyzer = SentimentIntensityAnalyzer()


load_dotenv()

# Configuration
UPLOAD_FOLDER = "uploads"
PERSIST_DIR = "chroma_db"
ALLOWED_EXTENSIONS = {"txt", "pdf"}
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
MAX_HISTORY_MESSAGES = 10  # Number of previous messages to include in context

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PERSIST_DIR, exist_ok=True)

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["SECRET_KEY"] = os.getenv("SECRET_KEY", "your-secret-key-change-this")

# Redis setup for user data and chat history
redis_client = redis.Redis(
    host=os.getenv("REDIS_HOST", "localhost"),
    port=int(os.getenv("REDIS_PORT", 6379)),
    db=0,
    decode_responses=True
)

# Embeddings
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Groq LLM
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
model = ChatGroq(api_key=GROQ_API_KEY, model="llama-3.1-8b-instant")


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def get_user_collection_name(email):
    """Generate unique collection name for user."""
    email_hash = hashlib.md5(email.encode()).hexdigest()[:16]
    return f"user_{email_hash}"


def get_user_chroma_client(email):
    """Get or create Chroma collection for specific user."""
    chroma_client = PersistentClient(path=PERSIST_DIR)
    collection_name = get_user_collection_name(email)
    
    try:
        collection = chroma_client.get_collection(collection_name)
    except:
        collection = chroma_client.create_collection(collection_name)
    
    return chroma_client, collection, collection_name


def store_chat_message(email, role, message):
    """Store chat message in Redis for user."""
    chat_key = f"chat_history:{email}"
    message_data = {
        "role": role,
        "message": message,
        "timestamp": datetime.now().isoformat()
    }
    redis_client.rpush(chat_key, json.dumps(message_data))
    
    # Keep only last MAX_HISTORY_MESSAGES * 2 (user + bot messages)
    redis_client.ltrim(chat_key, -(MAX_HISTORY_MESSAGES * 2), -1)


def get_chat_history(email):
    """Retrieve chat history for user."""
    chat_key = f"chat_history:{email}"
    messages = redis_client.lrange(chat_key, 0, -1)
    return [json.loads(msg) for msg in messages]

def get_user_summary(email):
    """Get long-term summary memory for user."""
    key = f"user_summary:{email}"
    return redis_client.get(key) or ""


def update_user_summary(email, new_summary):
    """Persist updated summary memory."""
    key = f"user_summary:{email}"
    redis_client.set(key, new_summary)


def build_conversation_context(email):
    """Build conversation context from chat history."""
    history = get_chat_history(email)
    if not history:
        return ""
    
    context_parts = []
    for msg in history[-MAX_HISTORY_MESSAGES:]:
        role = "User" if msg["role"] == "user" else "Assistant"
        context_parts.append(f"{role}: {msg['message']}")
    
    return "\n".join(context_parts)

def generate_n_responses(query: str, retriever, n: int = 3) -> list:
    """
    Generate N candidate responses using the RAG chain.
    Uses temperature variation to get diverse candidates.
    """
    from langchain_groq import ChatGroq
    from langchain.chains import RetrievalQA
    from prompts import QA_PROMPT
 
    candidates = []
    temperatures = [0.3, 0.7, 1.0]  # low → high diversity
 
    for i in range(n):
        try:
            temp_model = ChatGroq(
                api_key=GROQ_API_KEY,
                model="llama-3.1-8b-instant",
                temperature=temperatures[i % len(temperatures)],
            )
            chain = RetrievalQA.from_chain_type(
                llm=temp_model,
                retriever=retriever,
                return_source_documents=False,
                chain_type_kwargs={"prompt": QA_PROMPT},
            )
            result = chain({"query": query})
            candidates.append(result["result"])
        except Exception as e:
            print(f"[RLHF] Candidate {i+1} generation failed: {e}")
 
    return candidates if candidates else ["I'm here to help. Could you tell me more?"]
 
 
def store_feedback(email: str, message_id: str, query: str, response: str, rating: int):
    """
    Store user feedback in Redis.
    rating: 1 = thumbs up, 0 = thumbs down
    """
    import json
    from datetime import datetime
 
    key  = f"feedback:{email}:{message_id}"
    data = {
        "query":     query,
        "response":  response,
        "rating":    rating,
        "email":     email,
        "timestamp": datetime.now().isoformat(),
    }
    redis_client.set(key, json.dumps(data))
    print(f"[RLHF] Feedback stored — {'👍' if rating == 1 else '👎'} for message {message_id}")
 

def ingest_file_to_chroma(path, filename, email):
    """Load file, chunk, embed, and add to user's Chroma collection."""
    ext = filename.rsplit('.', 1)[1].lower()
    if ext == 'txt':
        loader = TextLoader(path, encoding='utf-8')
        docs = loader.load()
    elif ext == 'pdf':
        loader = PyPDFLoader(path)
        docs = loader.load()
    else:
        return 0

    # Split into chunks
    text_splitter = CharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    split_docs = text_splitter.split_documents(docs)

    # Get user's collection
    _, collection, _ = get_user_chroma_client(email)

    # Extract text and metadata
    texts = [d.page_content for d in split_docs]
    metadatas = [d.metadata for d in split_docs]
    ids = [f"{filename}-{i}" for i in range(len(split_docs))]

    # Compute embeddings
    embeddings = embedding_model.embed_documents(texts)

    # Add to user's collection
    collection.add(ids=ids, metadatas=metadatas, documents=texts, embeddings=embeddings)

    return len(split_docs)


EMAIL_REGEX = re.compile(
    r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
)

def is_valid_email(email: str) -> bool:
    return bool(EMAIL_REGEX.match(email))

@app.route("/signup", methods=["POST"])
def signup():
    """User signup endpoint."""
    data = request.get_json()
    email = data.get("email", "").strip().lower()
    password = data.get("password", "")
    name = data.get("name", "").strip()

    # Basic presence checks
    if not email or not password:
        return jsonify({"error": "Email and password required"}), 400

    # Email format validation
    if not is_valid_email(email):
        return jsonify({"error": "Invalid email format"}), 400

    # Optional: enforce password strength (recommended)
    if len(password) < 8:
        return jsonify({"error": "Password must be at least 8 characters long"}), 400

    # Check if user exists
    user_key = f"user:{email}"
    if redis_client.exists(user_key):
        return jsonify({"error": "User already exists"}), 400

    # Hash password
    password_hash = bcrypt.hashpw(
        password.encode("utf-8"),
        bcrypt.gensalt()
    ).decode("utf-8")

    # Store user data
    user_data = {
        "email": email,
        "name": name,
        "password_hash": password_hash,
        "created_at": datetime.now().isoformat()
    }
    redis_client.hset(user_key, mapping=user_data)

    return jsonify({"message": "Signup successful", "email": email}), 201


@app.route("/signin", methods=["POST"])
def signin():
    """User signin endpoint."""
    data = request.get_json()
    email = data.get("email", "").strip().lower()
    password = data.get("password", "")

    if not email or not password:
        return jsonify({"error": "Email and password required"}), 400

    # Get user data
    user_key = f"user:{email}"
    user_data = redis_client.hgetall(user_key)

    if not user_data:
        return jsonify({"error": "Invalid credentials"}), 401

    # Verify password
    password_hash = user_data.get("password_hash", "")
    if not bcrypt.checkpw(password.encode('utf-8'), password_hash.encode('utf-8')):
        return jsonify({"error": "Invalid credentials"}), 401

    # Create session
    session["user_email"] = email
    session["user_name"] = user_data.get("name", "")
    session.permanent = True
    app.permanent_session_lifetime = timedelta(days=7)

    return jsonify({
        "message": "Signin successful",
        "email": email,
        "name": user_data.get("name", "")
    }), 200

 
@app.route("/feedback", methods=["POST"])
def submit_feedback():
    """
    Receives thumbs up/down from frontend.
    Stores (query, response, rating) for reward model training.
    """
    if "user_email" not in session:
        return jsonify({"error": "Not authenticated"}), 401
 
    data       = request.get_json()
    message_id = data.get("message_id", "")
    query      = data.get("query", "")
    response   = data.get("response", "")
    rating     = data.get("rating")   # 1 = thumbs up, 0 = thumbs down
 
    if not message_id or not query or not response or rating not in (0, 1):
        return jsonify({"error": "Invalid feedback data"}), 400
 
    store_feedback(
        email=session["user_email"],
        message_id=message_id,
        query=query,
        response=response,
        rating=rating,
    )
 
    return jsonify({"message": "Feedback recorded. Thank you!"}), 200
 
 

@app.route("/signout", methods=["POST"])
def signout():
    """User signout endpoint."""
    session.clear()
    return jsonify({"message": "Signout successful"}), 200


@app.route("/me", methods=["GET"])
def get_current_user():
    """Get current logged-in user."""
    if "user_email" not in session:
        return jsonify({"authenticated": False}), 200
    
    return jsonify({
        "authenticated": True,
        "email": session["user_email"],
        "name": session.get("user_name", "")
    }), 200



@app.route("/")
def index():
    return render_template("chat.html")


@app.route("/upload", methods=["POST"])
def upload_file():
    """Upload file for authenticated user."""
    if "user_email" not in session:
        return jsonify({"error": "Not authenticated"}), 401

    if "file" not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(file_path)
        
        # Ingest to user's collection
        email = session["user_email"]
        chunks = ingest_file_to_chroma(file_path, filename, email)
        
        return jsonify({"message": f"Uploaded and processed {chunks} chunks to your personal knowledge base."})
    else:
        return jsonify({"error": "Invalid file type"}), 400

@app.route("/voice_chat", methods=["POST"])
def voice_chat():
    if "user_email" not in session:
        return jsonify({"error": "Not authenticated"}), 401

    if "audio" not in request.files:
        return jsonify({"error": "No audio file"}), 400

    file = request.files["audio"]
    filepath = "temp_audio.webm"
    file.save(filepath)

    email = session["user_email"]

    # 1️⃣ Transcribe speech
    text = transcribe_audio(filepath)

    # 2️⃣ Detect audio emotion
    audio_emotion = detect_audio_emotion(filepath)

    # 3️⃣ Text sentiment
    sentiment = analyzer.polarity_scores(text)
    text_score = sentiment["compound"]

    print("Transcribed:", text)
    print("Audio emotion:", audio_emotion)
    print("Text sentiment:", text_score)

    # Store user voice message in history
    store_chat_message(email, "user", text)
    print("VOICE ROUTE HIT")

    # 4️⃣ Emotional routing (same logic as chat)
    # if text_score <= -0.5 or audio_emotion == "low_energy":
    #     comforting = (
    #         "I can sense that you might be going through something difficult. "
    #         "You're not alone. I'm here to listen. "
    #         "If you're feeling overwhelmed, please consider reaching out to a professional."
    #     )
    if text_score <= -0.5 or audio_emotion in ("sad", "fearful"):
        comforting = (
            "I can sense that you might be going through something really difficult. "
            "You're not alone, and I'm here to listen. "
            "If these feelings persist, please consider speaking to a mental health professional — "
            "it's one of the strongest things you can do for yourself."
        )
        message_id = str(uuid.uuid4())
        store_chat_message(email, "bot", comforting)
        return jsonify({
            "transcribed_text": text,
            "audio_emotion": audio_emotion,
            "response": comforting,
            "message_id": message_id
        })

    elif audio_emotion == "angry":
        comforting = (
            "I can hear some frustration in your voice. "
            "That's completely valid. Take a breath — I'm here to help."
        )
        message_id = str(uuid.uuid4())
        store_chat_message(email, "bot", comforting)
        return jsonify({
            "transcribed_text": text,
            "audio_emotion": audio_emotion,
            "response": comforting,
            "message_id": message_id
        })

    # happy and neutral fall through to RAG pipeline below
    conversation_context = build_conversation_context(email)
    long_term_summary = get_user_summary(email)

    chroma_client, _, collection_name = get_user_chroma_client(email)
    vectorstore = Chroma(
        client=chroma_client,
        collection_name=collection_name,
        embedding_function=embedding_model
    )
    retriever = vectorstore.as_retriever()

    if conversation_context:
        enhanced_query = f"""
    Long-term memory:
    {long_term_summary}

    Recent conversation:
    {conversation_context}

    Current question:
    {text}
    """
    else:
        enhanced_query = text

    if rlhf_ranker.is_ready():
        candidates = generate_n_responses(enhanced_query, retriever, n=3)
        bot_response, reward_score = rlhf_ranker.best_of_n(text, candidates)
        print(f"[RLHF] Voice Best-of-3 selected (reward={reward_score:.3f})")
    else:
        qa_chain = RetrievalQA.from_chain_type(
            llm=model,
            retriever=retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": QA_PROMPT}
        )
        response = qa_chain({"query": enhanced_query})
        bot_response = response["result"]
        print("[RLHF] No reward model yet — using single response.")

    message_id = str(uuid.uuid4())
    store_chat_message(email, "bot", bot_response)

    return jsonify({
        "transcribed_text": text,
        "audio_emotion": audio_emotion,
        "response": bot_response,
        "message_id": message_id
    })
       



@app.route("/chat", methods=["POST"])
def chat():
    """Chat endpoint with RLHF Best-of-N response selection."""
    if "user_email" not in session:
        return jsonify({"error": "Not authenticated"}), 401
 
    email = session["user_email"]
    data  = request.get_json()
    query = data.get("query", "")
 
    sentiment = analyzer.polarity_scores(query)
    score     = sentiment["compound"]
 
    print(f"\n=== Sentiment: {score} ===")
 
    # Emotional routing — same as before
    if score <= -0.5:
        comforting = (
            "I'm really sorry you're feeling this way. "
            "You're not alone, and I'm right here with you. "
            "Would you like to talk about what's troubling you? "
            "If you're feeling really bad, please seek professional help."
        )
        message_id = str(uuid.uuid4())
        store_chat_message(email, "bot", comforting)
        return jsonify({"response": comforting, "message_id": message_id})
 
    elif -0.5 < score < -0.1:
        gentle = (
            "I can hear that you're going through something difficult. "
            "I'm here for you. How can I support you right now?"
        )
        message_id = str(uuid.uuid4())
        store_chat_message(email, "bot", gentle)
        return jsonify({"response": gentle, "message_id": message_id})
 
    if not query:
        return jsonify({"error": "Empty query"}), 400
 
    store_chat_message(email, "user", query)
 
    conversation_context = build_conversation_context(email)
    long_term_summary    = get_user_summary(email)
 
    chroma_client, _, collection_name = get_user_chroma_client(email)
    vectorstore = Chroma(
        client=chroma_client,
        collection_name=collection_name,
        embedding_function=embedding_model,
    )
    retriever = vectorstore.as_retriever()
 
    if conversation_context:
        enhanced_query = f"""
Long-term memory about the user:
{long_term_summary}
 
Recent conversation:
{conversation_context}
 
Current question:
{query}
"""
    else:
        enhanced_query = query
 
    # ── Best-of-N with reward model ────────────────────────────────────────
    if rlhf_ranker.is_ready():
        # Generate 3 candidates, pick the best one
        candidates   = generate_n_responses(enhanced_query, retriever, n=3)
        bot_response, reward_score = rlhf_ranker.best_of_n(query, candidates)
        print(f"[RLHF] Best-of-3 selected (reward={reward_score:.3f})")
    else:
        # Reward model not trained yet — fall back to single response
        qa_chain = RetrievalQA.from_chain_type(
            llm=model,
            retriever=retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": QA_PROMPT},
        )
        response     = qa_chain({"query": enhanced_query})
        bot_response = response["result"]
        print("[RLHF] No reward model yet — using single response.")
 
    # Unique ID so frontend can attach feedback to this specific response
    message_id = str(uuid.uuid4())
 
    store_chat_message(email, "bot", bot_response)
    increment_message_count(email)
    count = get_message_count(email)
 
    if count % 5 == 0:
        recent_history = build_conversation_context(email)
        old_summary    = get_user_summary(email)
        summary_chain  = SUMMARY_PROMPT | model
        updated_summary = summary_chain.invoke({
            "old_summary": old_summary,
            "conversation": recent_history,
        }).content
        update_user_summary(email, updated_summary)
 
    return jsonify({"response": bot_response, "message_id": message_id})



@app.route("/history", methods=["GET"])
def get_history():
    """Get chat history for current user."""
    if "user_email" not in session:
        return jsonify({"error": "Not authenticated"}), 401

    email = session["user_email"]
    history = get_chat_history(email)
    
    return jsonify({"history": history}), 200

def get_message_count(email):
    key = f"msg_count:{email}"
    return int(redis_client.get(key) or 0)


def increment_message_count(email):
    key = f"msg_count:{email}"
    redis_client.incr(key)

def transcribe_audio(path):
    segments, info = whisper_model.transcribe(path)
    text = ""
    for segment in segments:
        text += segment.text + " "
    return text.strip()

def energy_heuristic(audio: np.ndarray) -> tuple:
    energy = np.mean(audio ** 2)
    if energy < 0.001:
        return "low_energy", 0.9
    elif energy < 0.005:
        return "mild_distress", 0.7
    else:
        return "neutral", 0.8
        
def detect_audio_emotion(file_path: str) -> str:
    """
    Two-method emotion detection:
      1. Primary:  fine-tuned wav2vec2 SER model
      2. Fallback: energy heuristic (if model absent or low confidence)

    Returns one of: 'neutral', 'mild_distress', 'low_energy'
    """
    TARGET_SR   = 16000
    MAX_SAMPLES = TARGET_SR * 5   # 5-second clip

    # ── Convert webm → wav ──────────────────────────────────────────────────
    wav_path = "converted_audio.wav"
    subprocess.run(
        ["ffmpeg", "-i", file_path, "-ac", "1", "-ar", str(TARGET_SR), wav_path, "-y"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    # ── Load audio ──────────────────────────────────────────────────────────
    try:
        audio, sr = sf.read(wav_path)
        if sr != TARGET_SR:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=TARGET_SR)
        audio = audio.astype(np.float32)
        # Normalize amplitude
        max_val = np.max(np.abs(audio))
        if max_val > 0:
            audio = audio / max_val
    except Exception as e:
        print(f"[SER] Audio load error: {e}. Defaulting to neutral.")
        return "neutral"

    # ── Energy heuristic (always compute as fallback) ────────────────────────
    energy_label, energy_conf = energy_heuristic(audio)

    # ── Model inference ──────────────────────────────────────────────────────
    if ser_model is None or ser_feature_extractor is None:
        print("[SER] No model loaded — using energy heuristic.")
        return energy_label

    try:
        # Clip or pad to MAX_SAMPLES
        if len(audio) > MAX_SAMPLES:
            audio_input = audio[:MAX_SAMPLES]
        else:
            audio_input = np.pad(audio, (0, MAX_SAMPLES - len(audio)))

        inputs = ser_feature_extractor(
            audio_input,
            sampling_rate=TARGET_SR,
            return_tensors="pt",
            padding=False,
        )

        with torch.no_grad():
            outputs = ser_model(input_values=inputs["input_values"])

        logits = outputs.logits[0]
        probs  = torch.softmax(logits, dim=-1)
        confidence, predicted_id = probs.max(dim=-1)

        confidence   = confidence.item()
        predicted_id = predicted_id.item()
        model_label  = SER_ID_TO_LABEL.get(predicted_id, "neutral")

        print(f"[SER] Model → {model_label} (conf={confidence:.2f}) | Energy → {energy_label}")

        # ── Fusion: trust model if confidence is high enough ─────────────────
        if confidence >= SER_CONFIDENCE_THRESHOLD:
            return model_label
        else:
            print(f"[SER] Low confidence ({confidence:.2f}) — falling back to energy heuristic.")
            return energy_label

    except Exception as e:
        print(f"[SER] Model inference error: {e}. Using energy heuristic.")
        return energy_label
        
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7860, debug=True)



