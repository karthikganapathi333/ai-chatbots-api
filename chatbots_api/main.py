# chatbots_api/main.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
from openai import OpenAI
from dotenv import load_dotenv
import sqlite3
from datetime import datetime
from typing import List, Dict

# load env and init client
load_dotenv()
OPENAI_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_KEY:
    raise RuntimeError("OPENAI_API_KEY missing in .env")

client = OpenAI(api_key=OPENAI_KEY)

app = FastAPI(title="AI Automation Studio Chatbots API")

# DB (SQLite)
DB_PATH = "chat_history.db"

def get_db():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db()
    cur = conn.cursor()

    cur.execute("""
    CREATE TABLE IF NOT EXISTS chats (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        title TEXT,
        created_at TEXT
    )
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS messages (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        chat_id INTEGER,
        sender TEXT,
        text TEXT,
        timestamp TEXT,
        FOREIGN KEY(chat_id) REFERENCES chats(id)
    )
    """)

    conn.commit()
    conn.close()

def save_message(chat_id: int, sender: str, text: str):
    conn = get_db()
    cur = conn.cursor()
    now = datetime.now().isoformat()
    cur.execute(
        "INSERT INTO messages (chat_id, sender, text, timestamp) VALUES (?, ?, ?, ?)",
        (chat_id, sender, text, now)
    )
    conn.commit()
    conn.close()

init_db()

# CORS - allow all for local dev (tighten on production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request models
class Message(BaseModel):
    chat_id: int
    message: str

class NewChatRequest(BaseModel):
    # optionally can include an initial message or name
    pass

# ---------------- Chat session endpoints ----------------

@app.post("/api/chat/new")
async def create_chat():
    conn = get_db()
    cur = conn.cursor()
    now = datetime.now().isoformat()
    cur.execute("INSERT INTO chats (title, created_at) VALUES (?, ?)", ("New Chat", now))
    chat_id = cur.lastrowid
    conn.commit()
    conn.close()
    return {"chat_id": chat_id}

@app.post("/api/chat/title")
async def generate_title(data: Dict):
    # data: { "chat_id": X, "message": "..." }
    chat_id = data.get("chat_id")
    user_message = data.get("message", "")
    if not chat_id:
        raise HTTPException(status_code=400, detail="chat_id required")

    # Use OpenAI to generate a concise title (max ~5 words)
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Generate a short, clean chat title (max 5 words) summarizing the user's message. Return only the title."},
                {"role": "user", "content": user_message}
            ],
            max_tokens=16,
        )
        title = response.choices[0].message.content.strip()
        if not title:
            title = "Conversation"
    except Exception as e:
        title = "Conversation"

    conn = get_db()
    cur = conn.cursor()
    cur.execute("UPDATE chats SET title = ? WHERE id = ?", (title, chat_id))
    conn.commit()
    conn.close()

    return {"title": title}

@app.get("/api/chat/list")
async def list_chats():
    conn = get_db()
    cur = conn.cursor()
    cur.execute("SELECT id, title, created_at FROM chats ORDER BY id DESC")
    rows = cur.fetchall()
    chats = [{"id": r["id"], "title": r["title"] or "New Chat", "created_at": r["created_at"]} for r in rows]
    conn.close()
    return {"chats": chats}

@app.get("/api/chat/{chat_id}/messages")
async def get_chat_messages(chat_id: int):
    conn = get_db()
    cur = conn.cursor()
    cur.execute("SELECT id, chat_id, sender, text, timestamp FROM messages WHERE chat_id = ? ORDER BY id ASC", (chat_id,))
    rows = cur.fetchall()
    msgs = [{"id": r["id"], "chat_id": r["chat_id"], "sender": r["sender"], "text": r["text"], "timestamp": r["timestamp"]} for r in rows]
    conn.close()
    return {"messages": msgs}

@app.delete("/api/chat/{chat_id}/delete")
async def delete_chat(chat_id: int):
    conn = get_db()
    cur = conn.cursor()
    cur.execute("DELETE FROM messages WHERE chat_id = ?", (chat_id,))
    cur.execute("DELETE FROM chats WHERE id = ?", (chat_id,))
    conn.commit()
    conn.close()
    return {"status": "deleted"}

# ---------------- AI helper (single function) ----------------
def call_chat_model(system_prompt: str, user_text: str) -> str:
    prompt = f"User message: {user_text}"
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]
    )
    ai_reply = response.choices[0].message.content.strip()
    return ai_reply

# ---------------- Chatbot endpoints (each saves messages) ----------------

# Real Estate
@app.post("/api/real-estate/chat")
async def chat_real_estate(msg: Message):
    chat_id = msg.chat_id
    user_msg = msg.message or ""
    # Save user
    save_message(chat_id, "user", user_msg)
    system_prompt = (
        "You are a professional and friendly real estate assistant. "
        "Provide clear, structured answers using short headings, bullet points, and short paragraphs. "
        "Keep tone helpful and respectful. Avoid markdown symbols like ** or ###."
    )
    ai_reply = call_chat_model(system_prompt, user_msg)
    save_message(chat_id, "bot", ai_reply)
    return {"reply": ai_reply}

# Student Mentor
@app.post("/api/student-mentor/chat")
async def chat_student_mentor(msg: Message):
    chat_id = msg.chat_id
    user_msg = msg.message or ""
    save_message(chat_id, "user", user_msg)
    system_prompt = (
        "You are a friendly and knowledgeable student mentor. "
        "Help with studies, exams, career guidance and productivity. "
        "Use bullet points and short paragraphs. Keep a warm, motivating tone."
    )
    ai_reply = call_chat_model(system_prompt, user_msg)
    save_message(chat_id, "bot", ai_reply)
    return {"reply": ai_reply}

# Fitness Coach
@app.post("/api/fitness-coach/chat")
async def chat_fitness_coach(msg: Message):
    chat_id = msg.chat_id
    user_msg = msg.message or ""
    save_message(chat_id, "user", user_msg)
    system_prompt = (
        "You are a certified fitness coach and nutrition expert. "
        "Give workout plans, diet advice, and practical tips. "
        "Format answers with steps and bullet points."
    )
    ai_reply = call_chat_model(system_prompt, user_msg)
    save_message(chat_id, "bot", ai_reply)
    return {"reply": ai_reply}

# Restaurant
@app.post("/api/restaurant/chat")
async def chat_restaurant(msg: Message):
    chat_id = msg.chat_id
    user_msg = msg.message or ""
    save_message(chat_id, "user", user_msg)
    system_prompt = (
        "You are a friendly restaurant and culinary assistant. "
        "Help with recipes, menu ideas, cooking instructions, and operations. "
        "Format with clear short sections and bullet points."
    )
    ai_reply = call_chat_model(system_prompt, user_msg)
    save_message(chat_id, "bot", ai_reply)
    return {"reply": ai_reply}

# Travel Planner
@app.post("/api/travel-planner/chat")
async def chat_travel_planner(msg: Message):
    chat_id = msg.chat_id
    user_msg = msg.message or ""
    save_message(chat_id, "user", user_msg)
    system_prompt = (
        "You are a helpful travel planner assistant. "
        "Suggest itineraries, budgets, places to visit and packing tips. "
        "Structure answers with headings, bullet points and short paragraphs."
    )
    ai_reply = call_chat_model(system_prompt, user_msg)
    save_message(chat_id, "bot", ai_reply)
    return {"reply": ai_reply}

# Health
@app.get("/health")
async def health():
    return {"status": "ok"}
