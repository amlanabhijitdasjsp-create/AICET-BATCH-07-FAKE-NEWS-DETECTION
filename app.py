import streamlit as st
import tkinter as tk
from tkinter import messagebox, ttk
import joblib
import re
import math

# ==============================
# LOAD MODEL & VECTORIZER
# ==============================

model = joblib.load("model.joblib")
vectorizer = joblib.load("vectorizer.joblib")

# ==============================
# TEXT CLEANING
# ==============================

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z ]', '', text)
    return text

# ==============================
# PREDICTION FUNCTION
# ==============================

def check_news():
    news = text_box.get("1.0", tk.END).strip()

    if news == "":
        messagebox.showwarning("Warning", "Please enter news text!")
        return

    news_clean = clean_text(news)
    news_vector = vectorizer.transform([news_clean])

    prediction = model.predict(news_vector)[0]

    # Convert decision score to probability
    score = model.decision_function(news_vector)[0]
    confidence = 1 / (1 + math.exp(-score))
    confidence = round(confidence * 100, 2)

    # Update progress bar
    progress["value"] = confidence

    if prediction == "REAL":
        result_label.config(
            text=f"✅ REAL NEWS",
            fg="#2ecc71"
        )
    else:
        result_label.config(
            text=f"⚠️ FAKE NEWS",
            fg="#e74c3c"
        )

    confidence_label.config(text=f"Confidence: {confidence}%")

# ==============================
# UI DESIGN
# ==============================

root = tk.Tk()
root.title("Fake News Detector - AI System")
root.geometry("750x520")
root.configure(bg="#1e272e")
root.resizable(False, False)

# Title
title = tk.Label(
    root,
    text="📰 AI Fake News Detector",
    font=("Helvetica", 22, "bold"),
    bg="#1e272e",
    fg="white"
)
title.pack(pady=20)

# Frame Container
frame = tk.Frame(root, bg="#2f3640", bd=0)
frame.pack(pady=10, padx=30, fill="both", expand=True)

# Instruction
instruction = tk.Label(
    frame,
    text="Enter News Article:",
    font=("Helvetica", 13),
    bg="#2f3640",
    fg="white"
)
instruction.pack(pady=5)

# Text Box
text_box = tk.Text(
    frame,
    height=8,
    width=75,
    font=("Helvetica", 11),
    bg="#dcdde1",
    fg="black",
    relief="flat"
)
text_box.pack(pady=10)

# Check Button
check_btn = tk.Button(
    frame,
    text="Analyze News",
    font=("Helvetica", 13, "bold"),
    bg="#00a8ff",
    fg="white",
    activebackground="#0097e6",
    relief="flat",
    padx=20,
    pady=8,
    command=check_news
)
check_btn.pack(pady=15)

# Result Label
result_label = tk.Label(
    frame,
    text="Result will appear here",
    font=("Helvetica", 16, "bold"),
    bg="#2f3640",
    fg="white"
)
result_label.pack(pady=10)

# Confidence Label
confidence_label = tk.Label(
    frame,
    text="Confidence: 0%",
    font=("Helvetica", 12),
    bg="#2f3640",
    fg="white"
)
confidence_label.pack()

# Progress Bar
progress = ttk.Progressbar(
    frame,
    orient="horizontal",
    length=400,
    mode="determinate"
)
progress.pack(pady=10)

root.mainloop()