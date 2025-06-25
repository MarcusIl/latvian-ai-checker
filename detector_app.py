import tkinter as tk
from tkinter import scrolledtext
import joblib
import torch
import numpy as np
from transformers import BertTokenizer, BertModel
import nltk
import os
import re
import tkinter.font as tkFont

nltk.download("punkt")

# Tokenizers un modelis tiek ielādēts lokāli
tokenizer_path = os.path.abspath("lvbert_tokenizer")
model_path = os.path.abspath("lvbert_model")

tokenizer = BertTokenizer.from_pretrained(tokenizer_path, local_files_only=True, trust_remote_code=False)
bert_model = BertModel.from_pretrained(model_path, local_files_only=True)
bert_model.eval()

# Klasifiera ielāde
clf = joblib.load("latvian_ai_text_detector.pkl")

def split_sentences(text):
    return [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]

# BERT CLS embedding
def get_cls_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512, padding="max_length")
    with torch.no_grad():
        outputs = bert_model(**inputs)
    return outputs.last_hidden_state[:, 0, :].squeeze().numpy()

# MI atpazīšana
def detect_ai():
    text = input_text.get("1.0", tk.END).strip()
    sentences = split_sentences(text)
    output_text.config(state=tk.NORMAL)
    output_text.delete("1.0", tk.END)

    current_index = 1.0  

    for i, sent in enumerate(sentences):
        try:
            emb = get_cls_embedding(sent).reshape(1, -1)
            prob = clf.predict_proba(emb)[0][1]

            red = int(255 * prob)
            green = int(255 * (1 - prob))
            color_hex = f"#{red:02x}{green:02x}00"

            
            bg_color = blend_with_white(color_hex, opacity=0.7)

            tag_name = f"sentence_{i}"

            output_text.insert(f"{current_index}", sent)
            line, char = map(int, str(current_index).split('.'))
            end_char = char + len(sent)
            end_index = f"{line}.{end_char}"

            output_text.tag_add(tag_name, current_index, end_index)
            output_text.tag_config(tag_name, background=bg_color)

            output_text.insert(end_index, " ")
            current_index = f"{line}.{end_char + 1}"

        except Exception as e:
            output_text.insert(tk.END, f"[Error] {sent}\n")

    output_text.config(state=tk.DISABLED)


def blend_with_white(color_hex, opacity=0.5):
    r = int(color_hex[1:3], 16)
    g = int(color_hex[3:5], 16)
    b = int(color_hex[5:7], 16)
    
    r_blend = int(r * opacity + 255 * (1 - opacity))
    g_blend = int(g * opacity + 255 * (1 - opacity))
    b_blend = int(b * opacity + 255 * (1 - opacity))
    
    return f"#{r_blend:02x}{g_blend:02x}{b_blend:02x}"


def create_gradient_bar(canvas, width, height):
    for i in range(width):
        ratio = i / width
        red = int(255 * ratio)
        green = int(255 * (1 - ratio))
        color = f"#{red:02x}{green:02x}00"
        canvas.create_line(i, 0, i, height, fill=color)


window = tk.Tk()
window.title("Latvian AI Text Detector")
window.geometry("800x600")
window.tk.call("tk", "scaling", 1.0)  


label_frame = tk.Frame(window)
label_frame.pack(pady=5)

tk.Label(label_frame, text="0% AI", fg="green").pack(side=tk.LEFT)
gradient_canvas = tk.Canvas(label_frame, width=300, height=20, highlightthickness=0)
gradient_canvas.pack(side=tk.LEFT, padx=5)
tk.Label(label_frame, text="100% AI", fg="red").pack(side=tk.LEFT)

create_gradient_bar(gradient_canvas, 300, 20)


tk.Label(window, text="Enter text below:").pack()

font_style = tkFont.Font(family="Helvetica", size=14)

input_text = scrolledtext.ScrolledText(window, wrap=tk.WORD, height=10, font=font_style)
input_text.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)

tk.Button(window, text="Detect AI", command=detect_ai).pack(pady=10)

output_text = scrolledtext.ScrolledText(window, wrap=tk.WORD, height=15, font=font_style)
output_text.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)

window.mainloop()
