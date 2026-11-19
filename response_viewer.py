import json, sys
import tkinter as tk
from tkinter import ttk, scrolledtext

def load(path):
    with open(path, encoding="utf-8") as f:
        text = f.read()
    try:
        data = json.loads(text)
        return data if isinstance(data, list) else [data]
    except json.JSONDecodeError:
        return[json.loads(line) for line in text.splitlines() if line.strip()]
    
records = load(sys.argv[1])
KEY = sys.argv[2] if len(sys.argv) > 2 else "output" 

i = 0

root = tk.Tk()
root.title("Response viewer")

meta = ttk.Treeview(root, columns=("value",), height=8)
meta.heading("#0", text="field")
meta.heading("value", text="value")
meta.pack(fill="x")

body = scrolledtext.ScrolledText(root, wrap="word", font=("monospace", 11))
body.pack(fill="both", expand=True)

nav = tk.Frame(root)
nav.pack(fill="x")
counter = tk.Label(nav)

def show():
    rec = records[i] 
    meta.delete(*meta.get_children())
    for k, v in rec.items():
        if k != KEY: 
            meta.insert("", "end", text=k, values=(v if isinstance(v, str) else json.dumps(v),))
    resp = rec.get(KEY, f"(no '{KEY}' field)")
    body.delete("1.0", "end")
    body.insert("1.0", resp if isinstance(resp, str) else json.dumps(resp, indent=2))
    counter.config(text=f"{i + 1} / {len(records)}")
    
def step(d):
    global i 
    i = (i +d) % len(records)
    show()

tk.Button(nav, text="< Prev", command=lambda: step(-1)).pack  (side="left")
counter.pack(side="left", expand=True)
tk.Button(nav, text="Next >", command=lambda: step(1)).pack(side="right")
root.bind("<Left>", lambda e: step(-1))
root.bind("<Right>", lambda e: step(1))

show()
root.mainloop()