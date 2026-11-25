import json, sys
import tkinter as tk
from tkinter import ttk, scrolledtext
from tkinter import simpledialog

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
root.geometry("900x700")

main = tk.Frame(root)
main.pack(fill="both", expand=True)

meta = ttk.Treeview(main, columns=("value",))
meta.heading("#0", text="field")
meta.heading("value", text="value")
meta.column("#0", width=140)
meta.column("value", width=160)
meta.pack(side="left", fill="y")

body = scrolledtext.ScrolledText(main, wrap="word", font=("monospace", 11))
body.pack(side="left", fill="both", expand=True)

nav = tk.Frame(root)
nav.pack(fill="x")
counter = tk.Label(nav)

def show():
    rec = records[i]
    meta.delete(*meta.get_children())
    for k, v in rec.items():
        if k == KEY:
            continue
        if isinstance(v, (dict, list)):
            parent = meta.insert("", "end", text=k, open=True)
            items = v.items() if isinstance(v, dict) else enumerate(v)
            for sk, sv in items:
                meta.insert(parent, "end", text=str(sk),
                            values=(sv if isinstance(sv, str) else json.dumps(sv),))
        else:
            meta.insert("", "end", text=k, values=(v if isinstance(v, str) else json.dumps(v),))
    resp = rec.get(KEY, f"(no '{KEY}' field)")
    body.delete("1.0", "end")
    body.insert("1.0", resp if isinstance(resp, str) else json.dumps(resp, indent=2))
    counter.config(text=f"{i + 1} / {len(records)}")
    highlight()
    
def step(d):
    global i 
    i = (i +d) % len(records)
    show()
    
body.tag_config("hit", background="yellow")   # what a match looks like
query = ""                                     # current search term

def highlight():
    body.tag_remove("hit", "1.0", "end")       # clear old highlights
    n, start = 0, "1.0"
    while query:
        pos = body.search(query, start, stopindex="end", nocase=True)
        if not pos:
            break
        end = f"{pos}+{len(query)}c"           # match start + N characters
        body.tag_add("hit", pos, end)
        start, n = end, n + 1
    if n:
        body.see(body.tag_nextrange("hit", "1.0")[0])   # scroll to first match
    root.title(f"Response viewer — {n} matches" if query else "Response viewer")

def find(event=None):
    global query
    q = simpledialog.askstring("Find", "Search for:", initialvalue=query, parent=root)
    if q is not None:                          # None means Cancel was pressed
        query = q
        highlight()

root.bind("<Control-f>", find)

tk.Button(nav, text="< Prev", command=lambda: step(-1)).pack  (side="left")
counter.pack(side="left", expand=True)
tk.Button(nav, text="Next >", command=lambda: step(1)).pack(side="right")
root.bind("<Left>", lambda e: step(-1))
root.bind("<Right>", lambda e: step(1))

show()
root.mainloop()