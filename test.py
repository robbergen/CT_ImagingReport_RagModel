import os

text_file = "./test.txt"
text = "Hello world"

# ensure directory exists
os.makedirs(os.path.dirname(text_file), exist_ok=True)

with open(text_file, "w", encoding="utf-8") as f:
    f.write(text)
    f.flush()
    os.fsync(f.fileno())

print("Wrote file successfully:", text_file, flush=True)
