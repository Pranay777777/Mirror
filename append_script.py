import os

source_file = "append_temp.txt"
dest_file = r"C:\Users\aim4g\.gemini\antigravity\brain\fb50b95e-af84-4e34-b911-83377ab1cb8d\walkthrough.md"

with open(source_file, "r", encoding="utf-8") as f_in:
    content = f_in.read()

with open(dest_file, "a", encoding="utf-8") as f_out:
    f_out.write("\n" + content + "\n")

if os.path.exists(source_file):
    os.remove(source_file)
    
print("Walkthrough artifact successfully appended.")
