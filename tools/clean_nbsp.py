import pathlib

root = pathlib.Path(__file__).parent

total = 0
cleaned = 0
no_issue = 0

for pyfile in root.rglob("*.py"):
    total += 1
    text = pyfile.read_text(encoding="utf-8")
    if "\u00a0" in text:
        clean = text.replace("\u00a0", " ")
        pyfile.write_text(clean, encoding="utf-8")
        print(f"✅ cleaned: {pyfile}")
        cleaned += 1
    else:
        print(f"👌 no issue: {pyfile}")
        no_issue += 1

print("\n--- SUMMARY ---")
print(f"📂 Total files scanned: {total}")
print(f"✅ Cleaned files: {cleaned}")
print(f"👌 No issues: {no_issue}")
