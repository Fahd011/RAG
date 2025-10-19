import fitz  # PyMuPDF
import pypandoc

pdf_path = "bill.pdf"

# Step 1: Extract text from PDF
doc = fitz.open(pdf_path)
text = ""
for page in doc:
    text += page.get_text("text")

# Step 2: Convert text to Markdown
try:
    md_text = pypandoc.convert_text(text, 'markdown', format='markdown')
except Exception as e:
    print("⚠️ Pandoc conversion failed, using raw text fallback.")
    md_text = text

# Step 3: Save as .md
with open("bill.md", "w", encoding="utf-8") as f:
    f.write(md_text)

print("✅ Successfully converted to bill.md")
