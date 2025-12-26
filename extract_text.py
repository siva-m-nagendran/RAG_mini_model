import pdfplumber

# Path to your PDF
pdf_path = "The-Jungle-Books-text.pdf"
output_txt = "/Users/siva-16986/Desktop/Projects/RAG_mini_model/The-Jungle-Books-text.txt"

# Optional: function to clean text
def clean_text(text):
    text = text.lower()
    text = text.replace("\n", " ")
    text = text.replace("\t", " ")
    text = " ".join(text.split())
    return text

# Extract and clean text from PDF
all_text = ""
with pdfplumber.open(pdf_path) as pdf:
    for i, page in enumerate(pdf.pages):
        raw_text = page.extract_text()
        if raw_text:
            cleaned_text = clean_text(raw_text)
            all_text += cleaned_text + "\n"
            print(f"Page {i+1} words: {len(cleaned_text.split())}")
        else:
            print(f"Page {i+1} empty or non-text")

# Save to TXT
with open(output_txt, "w", encoding="utf-8") as f:
    f.write(all_text)

print("PDF text extracted and cleaned successfully.")