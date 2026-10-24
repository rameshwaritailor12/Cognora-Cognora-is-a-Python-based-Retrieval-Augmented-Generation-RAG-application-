
from pypdf import PdfReader


def extract_pages(pdf_path):
    reader = PdfReader(pdf_path)
    pages = []

    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""

        if text.strip():
            pages.append({
                "page": page_number,
                "text": text
            })

    return pages


def split_into_chunks(pages, chunk_size=1000, overlap=150):
    chunks = []

    for page in pages:
        text = page["text"]
        start = 0

        while start < len(text):
            end = min(start + chunk_size, len(text))
            chunk_text = text[start:end].strip()

            if chunk_text:
                chunks.append({
                    "page": page["page"],
                    "text": chunk_text
                })

            if end == len(text):
                break

            start = end - overlap

    return chunks


# Read the PDF
pages = extract_pages("sample file.pdf")

# Split the extracted text into chunks
chunks = split_into_chunks(pages)

# Display the results
print("Pages with text:", len(pages))
print("Total chunks created:", len(chunks))

for index, chunk in enumerate(chunks[:3], start=1):
    print(f"\n--- Chunk {index} | PDF Page {chunk['page']} ---")
    print(chunk["text"][:300])
