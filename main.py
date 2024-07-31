import fitz
import os
import pathlib
import re
import pandas as pd
from PyPDF2 import PdfWriter, PdfReader
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch


def detect_toc_page(pages_text, model, tokenizer):
    toc_pages = []
    for i, text in enumerate(pages_text):
        prompt = f"Is the following text a table of contents?\n\n{text}\n\nAnswer:"
        inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
        with torch.no_grad():
            outputs = model.generate(**inputs, max_length=512)
        
        response = tokenizer.decode(outputs[0])

        if "yes" in response.lower():
            toc_pages.append(i)
            break
        print(toc_pages)
    return toc_pages

def extract_text_from_pdf(pdf_path):
    pdf_document = fitz.open(pdf_path)
    pages_text = []
    for page_num in range(pdf_document.page_count):
        page = pdf_document.load_page(page_num)
        text = page.get_text("text")
        pages_text.append(text)
    return pages_text

def parse_toc_text(toc_text):
    toc = []
    lines = toc_text.split('\n')
    for line in lines:
        match = re.match(r'(.*)\s+(\d+)', line)
        if match:
            title = match.group(1)
            page = int(match.group(2))
            toc.append((title, page))
    return toc

def add_toc_to_pdf(pdf_path, toc):
    pdf_reader = PdfReader(pdf_path)
    pdf_writer = PdfWriter()

    for page_num in range(len(pdf_reader.pages)):
        pdf_writer.add_page(pdf_reader.pages[page_num])

    for title, page in toc:
        pdf_writer.add_outline_item(title, page - 1)

    with open(pdf_path, "wb") as f_out:
        pdf_writer.write(f_out)

    print(f"TOC added to {pdf_path}")

def main(toc_file):
    toc_data = pd.read_csv(toc_file)
    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
    model = AutoModelForSequenceClassification.from_pretrained("mistralai/Mistral-7B-v0.1")
    model.eval()

    for index, row in toc_data.iterrows():
        pdf_path = row['path']
        toc_type = row['toc_type']
        toc_start_page_num = row['toc_start_page_num']

        if toc_type == 'doc_toc' and toc_start_page_num == -1:
            if not os.path.exists(pdf_path):
                pdf_path = f"{pdf_path}.pdf"
            pages_text = extract_text_from_pdf(pdf_path)
            toc_pages = detect_toc_page(pages_text, model, tokenizer)
            if len(toc_pages) > 0:
                toc = []
                for toc_num_page in toc_pages:
                    toc_text = pages_text[toc_num_page]
                    toc += parse_toc_text(toc_text)
                add_toc_to_pdf(pdf_path, toc)

if __name__ == "__main__":
    main("toc.txt")
