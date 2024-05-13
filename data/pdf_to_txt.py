from unstructured.partition.pdf import partition_pdf
from spire.pdf.common import *
from spire.pdf import *
import os
import json
from PyPDF2 import PdfReader, PdfWriter

from marker.marker.convert import convert_single_pdf
from marker.marker.models import load_all_models

def pdf_splitter(pdf_path: str, output_dir: str, num_of_files: int) -> None:
    if not os.path.exists(pdf_path):
        print("[INFO] The specified PDF file does not exist.")
        return
    name_of_file = os.path.splitext(os.path.basename(pdf_path))[0]
    if not os.path.exists(f"{output_dir}/pdf/{name_of_file}"):
        os.makedirs(f"{output_dir}/pdf/{name_of_file}")
    if not os.path.exists(f"{output_dir}/txt/{name_of_file}"):
        os.makedirs(f"{output_dir}/txt/{name_of_file}")

    reader = PdfReader(pdf_path)
    num_of_pages = len(reader.pages)
    num_of_pages_per_file = num_of_pages // num_of_files
    remainder = num_of_pages % num_of_files

    start_page = 0
    for i in range(num_of_files):
        writer = PdfWriter()

        # Adjust end_page for each segment
        end_page = start_page + num_of_pages_per_file - 1
        if i < remainder:
            end_page += 1

        # Add pages to each new document
        for j in range(start_page, min(end_page + 1, num_of_pages)):
            writer.add_page(reader.pages[j])

        # Save each new split PDF
        with open(f"{output_dir}/pdf/{name_of_file}/{name_of_file}_part_{i + 1}.pdf", 'wb') as f:
            writer.write(f)

        start_page = end_page + 1

def pdf_to_txt(name_of_file: str, pdf_dir: str="./tmp/pdf") -> None:
    dir_path = f"{pdf_dir}/{name_of_file}"
    if not os.path.exists(dir_path):
        print("[INFO] PDF Directory does not exist.")
    else:
        num_of_files = len([entry for entry in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, entry))])

    for i in range(num_of_files):
        print(f"[INFO] Converting {name_of_file},pdf ...")

        elements = partition_pdf(f"{pdf_dir}/{name_of_file}/{name_of_file}_part_{i + 1}.pdf", languages=["eng"])
        with open(f"{pdf_dir}/txt/{name_of_file}/{name_of_file}_part_{i + 1}.txt", "w") as file:
            file.write("\n".join([str(e) for e in elements]))
        
        print("[INFO] Completed.\n" + "="*30)

def pdf_to_md(name_of_file: str, tmp_path: str="./tmp") -> None:
    tmp_pdf_dir = f"{tmp_path}/pdf/{name_of_file}"
    tmp_md_dir = f"{tmp_path}/md/{name_of_file}"
    if not os.path.exists(tmp_pdf_dir):
        print("[INFO] PDF Directory does not exist.")
    else:
        num_of_files = len([entry for entry in os.listdir(tmp_pdf_dir) if os.path.isfile(os.path.join(tmp_pdf_dir, entry))])
    
    model_lst = load_all_models()

    for i in range(num_of_files):
        print(f"[INFO] Converting {name_of_file}_part_{i+1}.pdf ...")
        if not os.path.exists(f"{tmp_md_dir}/{name_of_file}_part_{i+1}.md"):
            with open(f"{tmp_md_dir}/{name_of_file}_part_{i+1}.md", 'w') as file:
                pass

        pdf_file_path = f"{tmp_pdf_dir}/{name_of_file}_part_{i+1}.pdf"
        md_file_path = f"{tmp_md_dir}/{name_of_file}_part_{i+1}.md"
        full_text, out_meta = convert_single_pdf(pdf_file_path, model_lst, max_pages=60, parallel_factor=1)
        with open(md_file_path, "w+", encoding='utf-8') as f:
            f.write(full_text)
        
        meta_json_file_path = f"{tmp_md_dir}/meta_json/{name_of_file}_part_{i+1}.json"
        with open(meta_json_file_path, "w+") as f:
            f.write(json.dumps(out_meta, indent=4))

        print("[INFO] Completed.\n" + "="*30)

if __name__ == "__main__":
    #  python convert_single.py ../../knowledge/pdf/dive_into_deep_learning.pdf ../../knowledge/md/dive_into_deep_learning.md --max_pages 10
    # pdf_parser("./knowledge/pdf/dive_into_deep_learning.pdf", "./tmp", 20)
    pdf_to_md(name_of_file="dive_into_deep_learning")