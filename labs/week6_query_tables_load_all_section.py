from llmsherpa.readers import LayoutPDFReader
from IPython.display import display, HTML
from llama_index.llms.ollama import Ollama
from llama_index.core import VectorStoreIndex
from llama_index.core import Document, ServiceContext, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings

# Source: https://medium.com/@jitsins/query-complex-pdfs-in-natural-language-with-llmsherpa-ollama-llama3-8b-13b4782243de
# To install:
# 1. run https://stackoverflow.com/questions/52805115/certificate-verify-failed-unable-to-get-local-issuer-certificate
# 2. install and run ollama:
# ollama pull llama3
# ollama run llama3
# 3. Install docker and run:
# docker pull ghcr.io/nlmatics/nlm-ingestor:latest
# docker run -p 5010:5001 ghcr.io/nlmatics/nlm-ingestor:latest
# This will expose the api link “http://localhost:5010/api/parseDocument?renderFormat=all” for you to utilize in your code.

# Initialize LLm
llm = Ollama(model="tinyllama", request_timeout=6000.0)

llmsherpa_api_url = "http://localhost:5010/api/parseDocument?renderFormat=all"
pdf_url = "https://abc.xyz/assets/91/b3/3f9213d14ce3ae27e1038e01a0e0/2024q1-alphabet-earnings-release-pdf.pdf"
pdf_reader = LayoutPDFReader(llmsherpa_api_url)

# Read PDF
doc = pdf_reader.read_pdf(pdf_url)

# a. Load all sections to avoid hard-coding a section with tables
all_sections_html = ""
for section in doc.sections():
    all_sections_html += section.to_html(include_children=True, recurse=True)


question1 = "What was Google's operating margin for 2024?"
resp1 = llm.complete(
    f"read this table and answer question: {question1}:\n{all_sections_html}")
print(resp1.text)

question2 = "What % Net income is of the Revenues?"
resp2 = llm.complete(
    f"read this table and answer question: {question2}:\n{all_sections_html}")
print(resp2.text)

# b. Test the capabilities of reasoning with table data: can it sum up numbers or do some other calculation?
question3 = "What is the sum of Google Services and Google Cloud revenues for Q1 2024?"
resp3 = llm.complete(
    f"read this table and answer question: {question3}:\n{all_sections_html}")
print(resp3.text)

