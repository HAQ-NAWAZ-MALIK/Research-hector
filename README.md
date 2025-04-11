# Research-hector

# Research-hector

https://haq-nawaz-malik.github.io/dsp/

![image](https://github.com/user-attachments/assets/d4f7147e-5261-46e8-8598-87aded472844)


```
# app.py
import os
from google import genai
import gradio as gr
import PyPDF2
import numpy as np

# Try importing DSPy for chain-of-thought reasoning
try:
    import dspy
    HAS_DSPY = True
except ImportError:
    HAS_DSPY = False

#############################################
# Load Gemini API key from environment variable
#############################################
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("Please set the GEMINI_API_KEY environment variable.")

# Initialize the Gemini API client with the secret key
client = genai.Client(api_key=GEMINI_API_KEY)

#############################################
# Custom DSPy Prompt Signature Function
#############################################
def custom_dspy_prompt(text, mode="summarization"):
    """
    Returns a custom chain-of-thought prompt signature for DSPy.
    Modes:
      - "summarization": for summarizing a text chunk.
      - "overall": for combining chunk summaries.
    """
    if mode == "summarization":
        return (f"EffectiveDSPyCOT: Please provide a detailed, robust, and token-expansive summary using chain-of-thought reasoning. "
                f"Preserve context and key details. Text:\n\n{text}")
    elif mode == "overall":
        return (f"EffectiveDSPyCOT: Combine the following chunk summaries into an overall comprehensive summary. "
                f"Expand on details and maintain context with chain-of-thought reasoning. Summaries:\n\n{text}")
    else:
        return text

#############################################
# Fallback Using Gemini's generate_content Method
#############################################
def fallback_predict(prompt, system_msg="You are a helpful assistant."):
    """
    Uses the Gemini API (generate_content method) to generate content.
    """
    try:
        full_prompt = f"{system_msg}\n\n{prompt}"
        response = client.models.generate_content(
            model="gemini-2.0-flash",  # Adjust model name as needed.
            contents=full_prompt
        )
        return response.text
    except Exception as e:
        return f"[Gemini fallback error]: {str(e)}"

#############################################
# PDF Extraction and Improved Chunking
#############################################
def extract_text_from_pdf(pdf_path):
    """
    Extract text from all pages of a PDF file.
    """
    text = ""
    with open(pdf_path, "rb") as f:
        pdf_reader = PyPDF2.PdfReader(f)
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text

def chunk_text(text, chunk_size=2000, overlap=300):
    """
    Split the text into overlapping chunks.
    Larger chunk size and overlap help maintain context and expand token capacity.
    """
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += chunk_size - overlap  # Advance with overlap
    return chunks

#############################################
# Summarizing a Single Chunk with Custom DSPy / Gemini
#############################################
def summarize_chunk(chunk):
    """
    Summarize a text chunk using a custom DSPy chain-of-thought prompt.
    Falls back to Gemini if DSPy is not available or fails.
    """
    prompt = custom_dspy_prompt(chunk, mode="summarization")
    if HAS_DSPY:
        try:
            summary = dspy.predict(prompt)
        except Exception as e:
            summary = fallback_predict(prompt, system_msg="You are a helpful summarizer.")
    else:
        summary = fallback_predict(prompt, system_msg="You are a helpful summarizer.")
    return summary

#############################################
# Summarizing the Entire PDF
#############################################
def summarize_document(pdf_path):
    """
    Extract text from PDF, split it into overlapping chunks, summarize each chunk,
    and then combine the chunk summaries into an overall document summary.
    """
    text = extract_text_from_pdf(pdf_path)
    chunks = chunk_text(text)
    
    summaries = []
    for chunk in chunks:
        summary = summarize_chunk(chunk)
        summaries.append(summary)
    
    overall_prompt = custom_dspy_prompt("\n\n".join(summaries), mode="overall")
    if HAS_DSPY:
        try:
            overall_summary = dspy.predict(overall_prompt)
        except Exception as e:
            overall_summary = fallback_predict(overall_prompt, system_msg="You are a helpful assistant that summarizes documents.")
    else:
        overall_summary = fallback_predict(overall_prompt, system_msg="You are a helpful assistant that summarizes documents.")
    
    return overall_summary, summaries

#############################################
# Enhanced Gradio Interface with Better UI Aesthetics (Summarization Only)
#############################################
custom_css = """
<style>
    body { background-color: #f4f7f9; }
    .gradio-container { font-family: 'Arial', sans-serif; }
    h1, h2, h3 { color: #333333; }
    .tab-header { background-color: #ffffff; border-bottom: 2px solid #e0e0e0; }
    .gr-button { background-color: #4CAF50; color: white; }
    .gr-textbox { background-color: #ffffff; }
</style>
"""

with gr.Blocks(css=custom_css) as demo:
    gr.Markdown("## PDF Summarization Interface with Gemini API\n"
                "Upload a PDF document to get a robust, detailed summary using a custom DSPy chain-of-thought prompt.\n")
    
    with gr.Row():
        pdf_input_sum = gr.File(label="Upload PDF for Summarization", file_types=['.pdf'])
        summarize_button = gr.Button("Summarize Document")
    overall_summary_output = gr.Textbox(label="Overall Document Summary", lines=8)
    chunk_summaries_output = gr.Textbox(label="Chunk Summaries", lines=10)
    
    def process_and_summarize(pdf_file):
        if pdf_file is None:
            return "No file uploaded.", "No file uploaded."
        file_path = pdf_file.name
        overall, chunks = summarize_document(file_path)
        return overall, "\n\n".join(chunks)
    
    summarize_button.click(
        fn=process_and_summarize,
        inputs=pdf_input_sum,
        outputs=[overall_summary_output, chunk_summaries_output]
    )
    
demo.launch()
```





![image](https://github.com/user-attachments/assets/5e795cf3-a384-4bb6-a17d-d7df8fbb5113)



![image](https://github.com/user-attachments/assets/f9eb07fb-a698-4877-bdd0-a43c8785bab8)




![image](https://github.com/user-attachments/assets/9156cf9e-065b-4cf4-92b1-5feda8062d5b)

![image](https://github.com/user-attachments/assets/838e6a4d-a694-453f-9e3c-4cc6724f3e7a)


![image](https://github.com/user-attachments/assets/c6290b16-bec3-40ec-ab44-ca3562c17fa4)

















![image](https://github.com/user-attachments/assets/8af22d2f-919c-4589-bc8a-9ade713cbbc6)



