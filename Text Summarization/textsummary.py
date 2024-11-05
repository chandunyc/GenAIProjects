import torch
import gradio
# Use a pipeline as a high-level helper
from transformers import pipeline

text_summary = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6", torch_dtype=torch.bfloat16)

def summarize(text):
    output=text_summary(text)
    return output[0]['summary_text']

gradio.close_all()
demo=gradio.Interface(fn=summarize, inputs=[gradio.Textbox(label="Input your text to summarize",lines=6)],
                      outputs=[gradio.Textbox(label="Summarized Text",lines=4)],
                      title="Text Summarization Application",
                      description="This application is used to summarize your text.")

demo.launch()