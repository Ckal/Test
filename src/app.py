from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import gradio as gr
from langfuse import Langfuse
from langfuse.decorators import observe, langfuse_context
import os 
import faiss
import pandas as pd
from sentence_transformers import SentenceTransformer
import datetime

# Initialize Langfuse
os.environ["LANGFUSE_PUBLIC_KEY"] = "pk-lf-9f2c32d2-266f-421d-9b87-51377f0a268c"
os.environ["LANGFUSE_SECRET_KEY"] = "sk-lf-229e10c5-6210-4a4b-a432-0f17bc66e56c"
os.environ["LANGFUSE_HOST"] = "https://chris4k-langfuse-template-space.hf.space"  # 🇪🇺 EU region

langfuse = Langfuse()

# Load the Llama model
model_name = "meta-llama/Llama-3.2-3B-Instruct"  # Replace with the exact model path
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, device_map=None, torch_dtype=torch.float32)

# Load FAISS and Embeddings
embedder = SentenceTransformer('distiluse-base-multilingual-cased')
url = 'https://www.bofrost.de/datafeed/DE/products.csv'
data = pd.read_csv(url, sep='|')

# Clean and process the dataset
columns_to_keep = ['ID', 'Name', 'Description', 'Price', 'ProductCategory', 'Grammage', 'BasePriceText', 'Rating', 'RatingCount', 'Ingredients', 'CreationDate', 'Keywords', 'Brand']
data_cleaned = data[columns_to_keep]
data_cleaned['Description'] = data_cleaned['Description'].str.replace(r'[^\w\s.,;:\'"/?!€$%&()\[\]{}<>|=+\\-]', ' ', regex=True)
data_cleaned['combined_text'] = data_cleaned.apply(lambda row: ' '.join([str(row[col]) for col in ['Name', 'Description', 'Keywords'] if pd.notnull(row[col])]), axis=1)

# Generate and add embeddings
embeddings = embedder.encode(data_cleaned['combined_text'].tolist(), convert_to_tensor=True).cpu().detach().numpy()
faiss_index = faiss.IndexFlatL2(embeddings.shape[1])
faiss_index.add(embeddings)

# Helper function for searching products
def search_products(query, top_k=7):
    query_embedding = embedder.encode([query], convert_to_tensor=True).cpu().detach().numpy()
    distances, indices = faiss_index.search(query_embedding, top_k)
    return data_cleaned.iloc[indices[0]].to_dict(orient='records')

# Prompt construction functions
def construct_system_prompt(context):
    return f"You are a friendly bot specializing in Bofrost products. Return comprehensive German answers. Always add product IDs. Use the following product descriptions:\n\n{context}\n\n"

def construct_prompt(user_input, context, chat_history, max_history_turns=1):
    system_message = construct_system_prompt(context)
    prompt = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{system_message}<|eot_id|>"
    for user_msg, assistant_msg in chat_history[-max_history_turns:]:
        prompt += f"<|start_header_id|>user<|end_header_id|>\n\n{user_msg}<|eot_id|>"
        prompt += f"<|start_header_id|>assistant<|end_header_id|>\n\n{assistant_msg}<|eot_id|>"
    prompt += f"<|start_header_id|>user<|end_header_id|>\n\n{user_input}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    return prompt

# Main function to interact with the model
@observe()
def chat_with_model(user_input, chat_history=[]):
    # Start trace for the entire chat process
    trace = langfuse.trace(
        name="ai-chat-execution",
        user_id="user_12345",
        metadata={"email": "user@example.com"},
        tags=["chat", "product-query"],
        release="v1.0.0"
    )

    # Span for product search
    retrieval_span = trace.span(
        name="product-retrieval",
        metadata={"source": "faiss-index"},
        input={"query": user_input}
    )

    # Search for products
    search_results = search_products(user_input)
    if search_results:
        context = "Product Context:\n" + "\n".join(
            [f"Produkt ID: {p['ID']}, Name: {p['Name']}, Beschreibung: {p['Description']}, Preis: {p['Price']}€" for p in search_results]
        )
    else:
        context = "Das weiß ich nicht."
    
    # End product search span with results
    retrieval_span.end(
        output={"search_results": search_results},
        status_message=f"Found {len(search_results)} products"
    )

    # Update trace with search context
    langfuse_context.update_current_observation(
        input={"query": user_input},
        output={"context": context},
        metadata={"search_results_found": len(search_results)}
    )

    # Generate prompt for Llama model
    prompt = construct_prompt(user_input, context, chat_history)
    input_ids = tokenizer.encode(prompt, return_tensors="pt", truncation=True, max_length=4096)
    
    # Span for AI generation
    generation_span = trace.span(
        name="ai-response-generation",
        metadata={"model": "Llama-3.2-3B-Instruct"},
        input={"prompt": prompt}
    )
    
    outputs = model.generate(input_ids, max_new_tokens=1200, do_sample=True, top_k=50, temperature=0.7)
   # response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Decode and clean the response to remove unwanted repetitions of "assistant"
    response = tokenizer.decode(outputs[0][input_ids.shape[-1]:], skip_special_tokens=True).strip()
    
    # Remove potential repeated assistant text
    response = response.replace("<|assistant|>", "").strip()
    
    # End model generation span
    generation_span.end(
        output={"response": response},
        status_message="AI response generated"
    )

    # Update Langfuse context with usage details
    langfuse_context.update_current_observation(
        usage_details={
            "input_tokens": len(input_ids[0]),
            "output_tokens": len(response)
        }
    )

    # Append the response to the chat history
    chat_history.append((user_input, response))

    # Update trace final output
    trace.update(
        metadata={"final_status": "completed"},
        output={"summary": response}
    )

    # Return the response
    return response, chat_history

# Gradio interface
def gradio_interface(user_input, history):
    response, updated_history = chat_with_model(user_input, history)
    return response, updated_history

with gr.Blocks() as demo:
    gr.Markdown("# 🦙 Llama Instruct Chat with LangFuse & Faiss Integration")
    user_input = gr.Textbox(label="Your Message", lines=2)
    submit_btn = gr.Button("Send")
    chat_history = gr.State([])
    chat_display = gr.Textbox(label="Chat Response", lines=10, interactive=False)
    submit_btn.click(gradio_interface, inputs=[user_input, chat_history], outputs=[chat_display, chat_history])

demo.launch(debug=True)
