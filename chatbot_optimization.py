
import os
import torch
import json
import pandas as pd
from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from nltk.translate.meteor_score import meteor_score
from nltk.tokenize import word_tokenize
from bert_score import score as bert_score
from sentence_transformers import SentenceTransformer, util
import nltk
from flask import Flask, request, jsonify
from datasets import load_dataset, Dataset
import logging
import random
import streamlit as st

# Suppress TensorFlow warnings
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow logging
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'  # Suppress Hugging Face symlinks warning

# Download required NLTK data
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)  # Required for sentence tokenization

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Create results folder for charts
os.makedirs("results", exist_ok=True)

# Step 1: Data Preparation
# Load Bitext Customer Support LLM Chatbot Training Dataset
try:
    dataset = load_dataset("bitext/Bitext-customer-support-llm-chatbot-training-dataset")
    logging.info("Bitext dataset loaded successfully.")
except Exception as e:
    logging.error(f"Error loading Bitext dataset: {e}")
    raise

# Preprocess dataset: Use 'instruction' as query and 'response' as ideal_response
def preprocess_dataset(dataset):
    """
    Preprocess the Bitext dataset to extract query-response pairs.
    
    Args:
        dataset: Hugging Face dataset object.
    
    Returns:
        list: List of dictionaries with 'query' and 'ideal_response' keys.
    """
    processed_data = []
    for item in dataset['train']:
        if item['instruction'] and item['response'] and isinstance(item['instruction'], str) and isinstance(item['response'], str):
            processed_data.append({
                "query": item['instruction'].strip(),
                "ideal_response": item['response'].strip()
            })
    logging.info(f"Processed {len(processed_data)} valid query-response pairs.")
    return processed_data[:1000]  # Limit to 1000 samples for CPU-based fine-tuning

DATASET = preprocess_dataset(dataset)

# Step 2: Initialize Models
try:
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token  # Set pad token to eos token
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    model.config.pad_token_id = tokenizer.eos_token_id  # Ensure model uses pad token
    logging.info("GPT-2 model and tokenizer loaded successfully.")
except Exception as e:
    logging.error(f"Error loading GPT-2: {e}")
    raise

try:
    sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
    logging.info("SentenceTransformer loaded successfully.")
except Exception as e:
    logging.error(f"Error loading SentenceTransformer: {e}")
    raise

# Step 3: Fine-Tune GPT-2
def fine_tune_gpt2(dataset):
    """
    Fine-tune GPT-2 on the customer support dataset.
    
    Args:
        dataset (list): List of query-response pairs.
    """
    try:
        texts = [f"Customer: {item['query']}\nSupport: {item['ideal_response']}{tokenizer.eos_token}" for item in dataset]
        hf_dataset = Dataset.from_dict({"text": texts})
        
        def tokenize_function(examples):
            encodings = tokenizer(
                examples["text"],
                padding="max_length",
                truncation=True,
                max_length=128,
                return_tensors="pt"
            )
            encodings["labels"] = encodings["input_ids"].clone()
            return encodings
        
        tokenized_dataset = hf_dataset.map(tokenize_function, batched=True)
        
        training_args = TrainingArguments(
            output_dir="./gpt2-finetuned",
            overwrite_output_dir=True,
            num_train_epochs=3,
            per_device_train_batch_size=4,
            save_steps=1000,
            save_total_limit=2,
            logging_dir='./logs',
            logging_steps=200,
            disable_tqdm=False,
        )
        
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_dataset,
        )
        
        trainer.train()
        logging.info("GPT-2 fine-tuning completed.")
        model.save_pretrained("./gpt2-finetuned")
        tokenizer.save_pretrained("./gpt2-finetuned")
        logging.info("Model and tokenizer saved.")
    except Exception as e:
        logging.error(f"Error during fine-tuning: {e}")
        raise

# Step 4: Chatbot Response Generation
def generate_response(query, max_length=100):
    """
    Generate a response to a customer query using fine-tuned GPT-2.
    
    Args:
        query (str): The customer query.
        max_length (int): Maximum length of the generated response.
    
    Returns:
        str: The generated response.
    """
    try:
        prompt = f"Customer: {query}\nSupport: "
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128,
            return_attention_mask=True
        )
        outputs = model.generate(
            inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_length=max_length,
            min_length=10,
            num_return_sequences=1,
            no_repeat_ngram_size=2,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            temperature=0.7,
            pad_token_id=tokenizer.eos_token_id
        )
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        if "Support: " in response:
            response = response.split("Support: ")[1].strip()
        else:
            response = "Sorry, I couldn't generate a response. Please try again."
        if not response or response.isspace():
            response = "Sorry, I couldn't generate a response. Please try again."
        logging.info(f"Generated response for query '{query}': {response}")
        return response
    except Exception as e:
        logging.error(f"Error generating response: {e}")
        return "Sorry, I couldn't process your request. Please try again."

# Step 5: Feedback Loop
def save_feedback(query, response, rating):
    """
    Save user feedback to a CSV file for future retraining.
    
    Args:
        query (str): The customer query.
        response (str): The generated response.
        rating (int): User rating (1-5).
    """
    feedback_data = {
        "query": [query],
        "response": [response],
        "rating": [rating],
        "timestamp": [pd.Timestamp.now()]
    }
    df = pd.DataFrame(feedback_data)
    feedback_file = "feedback.csv"
    if os.path.exists(feedback_file):
        df.to_csv(feedback_file, mode='a', header=False, index=False)
    else:
        df.to_csv(feedback_file, index=False)
    logging.info(f"Saved feedback for query '{query}' with rating {rating}")

# Step 6: Evaluation Metrics
def evaluate_response(generated_response, ideal_response):
    """
    Evaluate the generated response using multiple metrics.
    
    Args:
        generated_response (str): The chatbot's generated response.
        ideal_response (str): The ideal human-crafted response.
    
    Returns:
        dict: Dictionary containing BLEU, ROUGE, METEOR, BERTScore, Perplexity, and Cosine Similarity scores.
    """
    metrics = {}
    
    if not generated_response or generated_response.isspace():
        logging.warning("Generated response is empty or invalid. Assigning default metric values.")
        metrics = {
            'bleu': 0.0,
            'rouge1': 0.0,
            'rougeL': 0.0,
            'meteor': 0.0,
            'bertscore': 0.0,
            'cosine_similarity': 0.0,
            'perplexity': float('inf')
        }
        return metrics
    
    try:
        gen_tokens = word_tokenize(generated_response)
        ref_tokens = word_tokenize(ideal_response)
    except Exception as e:
        logging.error(f"Tokenization error: {e}")
        metrics = {
            'bleu': 0.0,
            'rouge1': 0.0,
            'rougeL': 0.0,
            'meteor': 0.0,
            'bertscore': 0.0,
            'cosine_similarity': 0.0,
            'perplexity': float('inf')
        }
        return metrics
    
    reference = [ref_tokens]
    candidate = gen_tokens
    metrics['bleu'] = sentence_bleu(reference, candidate, smoothing_function=SmoothingFunction().method1)
    
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
    rouge_scores = scorer.score(ideal_response, generated_response)
    metrics['rouge1'] = rouge_scores['rouge1'].fmeasure
    metrics['rougeL'] = rouge_scores['rougeL'].fmeasure
    
    metrics['meteor'] = meteor_score([ref_tokens], gen_tokens)
    
    P, R, F1 = bert_score([generated_response], [ideal_response], lang="en", verbose=False)
    metrics['bertscore'] = F1.item()
    
    embeddings = sentence_model.encode([generated_response, ideal_response], convert_to_tensor=True)
    metrics['cosine_similarity'] = util.cos_sim(embeddings[0], embeddings[1]).item()
    
    inputs = tokenizer(ideal_response, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs, labels=inputs["input_ids"])
        loss = outputs.loss
        metrics['perplexity'] = torch.exp(loss).item()
    
    logging.info(f"Evaluation metrics: {metrics}")
    return metrics

# Step 7: Metric Visualization
def save_metrics_chart(metrics, filename="metrics_chart.json"):
    """
    Save evaluation metrics as a Chart.js-compatible JSON file.
    
    Args:
        metrics (dict): Aggregated metrics.
        filename (str): Output filename in results folder.
    """
    chart_config = {
        "type": "bar",
        "data": {
            "labels": ["BLEU", "ROUGE-1", "ROUGE-L", "METEOR", "BERTScore", "Cosine Similarity"],
            "datasets": [{
                "label": "Metric Scores",
                "data": [
                    metrics['bleu'],
                    metrics['rouge1'],
                    metrics['rougeL'],
                    metrics['meteor'],
                    metrics['bertscore'],
                    metrics['cosine_similarity']
                ],
                "backgroundColor": ["#4e79a7", "#f28e2b", "#e15759", "#76b7b2", "#59a14f", "#edc948"],
                "borderColor": ["#3c5488", "#cc6b0e", "#b04040", "#5b8f8a", "#468239", "#d4a017"],
                "borderWidth": 1
            }]
        },
        "options": {
            "scales": {
                "y": {"beginAtZero": True, "max": 1, "title": {"display": True, "text": "Score"}},
                "x": {"title": {"display": True, "text": "Metrics"}}
            },
            "plugins": {"title": {"display": True, "text": "Chatbot Performance Metrics (Bitext Dataset)"}}
        }
    }
    with open(os.path.join("results", filename), 'w') as f:
        json.dump(chart_config, f, indent=2)
    logging.info(f"Saved metrics chart to results/{filename}")

# Step 8: Optimization Logic
def optimize_chatbot(dataset, num_iterations=3, sample_size=100):
    """
    Optimize the chatbot by evaluating responses and logging metrics.
    
    Args:
        dataset (list): List of query-response pairs.
        num_iterations (int): Number of evaluation iterations.
        sample_size (int): Number of samples to evaluate per iteration.
    
    Returns:
        dict: Aggregated metrics across all queries.
    """
    aggregated_metrics = {
        'bleu': [], 'rouge1': [], 'rougeL': [], 'meteor': [], 'bertscore': [],
        'cosine_similarity': [], 'perplexity': []
    }
    
    for i in range(num_iterations):
        sample = random.sample(dataset, min(sample_size, len(dataset)))
        for item in sample:
            query = item['query']
            ideal_response = item['ideal_response']
            generated_response = generate_response(query)
            metrics = evaluate_response(generated_response, ideal_response)
            
            for key in aggregated_metrics:
                aggregated_metrics[key].append(metrics[key])
        
        # Save metrics chart for each iteration
        avg_metrics_iter = {key: sum(values[-len(sample):]) / len(sample) for key, values in aggregated_metrics.items()}
        save_metrics_chart(avg_metrics_iter, f"metrics_iteration_{i+1}.json")
    
    avg_metrics = {key: sum(values) / len(values) for key, values in aggregated_metrics.items()}
    logging.info(f"Average metrics after {num_iterations} iterations: {avg_metrics}")
    save_metrics_chart(avg_metrics, "metrics_average.json")
    return avg_metrics

# Step 9: Flask Interface for Deployment
app = Flask(__name__)

@app.route('/chat', methods=['POST'])
def chat():
    """
    Endpoint to handle customer queries and return chatbot responses.
    
    Request JSON: {'query': 'customer query', 'rating': int (optional)}
    Response JSON: {'response': 'chatbot response', 'metrics': {evaluation metrics}}
    """
    try:
        data = request.get_json()
        query = data.get('query', '')
        rating = data.get('rating', None)
        if not query:
            return jsonify({'error': 'Query is required'}), 400
        
        generated_response = generate_response(query)
        
        if rating is not None and isinstance(rating, int) and 1 <= rating <= 5:
            save_feedback(query, generated_response, rating)
        
        best_metrics = None
        best_ideal_response = None
        for item in DATASET:
            if query.lower() in item['query'].lower():
                best_ideal_response = item['ideal_response']
                best_metrics = evaluate_response(generated_response, best_ideal_response)
                break
        
        if not best_metrics:
            best_metrics = {}
        
        return jsonify({
            'response': generated_response,
            'metrics': best_metrics
        })
    except Exception as e:
        logging.error(f"Error in /chat endpoint: {e}")
        return jsonify({'error': 'Internal server error'}), 500

# Step 10: Streamlit Frontend
def run_streamlit():
    """
    Run a Streamlit frontend for the chatbot.
    """
    st.set_page_config(page_title="Customer Support Chatbot", page_icon="🤖", layout="centered")
    st.title("Customer Support Chatbot")
    st.markdown("""
        Welcome to the Customer Support Chatbot! Enter your query below to get assistance.
        Powered by a fine-tuned GPT-2 model trained on the Bitext dataset.
    """)

    # Initialize session state for chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant" and "rating" in message:
                st.markdown(f"**Your Rating**: {message['rating']} / 5")

    # Input query
    query = st.chat_input("Enter your query (e.g., 'Where is my order?')")
    if query:
        # Add user query to chat history
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)

        # Generate and display response
        with st.chat_message("assistant"):
            generated_response = generate_response(query)
            st.markdown(generated_response)
            
            # Display metrics if available
            best_metrics = None
            for item in DATASET:
                if query.lower() in item['query'].lower():
                    best_metrics = evaluate_response(generated_response, item['ideal_response'])
                    break
            if best_metrics:
                st.markdown("**Evaluation Metrics**:")
                st.json(best_metrics)
            
            # Feedback input
            rating = st.slider("Rate this response (1-5)", 1, 5, key=f"rating_{len(st.session_state.messages)}")
            if st.button("Submit Rating", key=f"submit_{len(st.session_state.messages)}"):
                save_feedback(query, generated_response, rating)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": generated_response,
                    "rating": rating
                })
                st.success("Thank you for your feedback!")

# Step 11: Main Execution
if __name__ == "__main__":
    # Fine-tune the model
    logging.info("Starting GPT-2 fine-tuning...")
    fine_tune_gpt2(DATASET)
    
    # Run optimization
    logging.info("Starting chatbot optimization...")
    avg_metrics = optimize_chatbot(DATASET)
    print("Average Metrics:", avg_metrics)
    
    # Run Streamlit frontend
    logging.info("Starting Streamlit frontend...")
    run_streamlit()
