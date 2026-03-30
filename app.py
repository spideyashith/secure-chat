import streamlit as st
import torch
import numpy as np
import time
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(page_title="SecureChat AI Prototype", page_icon="🛡️", layout="centered")

# --- 2. PROFESSIONAL CSS INJECTION ---
# This hides default Streamlit branding and creates custom UI cards
st.markdown("""
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    .main-header {
        text-align: center;
        font-family: 'Helvetica Neue', sans-serif;
        color: #1E3A8A;
    }
    
    .result-card {
        padding: 20px;
        border-radius: 12px;
        color: white;
        text-align: center;
        font-family: 'Helvetica Neue', sans-serif;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin-top: 10px;
        margin-bottom: 20px;
    }
    
    .result-title { font-size: 24px; font-weight: 800; margin-bottom: 5px; }
    .result-score { font-size: 18px; opacity: 0.9; }
    
    .bg-neutral { background: linear-gradient(135deg, #10B981, #059669); }
    .bg-abusive { background: linear-gradient(135deg, #F59E0B, #D97706); }
    .bg-threat { background: linear-gradient(135deg, #EF4444, #DC2626); }
    </style>
""", unsafe_allow_html=True)

# --- 3. LOAD AI MODEL ---
@st.cache_resource(show_spinner=False)
def load_custom_model():
    model_path = "./securechat_model_v2_final"
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return tokenizer, model, device

with st.spinner("Initializing AI Engine..."):
    tokenizer, model, device = load_custom_model()

labels = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

# --- 4. DASHBOARD UI ---
st.markdown("<h1 class='main-header'>🛡️ SecureChat Inference Engine</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #6B7280; margin-bottom: 30px;'>On-device NLP threat and abuse detection prototype.</p>", unsafe_allow_html=True)

# Create a clean container for the input
with st.container(border=True):
    user_input = st.text_input("Intercepted Message Content:", placeholder="Type a message to simulate an incoming notification...", label_visibility="collapsed")
    analyze_btn = st.button("Run Security Analysis", use_container_width=True, type="primary")

# --- 5. LOGIC & LOADING EFFECTS ---
if analyze_btn:
    if not user_input.strip():
        st.error("Error: Message payload cannot be empty.")
    else:
        # Professional loading sequence
        progress_text = "Parsing text and removing noise..."
        my_bar = st.progress(0, text=progress_text)
        
        time.sleep(0.3)
        my_bar.progress(40, text="Passing tokens to DistilBERT ensemble...")
        
        # 1. Prepare text
        inputs = tokenizer(user_input, return_tensors="pt", truncation=True, max_length=128, padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        time.sleep(0.3)
        my_bar.progress(80, text="Applying weighted decision thresholds...")
        
        # 2. Run model
        with torch.no_grad():
            outputs = model(**inputs)
            
        # 3. Convert to percentages
        logits = outputs.logits.cpu().numpy()[0]
        sigmoid = lambda x: 1 / (1 + np.exp(-x))
        probs = sigmoid(logits)
        results = dict(zip(labels, probs))
        
        my_bar.progress(100, text="Analysis Complete.")
        time.sleep(0.2)
        my_bar.empty() # Remove the progress bar
        
        # --- 6. DISPLAY CUSTOM RESULTS ---
        st.subheader("System Verdict")
        
        if results['threat'] > 0.5:
            st.markdown(f"""
                <div class="result-card bg-threat">
                    <div class="result-title">⚠️ THREAT DETECTED</div>
                    <div class="result-score">Confidence Score: {results['threat']:.4f}</div>
                </div>
            """, unsafe_allow_html=True)
            
        elif results['toxic'] > 0.5 or results['insult'] > 0.5 or results['severe_toxic'] > 0.5:
            max_abusive = max(results['toxic'], results['insult'], results['severe_toxic'])
            st.markdown(f"""
                <div class="result-card bg-abusive">
                    <div class="result-title">🛑 ABUSIVE CONTENT</div>
                    <div class="result-score">Confidence Score: {max_abusive:.4f}</div>
                </div>
            """, unsafe_allow_html=True)
            
        else:
            st.markdown(f"""
                <div class="result-card bg-neutral">
                    <div class="result-title">✅ MESSAGE NEUTRAL</div>
                    <div class="result-score">No threats detected.</div>
                </div>
            """, unsafe_allow_html=True)
            
        # 7. Telemetry Data (Expander)
        with st.expander("View Model Telemetry (Raw Logits)"):
            st.code(
                f"Toxic:        {results['toxic']:.4f}\n"
                f"Severe Toxic: {results['severe_toxic']:.4f}\n"
                f"Obscene:      {results['obscene']:.4f}\n"
                f"Threat:       {results['threat']:.4f}\n"
                f"Insult:       {results['insult']:.4f}\n"
                f"Identity Hate:{results['identity_hate']:.4f}",
                language="yaml"
            )