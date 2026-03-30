<div align="center">
  
# 🛡️ SecureChat: AI Inference Engine
**Real-Time Harassment & Abuse Detection Prototype**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit App](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat&logo=Streamlit&logoColor=white)](https://streamlit.io/)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Transformers-yellow)](https://huggingface.co/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Open Source Love png1](https://badges.frapsoft.com/os/v1/open-source.png?v=103)](https://github.com/ellerbrock/open-source-badges/)

</div>

---

## 📖 Overview
SecureChat is an AI-driven, on-device harassment detection system designed to identify toxic, threatening, and abusive language in real-time. 

While the foundational research targets a system-level mobile application utilizing notification interception, **this repository contains the core Machine Learning pipeline and a Streamlit web prototype** used to validate the inference logic. 

Our custom DistilBERT model achieved **83.8% accuracy** in real-time classification, effectively mimicking the decision algorithm of a larger proposed multi-model ensemble.

### 📊 Model Training Pipeline
```mermaid
graph TD
    A[Raw Jigsaw Dataset 150k+ rows] -->|Massive Neutral Bias| B(Data Engineering)
    B -->|1:3 Downsampling| C[Balanced Dataset 64,900 rows]
    C --> D(DistilBERT Tokenizer)
    D --> E((DistilBERT Base Uncased))
    E -->|4 Epochs, BCE Loss| F[Custom Trained Model]
    F --> G[(securechat_model_v2_final)]
    
    style A fill:#f9f9f9,stroke:#333,stroke-width:2px
    style C fill:#e1f5fe,stroke:#0288d1,stroke-width:2px
    style E fill:#fff3e0,stroke:#f57c00,stroke-width:2px
    style G fill:#e8f5e9,stroke:#388e3c,stroke-width:2px
````

### 🧠 On-Device Inference Architecture

```mermaid
flowchart LR
    A[Intercepted Message] --> B[Streamlit Dashboard UI]
    B --> C(AutoTokenizer pt)
    C --> D{DistilBERT Inference}
    D -->|Raw Logits| E[Sigmoid Activation]
    E --> F{Weighted Decision Logic}
    
    F -->|Threat > 0.5| G[⚠️ Threat Alert]
    F -->|Abusive > 0.5| H[🛑 Abusive Content]
    F -->|All < 0.5| I[✅ Neutral Pass]

    style A fill:#f3e5f5,stroke:#8e24aa,stroke-width:2px
    style D fill:#ffebee,stroke:#d32f2f,stroke-width:2px
    style G fill:#ffcdd2,stroke:#c62828,stroke-width:2px
    style H fill:#ffe0b2,stroke:#ef6c00,stroke-width:2px
    style I fill:#c8e6c9,stroke:#2e7d32,stroke-width:2px
```

-----

## 🚀 Installation & Local Setup

### 1\. Clone the Repository

```bash
git clone https://github.com/spideyashith/secure-chat.git
cd secure-chat
```

### 2\. Install Dependencies

Ensure you have Python installed, then install the required ML libraries:

```bash
pip install -r requirements.txt
```

*(Note: If you don't have a `requirements.txt`, you can run: `pip install streamlit torch numpy transformers scikit-learn pandas datasets`)*

### 3\. Obtain the AI Model (Heavy Weights)

Because Git does not support massive binary files, the trained `model.safetensors` file is not hosted in this repository.

**To get the fully fine-tuned V2 model files, please contact the developer:**
📧 **ashithfernandes25@gmail.com**

Once received, extract the model files and place the `securechat_model_v2_final` folder directly in the root directory of this project.

### 4\. Run the Prototype

```bash
streamlit run app.py
```

-----

## 🤝 Contributing & Open Source

We want to push this model past the 90% accuracy mark and add an NSFW/Sexual content classifier\! This project is completely open source and we welcome pull requests from the community.

**Areas we need help with:**

  * **Dataset Augmentation:** Adding robust NSFW/Sexual harassment text datasets to the training pipeline.
  * **Model Compression:** Converting the PyTorch `.safetensors` model to `TensorFlow Lite` (.tflite) for future mobile deployment.
  * **Hyperparameter Tuning:** Improving the F1-Macro score of the current DistilBERT model.

**How to contribute:**

1.  Fork the Project
2.  Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3.  Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4.  Push to the Branch (`git push origin feature/AmazingFeature`)
5.  Open a Pull Request

-----

## 🏆 Credits & Acknowledgments

  * **Project Lead & Prototype Developer:** Ashith Joswa Fernandes
  * **Academic Guidance:** Dr. Srinivas BL
  * **Special Thanks & Concept Contributors:** Karthik and Prathamesh for their foundational support and conceptual contributions to the project ecosystem.

-----

\<div align="center"\>
\<img src="[https://media.giphy.com/media/L8K62iDadb1MQ/giphy.gif](https://media.giphy.com/media/L8K62iDadb1MQ/giphy.gif)" width="100%" height="100" style="border-radius: 8px; margin-bottom: 20px;"\>

\<h3\>Thank you for exploring the SecureChat Prototype\!\</h3\>

\<p\>\<i\>"Privacy is not an option, and it shouldn't be the price we accept for just getting on the internet."<br>— Gary Kovacs\</i\>\</p\>

\<p\>Feel free to reach out, fork the repo, and contribute to making on-device safety a reality.\</p\>
\</div\>

```

Save it, run `git add .`, `git commit -m "Final README formatting fixes"`, and `git push`. This will render perfectly. Are you ready to move on to anything else, or is this repository officially finished?
```
