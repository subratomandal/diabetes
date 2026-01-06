#  DiabetesSLM

[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/subratomandalme/diabetes)

A simple web application hosted on Hugging Face Spaces that utilizes a Small Language Model (Google T5 large) to answer basic questions related to diabetes based on a provided dataset.

**Disclaimer:** This application is for informational and demonstrational purposes only. It does **NOT** provide medical advice. Always consult with a qualified healthcare professional for any health concerns or before making any decisions related to your health or treatment.

---

##  Live

You can try the live application hosted on Hugging Face Spaces:  **[Access the Diabetes Space Here](https://huggingface.co/spaces/subratomandalme/diabetes)**


---

##  Features

* Provides answers to common questions about diabetes.
* Utilizes a fine-tuned Small Language Model (SLM) for question answering.
* Simple and intuitive web interface (likely using Gradio or Streamlit).
* Based on the `medical_qa.csv` dataset.

---

##  Screenshots

<img width="1466" height="781" alt="Screenshot 2026-01-06 at 7 00 31 PM" src="https://github.com/user-attachments/assets/528c8a2e-92b3-4cbc-9341-52a2292d1e3f" />

---

<img width="1465" height="778" alt="Screenshot 2026-01-06 at 7 03 27 PM" src="https://github.com/user-attachments/assets/e83769ea-b9f4-4b7c-8037-b1bdf67907ae" />

---

<img width="1467" height="781" alt="Screenshot 2026-01-06 at 7 03 38 PM" src="https://github.com/user-attachments/assets/e49f96ce-7e9e-4599-984f-14688a91faf7" />

---

##  How It Works

This application follows a simple workflow:

1.  **User Input:** The user types a diabetes-related question into the web interface.
2.  **Frontend (app.py):** The UI framework (Gradio/Streamlit) running via `app.py` captures the input.
3.  **Backend Logic (model_runner.py):** The input is processed, potentially by helper functions in `model_runner.py`.
4.  **SLM Inference:** The processed query is fed into the fine-tuned Small Language Model.
5.  **Response Generation:** The SLM generates an answer based on its training data.
6.  **Output Display:** The generated answer is sent back to `app.py` and displayed to the user in the web interface.

---

##  Usage

To use the application:

1.  Navigate to the Hugging Face Space URL: [LINK](https://huggingface.co/spaces/subratomandalme/diabetes)
2.  Wait for the application to load.
3.  Enter your question about diabetes in the provided text input field.
4.  Click the "Ask" button.
5.  The model's answer will appear in the output area.

---

##  Dataset

This model was trained or fine-tuned using the `medical_qa.csv` dataset included in this repository.

* **Source:** https://www.nhs.uk/conditions/diabetes/
* **Content:** The dataset contains pairs of medical questions and answers, focused on diabetes.
* **Format:** CSV.

---

##  Model

The core of this application is a Small Language Model (SLM).

* **Type:** [Fine-tuned Flan-T5-large.]
* **Training:** The model was fine-tuned using the `train_slm.py` script on the `medical_qa.csv` dataset to specialize in answering diabetes-related questions.

---

## Technology Stack

* **Language:** Python
* **ML Framework:** PyTorch
* **Core Libraries:**
 * `flask` (For the web application/API)
 * `torch` (The core PyTorch library)
 * `transformers` (For interacting with Hugging Face models FLAN T5 LARGE)
 * `datasets` (For data handling)
 * `sentencepiece` (For text tokenization)
 * `accelerate` (For simplifying multi-GPU/distributed training)
* **Platform:** Hugging Face Spaces
* **Version Control:** Git / Git LFS
* **Containerization (Optional):** Docker (if you used `Dockerfile.dockerfile` - remember to rename it to `Dockerfile` if using the Docker SDK on Spaces)

---

##  Limitiations & Future Work

**Limitations:**

* **NOT Medical Advice:** This tool cannot replace professional medical consultation.
* **Knowledge Scope:** Answers are limited to the information present in the `medical_qa.csv` dataset and the SLM's training. It may not know about recent developments or highly specific cases.
* **Accuracy:** While fine-tuned, the SLM may still generate incorrect or nonsensical answers (hallucinations).
* **Basic Understanding:** The model may struggle with very complex, nuanced, or poorly phrased questions.



##  Contact

Created by subratomandal - https://github.com/subratomandal
