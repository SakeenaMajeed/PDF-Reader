# Chat with PDF using Gemini - Streamlit App

## 📌 Overview
This Streamlit app allows users to upload PDF files and interact with them using Google's Gemini AI model. The app processes PDF content, converts it into embeddings using FAISS, and enables users to ask questions based on the uploaded documents.

## 🚀 Features
- **Upload and Process Multiple PDFs** 📂
- **Generate AI-Powered Answers** using Google Gemini 🤖
- **FAISS Vector Store Integration** for efficient search 🔍
- **Modern & Responsive UI** with glassmorphism styling ✨

## 🛠️ Setup & Installation
### 1️⃣ Prerequisites
Ensure you have the following installed:
- Python 3.8+
- Streamlit
- PyPDF2
- LangChain
- FAISS
- Google Generative AI API Key

### 2️⃣ Installation Steps
1. Clone this repository:
   ```bash
   git clone https://github.com/your-repo/chat-with-pdf.git
   cd chat-with-pdf
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set your Google API Key in the environment:
   ```bash
   export GOOGLE_API_KEY="your_api_key_here"
   ```

4. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```

## 📂 Project Structure
```
├── app.py                # Main Streamlit app
├── requirements.txt      # Required dependencies
├── faiss_index/         # FAISS vector store directory
└── README.md            # This file
```

## 🎨 UI Enhancements
- **Sidebar Menu** with PDF Upload 📁
- **Glassmorphism Effects** for modern design ✨
- **Gradient Background** for a fresh look 🎨
- **Smooth Hover Effects** for buttons 🔥

## 💡 Usage Guide
1. Upload your PDF files via the sidebar.
2. Click **Submit & Process** to generate embeddings.
3. Enter your question in the text box and get AI-powered answers!

🌐 Live Demo

Check out the live version of this app here: Live Demo

🔗 Connect with Me

LinkedIn

## 🤝 Contributing
Feel free to fork this repository and make improvements. Pull requests are welcome!

## 📝 License
This project is open-source under the MIT License.

---
💡 **Developed by [Sakeena Majeed]** 🚀

