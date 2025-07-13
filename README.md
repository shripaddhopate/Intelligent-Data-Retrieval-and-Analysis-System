# 🧠 Intelligent Data Retrieval and Analysis System
  A Flask-based web application that enables users to:
  Fetch data from a MySQL database
  Perform Exploratory Data Analysis (EDA) on CSV files
  Ask natural language questions about data using LangChain and Google’s Gemini model

### 🚀 Features
- ✅ Fetch Data
  Connect to a MySQL database
  Execute SQL queries or natural language prompts
  Save query results as a CSV file for further analysis

- 📊 Exploratory Data Analysis (EDA)
  Upload a CSV file or use a preloaded dataset
  Generate visualizations and insights from prompts using eda_script.py

- 💬 Question-Answering (QA)
  Ask natural language questions about CSV data
  Powered by LangChain’s pandas agent with Google Gemini

- 💼 Session Management
  Stores preloaded CSV files across the Fetch → EDA → QA flow

- 🧹 File Management
  Automatically deletes temporary files (e.g., plots) to manage disk space

### 🔧 Prerequisites
  Python 3.8+
  MySQL database (for Fetch Data feature)
  Google API Key (for Gemini-powered Q/A)

### 📦 Installation
- 1️⃣ Clone the Repository
git clone <repository-url>
cd data-analysis-platform
- 2️⃣ Create a Virtual Environment
python -m venv venv
### For macOS/Linux:
source venv/bin/activate
### For Windows:
venv\Scripts\activate
- 3️⃣ Install Dependencies
pip install flask flask-cors flask-session pandas langchain-google-genai \
langchain-experimental python-dotenv mysql-connector-python
- 4️⃣ Set Up Environment Variables
Create a .env file in the root directory:
GOOGLE_API_KEY=your-google-api-key-here
### 📁 Directory Structure
<pre lang="markdown">```data-analysis-platform/
├── app.py
├── config/
│   └── db_config.py
├── templates/
│   ├── landing.html
│   ├── fetch_data.html
│   ├── eda.html
│   └── qa.html
├── uploads/                  # Temporary CSV and plot storage
├── eda_script.py
├── fetch_data.py
└── .env```</pre>
📝 Note: uploads/ is auto-created. Make sure your server can write to this folder.

### ⚙️ Configuration
Flask Secret Key: Replace 'your-secret-key' in app.py with a secure key.

Database Config: Update config/db_config.py as needed.

File Permissions: Ensure write access for uploads/ directory.

### ▶️ Running the Application
python app.py
Visit http://localhost:5000 in your browser.

🧭 Usage Guide
🏠 Landing Page
Choose from: Fetch Data, EDA, or QA

### 📥 Fetch Data
Enter MySQL credentials & SQL query or prompt

Choose a redirect: EDA or QA

Fetched data is stored in session as CSV

### 📈 EDA
Upload or use the session CSV

Enter custom prompts for charts or summaries

Plots are served and auto-deleted after 5 seconds

### ❓ Q&A
Upload or use the session CSV

Ask natural language questions about your data

Answers powered by Gemini via LangChain

### 📡 API Endpoints
Endpoint	Method	Description
/	GET	Landing page with navigation options
/fetch-data	GET	Fetch Data input form
/eda	GET	EDA input form
/qa	GET	Q&A input form
/api/fetch-data	POST	Fetch MySQL data & save as CSV
/api/run-eda	POST	Generate EDA output from prompt
/api/run-qa	POST	Ask questions about CSV data
/uploads/<filename>	GET	Serve plots temporarily (auto-delete in 5s)

📈 Future Improvements
🔐 Add authentication for secure credential handling

💾 Support persistent storage of analysis sessions

📊 Enhance frontend UI/UX with modern visualizations

🧩 Support for other database systems (e.g., PostgreSQL, SQLite)

📂 Accept other data formats like Excel or JSON
