from flask import Flask, request, jsonify, render_template, send_from_directory, session
from flask_cors import CORS
from flask_session import Session
import os
from werkzeug.utils import secure_filename
from eda_script import get_columns_by_type, run_eda
from fetch_data import fetch_and_save_data
import pandas as pd
import uuid
import time
import threading
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_experimental.agents import create_pandas_dataframe_agent
from dotenv import load_dotenv
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

app = Flask(__name__)
CORS(app)

app.config['SECRET_KEY'] = 'your-secret-key'  
app.config['SESSION_TYPE'] = 'filesystem'
Session(app)

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'csv'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def cleanup_file(file_path, delay=5):
    """Delete the file after a delay to allow the client to load it."""
    time.sleep(delay)
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
    except Exception as e:
        logger.error(f"Error cleaning up file {file_path}: {e}")

def run_db_query(db_config, prompt):
    """
    Execute a database query using DBFetcher and return the result.
    """
    try:
        output_csv = os.path.join(app.config['UPLOAD_FOLDER'], f"query_result_{uuid.uuid4()}.csv")
        
        from config.db_config import DB_CONFIG
        DB_CONFIG.update({
            'host': db_config['host'],
            'database': db_config['database'],
            'user': db_config['user'],
            'password': db_config['password'],
            'port': db_config.get('port', 3306)
        })
        
        success = fetch_and_save_data(prompt, output_csv)
        if not success:
            return {'error': 'Query execution failed'}
        
        df = pd.read_csv(output_csv)
        data = df.to_dict(orient='records')
        
        return {'data': data, 'csv_path': os.path.basename(output_csv)}
    
    except Exception as e:
        logger.error(f"Query execution error: {str(e)}")
        return {'error': f'Query execution failed: {str(e)}'}

class QAAgent:
    """A class to handle Question-Answering (Q/A) on a CSV file using a LangChain pandas agent."""
    
    def __init__(self, google_api_key: str, model_name: str = "gemini-1.5-flash"):
        """
        Initialize the Q/A agent with a specified LLM.
        """
        try:
            self.llm = ChatGoogleGenerativeAI(
                model=model_name,
                google_api_key=google_api_key
            )
            self.df = None
            self.agent = None
        except Exception as e:
            logger.error(f"Error initializing QAAgent: {str(e)}")
            raise

    def load_csv(self, filepath: str) -> bool:
        """
        Load a CSV file into a pandas DataFrame.
        """
        try:
            self.df = pd.read_csv(filepath)
            logger.info(f"Successfully loaded CSV file: {filepath}")
            return True
        except FileNotFoundError:
            logger.error(f"File '{filepath}' not found.")
            return False
        except pd.errors.EmptyDataError:
            logger.error(f"File '{filepath}' is empty.")
            return False
        except Exception as e:
            logger.error(f"Error loading CSV: {str(e)}")
            return False

    def create_qa_agent(self) -> bool:
        """
        Create a pandas DataFrame agent for Q/A.
        """
        if self.df is None:
            logger.error("No DataFrame loaded. Please load a CSV file first.")
            return False
        try:
            self.agent = create_pandas_dataframe_agent(
                self.llm,
                self.df,
                verbose=True,
                allow_dangerous_code=True
            )
            logger.info("Q/A agent created successfully.")
            return True
        except Exception as e:
            logger.error(f"Error creating Q/A agent: {str(e)}")
            return False

    def ask_question(self, question: str) -> str:
        """
        Ask a question to the Q/A agent and get a response.
        """
        if self.agent is None:
            logger.error("Q/A agent not initialized. Please load a CSV and create an agent.")
            return "❌ Error: Q/A agent not initialized. Please load a CSV and create an agent."
        try:
            response = self.agent.run(question)
            logger.info(f"Q/A response: {response}")
            return response
        except Exception as e:
            logger.error(f"Error processing question: {str(e)}")
            return f"❌ Error processing question: {str(e)}"

@app.route('/')
def landing():
    return render_template('landing.html')

@app.route('/fetch-data')
def fetch_data():
    return render_template('fetch_data.html')

@app.route('/eda')
def index():
    csv_file = session.get('csv_file', None)
    return render_template('eda.html', preloaded_csv=csv_file)

@app.route('/qa')
def qa():
    csv_file = session.get('csv_file', None)
    return render_template('qa.html', preloaded_csv=csv_file)

@app.route('/api/run-eda', methods=['POST'])
def run_eda_endpoint():
    try:
        if 'csv_file' in request.form and request.form['csv_file']:
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], request.form['csv_file'])
            if not os.path.exists(file_path):
                logger.error(f"Preloaded CSV not found: {file_path}")
                return jsonify({'error': 'Preloaded CSV not found'}), 404
            prompt = request.form.get('prompt', '')
        else:
            if 'file' not in request.files:
                logger.error("No file uploaded in EDA request")
                return jsonify({'error': 'No file uploaded'}), 400
            file = request.files['file']
            prompt = request.form.get('prompt', '')
            if not file or file.filename == '':
                logger.error("No file selected in EDA request")
                return jsonify({'error': 'No file selected'}), 400
            if not allowed_file(file.filename):
                logger.error(f"Invalid file type: {file.filename}")
                return jsonify({'error': 'Invalid file type. Only CSV is allowed'}), 400
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            logger.info(f"Saved uploaded file: {file_path}")
        
        if not prompt:
            columns = get_columns_by_type(file_path)
            os.remove(file_path)
            if columns is None:
                logger.error("Failed to load CSV file for columns")
                return jsonify({'error': 'Failed to load CSV file'}), 500
            return jsonify({'columns': columns})
        
        result = run_eda(file_path, prompt, app.config['UPLOAD_FOLDER'])
        if 'csv_file' not in request.form or not request.form['csv_file']:
            os.remove(file_path)
        
        plot_file = None
        if 'saved at' in result:
            plot_file = result.split('saved at ')[1].strip()
            plot_file = os.path.basename(plot_file)
        
        return jsonify({'data': result, 'plot': plot_file})
    
    except Exception as e:
        logger.error(f"EDA endpoint error: {str(e)}")
        return jsonify({'error': f'Server error: {str(e)}'}), 500

@app.route('/api/fetch-data', methods=['POST'])
def fetch_data_endpoint():
    try:
        data = request.get_json()
        if not data:
            logger.error("No data provided in fetch-data request")
            return jsonify({'error': 'No data provided'}), 400
        
        required_fields = ['host', 'database', 'user', 'password', 'prompt']
        for field in required_fields:
            if field not in data or not data[field]:
                logger.error(f"Missing or empty field: {field}")
                return jsonify({'error': f'Missing or empty field: {field}'}), 400
        
        db_config = {
            'host': data['host'],
            'database': data['database'],
            'user': data['user'],
            'password': data['password'],
            'port': data.get('port', 3306)
        }
        prompt = data['prompt']
        
        result = run_db_query(db_config, prompt)
        if 'error' in result:
            logger.error(f"DB query error: {result['error']}")
            return jsonify({'error': result['error']}), 500
        
        session['csv_file'] = result['csv_path']
        logger.info(f"Stored CSV in session: {result['csv_path']}")
        
        return jsonify({
            'data': result['data'],
            'csv': result['csv_path'],
            'redirect': data.get('redirect', '/eda')
        })
    
    except Exception as e:
        logger.error(f"Fetch data endpoint error: {str(e)}")
        return jsonify({'error': f'Server error: {str(e)}'}), 500

@app.route('/api/run-qa', methods=['POST'])
def run_qa_endpoint():
    try:
        google_api_key = os.getenv("GOOGLE_API_KEY")
        if not google_api_key:
            logger.error("Google API key not configured")
            return jsonify({'error': 'Google API key not configured'}), 500
        
        if 'csv_file' in request.form and request.form['csv_file']:
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], request.form['csv_file'])
            if not os.path.exists(file_path):
                logger.error(f"Preloaded CSV not found: {file_path}")
                return jsonify({'error': 'Preloaded CSV not found'}), 404
        else:
            if 'file' not in request.files:
                logger.error("No file uploaded in QA request")
                return jsonify({'error': 'No file uploaded'}), 400
            file = request.files['file']
            if not file or file.filename == '':
                logger.error("No file selected in QA request")
                return jsonify({'error': 'No file selected'}), 400
            if not allowed_file(file.filename):
                logger.error(f"Invalid file type: {file.filename}")
                return jsonify({'error': 'Invalid file type. Only CSV is allowed'}), 400
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            logger.info(f"Saved uploaded file: {file_path}")
        
        question = request.form.get('question', '')
        if not question:
            logger.error("No question provided in QA request")
            return jsonify({'error': 'No question provided'}), 400
        
        qa_agent = QAAgent(google_api_key=google_api_key)
        if not qa_agent.load_csv(file_path):
            logger.error(f"Failed to load CSV file: {file_path}")
            return jsonify({'error': 'Failed to load CSV file'}), 500
        if not qa_agent.create_qa_agent():
            logger.error("Failed to create Q/A agent")
            return jsonify({'error': 'Failed to create Q/A agent'}), 500
        
        response = qa_agent.ask_question(question)
        
        if 'csv_file' not in request.form or not request.form['csv_file']:
            os.remove(file_path)
            logger.info(f"Cleaned up file: {file_path}")
        
        return jsonify({'data': response})
    
    except Exception as e:
        logger.error(f"QA endpoint error: {str(e)}")
        return jsonify({'error': f'Server error: {str(e)}'}), 500

@app.route('/uploads/<filename>')
def serve_plot(filename):
    try:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        response = send_from_directory(app.config['UPLOAD_FOLDER'], filename)
        threading.Thread(target=cleanup_file, args=(file_path, 5)).start()
        return response
    except FileNotFoundError:
        logger.error(f"File not found: {filename}")
        return jsonify({'error': 'File not found'}), 404

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)