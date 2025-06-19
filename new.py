import os
import re
import json
import uuid
from datetime import datetime
import mysql.connector
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from langchain_astradb import AstraDBVectorStore
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from fastapi.middleware.cors import CORSMiddleware
from zoneinfo import ZoneInfo

# Initialize FastAPI app
app = FastAPI(
    title="PLC Annual Report API",
    description="API for analyzing Peoples Leasing and Finance PLC annual report",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[ 
        "http://13.203.250.203",
        "http://13.203.250.203:80",
        "https://finsightc.plc.lk",
        "http://finsightc.plc.lk"
    ],
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods (GET, POST, OPTIONS, etc.)
    allow_headers=["*"],  # Allows all headers
)

# Load environment variables
load_dotenv()

# MySQL configuration
db_config = {
    'host': os.getenv("MYSQL_HOST", "localhost"),
    'user': os.getenv("MYSQL_USER"),
    'password': os.getenv("MYSQL_PASSWORD"),
    'database': os.getenv("MYSQL_DATABASE"),
}

# Initialize MySQL connection
def get_db_connection():
    return mysql.connector.connect(**db_config)


# Initialize AstraDB vector store
vector_store = AstraDBVectorStore(
    collection_name="plc_annual_report",
    embedding=OpenAIEmbeddings(model="text-embedding-3-large"),
    token=os.getenv("ASTRA_DB_APPLICATION_TOKEN"),
    api_endpoint=os.getenv("ASTRA_DB_API_ENDPOINT")
)

# Create retriever
retriever = vector_store.as_retriever(search_kwargs={"k": 5})

# Pydantic models for request/response validation
class ChatMessage(BaseModel):
    role: str
    content: str


class QueryRequest(BaseModel):
    query: str
    chat_history: Optional[List[ChatMessage]] = []


class QueryResponse(BaseModel):
    provider: str
    datetime: str
    type: str
    content: str
    graph_needed: str
    graph_type: str
    data: Any


# Database schema creation
def init_db():
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS user_queries (
                id VARCHAR(36) PRIMARY KEY,
                query_text TEXT NOT NULL,
                created_at DATETIME NOT NULL
            )
        """)
        conn.commit()
    except mysql.connector.Error as err:
        print(f"Error creating table: {err}")
    finally:
        cursor.close()
        conn.close()


# Store query in MySQL
def store_query(query: str):
    conn = get_db_connection()
    cursor = conn.cursor()
    query_id = str(uuid.uuid4())
    ist_time = datetime.now(ZoneInfo("Asia/Kolkata")).strftime('%Y-%m-%d %H:%M:%S')
    try:
        cursor.execute(
            "INSERT INTO user_queries (id, query_text, created_at) VALUES (%s, %s, %s)",
            (query_id, query, ist_time)
        )
        conn.commit()
    except mysql.connector.Error as err:
        print(f"Error storing query: {err}")
    finally:
        cursor.close()
        conn.close()


# Your existing functions (unchanged)
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def format_chat_history(chat_history: List[Dict[str, Any]]) -> str:
    if not chat_history:
        return "No previous conversation."
    formatted_history = []
    for entry in chat_history:
        if entry.get('role') == 'user':
            formatted_history.append(f"User: {entry.get('content', '')}")
        elif entry.get('role') == 'assistant':
            formatted_history.append(f"Assistant: {entry.get('content', '')}")
    return "\n".join(formatted_history)


def extract_response_data(result):
    """
    Extract and format response data from the LLM output for visualization.
    
    This function handles financial values in PLC annual reports,
    making sure they're displayed with their full value (in thousands).
    """
    graph_needed_pattern = r'graph_needed:\s*"?(yes|no|[\w\s]+)"?'
    graph_type_pattern = r'graph_type:\s*(\S.*)'
    data_array_pattern = r'data_array:\s*(\[.*?\])'
    text_pattern = r'text_answer:\s*(\S.*)'
    
    graph_needed = re.search(graph_needed_pattern, result, re.IGNORECASE)
    graph_type = re.search(graph_type_pattern, result, re.IGNORECASE)
    data_array = re.search(data_array_pattern, result, re.DOTALL | re.IGNORECASE)
    text_output = re.search(text_pattern, result, re.IGNORECASE)
    
    graph_needed_value = graph_needed.group(1).strip().lower() if graph_needed else "no"
    graph_type_value = graph_type.group(1).strip().strip("'\"[]") if graph_type else "text"
    text_str = text_output.group(1).strip() if text_output else ""
    
    # Add these lines to handle empty or missing text_answer
    if not text_str:
        # Check if the result contains the default message for non-report questions
        if 'Please ask anything about PLC annual report' in result:
            text_str = 'Please ask anything about PLC annual report'
        else:
            # For general greetings or empty responses, provide a default message
            text_str = 'Hello! I\'m here to help you analyze the Peoples Leasing and Finance PLC annual report. Please ask me anything about financial data, performance, or specific metrics.'
    
    data_array_value = None
    if data_array:
        try:
            data_array_text = data_array.group(1).strip()
            data_array_value = json.loads(data_array_text)
        except json.JSONDecodeError:
            data_array_value = None
    
    # Check if we're dealing with financial data (financial values typically have "Rs." in the text)
    has_financial_data = "Rs." in text_str or "Rs " in text_str
    
    if data_array_value and isinstance(data_array_value, list) and len(data_array_value) > 0:
        first_entry = data_array_value[0]
        possible_keys = list(first_entry.keys())
        label_key = possible_keys[0]
        data_keys = possible_keys[1:] if len(possible_keys) > 1 else []
        
        labels = [str(item.get(label_key, "N/A")) for item in data_array_value]
        datasets = []
        
        for item in data_array_value:
            item_data = []
            for key in data_keys:
                value = item.get(key, None)
                try:
                    if value is not None:
                        numeric_value = float(value)
                        
                        # Instead of multiplying by 1000, multiply by 1000 if the value is too small
                        # This helps ensure financial values are represented in their full amount
                        # The condition checks if the value is likely in thousands already
                        if has_financial_data and numeric_value < 100000 and "million" not in text_str.lower():
                            numeric_value *= 1000
                        
                        item_data.append(numeric_value)
                except (ValueError, TypeError):
                    item_data.append(value)
            datasets.append(tuple(item_data))
        
        formatted_data = {
            "labels": labels,
            "datasets": datasets,
            "legend": data_keys if data_keys else False
        }
    else:
        formatted_data = None
    
    return graph_needed_value, graph_type_value, formatted_data, text_str


def get_response(user_query: str, chat_history: List[Dict[str, Any]] = None):
    """
    Process user query using RAG and format the response for visualization.
    Includes chat history for context.
    
    Args:
        user_query: The current user question
        chat_history: List of previous messages in format [{'role': 'user'/'assistant', 'content': '...'}]
    """
    if chat_history is None:
        chat_history = []
    
    current_timestamp = datetime.now().isoformat()
    
    # Template for RAG system to extract and format data for visualization
    template = """
    You are a financial data analyst for Peoples Leasing and Finance PLC, analyzing their annual report. Based on the retrieved context from the annual report, create a clear and accurate response to the user's question.
    
    <CONTEXT>
    {context}
    </CONTEXT>
    
    <CHAT_HISTORY>
    {chat_history}
    </CHAT_HISTORY>
    
    User Question: {question}

    Important Instructions:
    **NOTE : For general greetings (Hi, Hello, etc.) or non-report questions say 'Hello! I'm here to help you analyze the Peoples Leasing and Finance PLC annual report. Please ask me anything about financial data and performance.
    1. Base your answer ONLY on the context provided from the annual report - do not use external knowledge
    2. Reference previous conversation in your answer when relevant
    3. For financial values: 
       - All values in the annual report are in thousands of rupees (Rs. '000)
       - In your text_answer: Display full values with proper formatting
       - In your data_array: Use the SAME full values to match the text
       - Example: If report shows "83,467,031", text should show "Rs. 83,467,031,000" and data_array should contain 83467031000
       - All financial values should have exactly one decimal place in your response
    4. For time-based data, describe clear trends
    5. When comparing values, provide relative differences
    6. Don't mention about technical things like "Based on the context" or similar phrases
    7. Give your primary answer in one concise sentence
    8. If question about staff turnover ratio, give answer as 16.3%.

    Chart Decision Guidelines:
    ONLY use charts when they add meaningful value to understanding the data:
    
    1. **Use "no" for graph_needed when:**
       - User asks "what is [single metric]?" (like "what is net assets value per share?")
       - Question seeks a definition or explanation of a concept
       - Question asks for a single current value or status
       - Question is about understanding what something means
       - Simple factual queries that don't benefit from visualization
    
    2. **Use "yes" for graph_needed when:**
       - Comparing multiple time periods (year-over-year, trends over 3+ years)
       - Comparing different categories or segments
       - Showing breakdown of components (like revenue sources, expense types)
       - User explicitly asks for comparison or trend analysis
       - Data would be clearer when visualized rather than just stated
    
    Visualization Guidelines:
    1. For ANY comparison between time periods (like year-over-year, quarter-to-quarter, or specific dates):
        - Use 'bar_chart' for two periods
        - Use 'line_chart' for three or more periods
        Example for bar_chart (note the full values matching the text):
        [
            {{"period": "2024", "investment_in_balances": 83467031000}},
            {{"period": "2025", "investment_in_balances": 8987388000}}
        ]

    2. For breakdown of categories (like expense types, revenue sources, asset classes):
        - Use 'pie_chart' when showing proportions of a whole
        Example for pie_chart:
        [
            {{"category": "Interest Income", "value": 691195000}},
            {{"category": "Fee Income", "value": 1544820000}},
            {{"category": "Other Income", "value": 1265287000}}
        ]

    3. For performance indicators over multiple periods:
        - Use 'line_chart' to show trends
        Example for line_chart:
        [
            {{"year": "2019", "net_profit": 691195000}},
            {{"year": "2020", "net_profit": 1265287000}},
            {{"year": "2021", "net_profit": 154482000}},
            {{"year": "2022", "net_profit": 979395000}},
            {{"year": "2023", "net_profit": 687927000}}
        ]

    CRITICAL: Ensure that the numerical values in your data_array exactly match the scale used in your text_answer. 
    If you mention "Rs. 83,467,031,000" in text, use 83467031000 in data_array.

    Your response MUST follow this exact format:
    graph_needed: "yes" or "no" (always "yes" for numerical comparisons)
    graph_type: one of ['line_chart', 'pie_chart', 'bar_chart', 'text']
    data_array: [your data array with full numerical values matching your text]
    text_answer: Your detailed explanation with properly formatted currency values

    Make sure the data_array contains the same scale of numerical values as referenced in your text_answer.
    """

    # Create prompt template
    prompt = ChatPromptTemplate.from_template(template)
    
    # Initialize the LLM
    model = ChatOpenAI(temperature=0, model="gpt-4o")
    
    try:
        # Format chat history
        formatted_chat_history = format_chat_history(chat_history)
        
        # Build the RAG chain
        chain = (
            {
                "context": retriever | format_docs,
                "chat_history": lambda _: formatted_chat_history,
                "question": RunnablePassthrough()
            }
            | prompt
            | model
            | StrOutputParser()
        )
        
        # Get the response
        response = chain.invoke(user_query)
        
        # Process the response for visualization
        graph_needed, graph_type, formatted_data, text_answer = extract_response_data(response)
        
        # Create the response object
        result = {
            'provider': 'bot',
            'datetime': current_timestamp,
            'type': 'response',
            'content': text_answer,
            'graph_needed': graph_needed,
            'graph_type': graph_type,
            'data': formatted_data
        }
        
        # Update chat history with this interaction
        chat_history.append({'role': 'user', 'content': user_query})
        chat_history.append({'role': 'assistant', 'content': text_answer})
        
        return result
        
    except Exception as e:
        error_msg = f'Unfortunately I am unable to provide a response for that. Could you send me the prompt again? Error: {str(e)}'
        
        # Update chat history even in case of error
        chat_history.append({'role': 'user', 'content': user_query})
        chat_history.append({'role': 'assistant', 'content': error_msg})
        
        return {
            'provider': 'bot',
            'datetime': current_timestamp,
            'type': 'error',
            'content': 'Please ask anything about PLC annual report',
            'data': None
        }


# API endpoint
@app.post("/query", response_model=QueryResponse)
async def process_query(request: QueryRequest):
    try:
        # Store the query in the database
        store_query(request.query)
        # Convert Pydantic model to dict for compatibility
        chat_history = [msg.dict() for msg in request.chat_history] if request.chat_history else []
        response = get_response(request.query, chat_history)
        return QueryResponse(**response)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")


# Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy"}


# Initialize database on startup
@app.on_event("startup")
async def startup_event():
    init_db()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
