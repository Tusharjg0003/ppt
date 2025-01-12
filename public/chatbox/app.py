# from fastapi import FastAPI,HTTPException
# from pydantic import BaseModel
# import chromadb
# from langchain_community.document_loaders import PyPDFLoader
# from langchain.text_splitter import TokenTextSplitter
# from langchain_openai import ChatOpenAI, OpenAIEmbeddings
# from langchain_community.vectorstores import Chroma
# from langchain.prompts import PromptTemplate
# from typing_extensions import List, TypedDict
# from fastapi.middleware.cors import CORSMiddleware
# import os
# import requests
# from datetime import datetime
# import base64
# import json

# # Initialize FastAPI
# app = FastAPI()


# # Add this after initializing the `app`
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],  # Adjust to allow specific origins, e.g., ["http://localhost:3000"]
#     allow_credentials=True,
#     allow_methods=["*"],  # Allow all HTTP methods
#     allow_headers=["*"],  # Allow all headers
# )


# @app.get("/")
# def read_root():
#     return {"message": "Welcome to the HR Assistant API"}

# @app.get("/favicon.ico")
# def favicon():
#     return {"message": "Favicon not available"}


# os.environ["OPENAI_API_KEY"] = "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"

# # Input model
# class Inputs(BaseModel):
#     question: str
#     userid: str
#     chatid: str

# # Global variables to store initialized components
# vectDB = None
# llm = None
# classification_prompt = None
# main_prompt = None

# # Initialize ChromaDB client
# def initialize_chroma_db():
#     global vectDB
#     if vectDB is None:
#         client = chromadb.Client()

#         # Load and split PDF document
#         loader = PyPDFLoader("Policies.pdf")
#         pdfData = loader.load()

#         text_splitter = TokenTextSplitter(chunk_size=300, chunk_overlap=100)
#         splitData = text_splitter.split_documents(pdfData)

#         # Define storage for Chroma database
#         collection_name = "Ubi"
#         local_directory = "hammadrizwan"
#         persist_directory = os.path.join(os.getcwd(), local_directory)

#         embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))

#         # Create Chroma vector database
#         vectDB = Chroma.from_documents(
#             splitData,
#             embeddings,
#             collection_name=collection_name,
#             persist_directory=persist_directory
#         )

# # Initialize LLM
# def initialize_llm():
#     global llm
#     if llm is None:
#         api_key = os.getenv("OPENAI_API_KEY")
#         llm = ChatOpenAI(
#             openai_api_key=api_key,
#             model_name="gpt-4o",
#             temperature=0.7
#         )

# # Create classification prompt
# def initialize_classification_prompt():
#     global classification_prompt
#     if classification_prompt is None:
#         classification_template = """You are an HR assistant. Classify the following query into one of two categories: 'Company' or 'Leave Request'.
#         If the query is about Company related questions, classify it as 'Company'.
#         If the query is about a Leave Request interaction, classify it as 'Leave Request'.
#         If there are dates provided without context or showing start and end dates it should classify as 'Leave Request'

#         Query: {input}

#         Provide only one word as the classification: either 'Company' or 'Leave Request'."""

#         classification_prompt = PromptTemplate(
#             template=classification_template,
#             input_variables=["input"],
#         )

# # Create main prompt template
# def initialize_main_prompt():
#     global main_prompt
#     if main_prompt is None:
#         template = """  
#         You are a HR Assistant, a knowledgeable and professional HR Assistant chatbot for Production Precision Trading, dedicated to providing exceptional support to employees. With a focus on precision, efficiency, and clarity, you ensure every interaction is tailored to meet the questioner's needs, including assistance with policy documents, leave requests, company information, and other formal employee-related needs.

#         When responding to inquiries about yourself, you explain that your primary role is to assist employees by providing quick access to company policies, facilitating leave applications, sharing company-related information, and addressing formal employee needs. You emphasize your commitment to making HR processes seamless and efficient while supporting employees in their day-to-day requirements.

#         When responding:

#         Use a friendly yet professional tone, as if speaking directly to a valued client.
#         Provide clear, concise, and accurate answers while ensuring the response is engaging and helpful.
#         Maintain a tone of professionalism and reliability without excessive formality.
#         Context:
#         {context}

#         Question:
#         {question}

#         Your Response:
#         """  

#         main_prompt = PromptTemplate(
#             input_variables=["context", "question"],
#             template=template
#         )

# # Deduplicate results
# def deduplicate_results(results):
#     unique_results = []
#     seen_content = set()
#     for doc in results:
#         if doc.page_content not in seen_content:
#             unique_results.append(doc)
#             seen_content.add(doc.page_content)
#     return unique_results

# # Retrieve documents
# def retrieve(question):
#     retrieved_docs = vectDB.similarity_search(question, k=7)
#     filtered_docs = deduplicate_results(retrieved_docs)
#     return filtered_docs

# # Generate response
# def generate(question, context):
#     docs_content = "\n\n".join(doc.page_content for doc in context)
#     messages = main_prompt.invoke({"question": question, "context": docs_content})
#     response = llm.invoke(messages)

#     # Extract the text content from the response
#     response_text = response.content if hasattr(response, "content") else str(response)
#     return response_text.strip()

# # Classification function
# def classify_question(question):
#     """
#     Classifies the question into 'Company' or 'Leave Request'.
#     """
#     global classification_prompt

#     # Create the input prompt for classification
#     classification_input = classification_prompt.format(input=question)

#     # Invoke the LLM
#     response = llm.invoke(classification_input)

#     # Extract the text content from the response
#     result_text = response.content if hasattr(response, "content") else str(response)
#     result_text = result_text.strip().lower()

#     # Debug the output
#     print(f"Classification Response: {result_text}")

#     # Validate output
#     if result_text in ["company", "leave request"]:
#         return result_text
#     else:
#         return None
    
# import re


# # Global variables to track session history and leave details
# chat_history = []
# leave_details = {"start_date": None, "end_date": None, "reason": None}

# def request_additional_info_with_history(question):
#     """
#     Collect missing leave details and call the API once all information is gathered.
#     """
#     global chat_history, leave_details

#     # Extract relevant information from the user's response
#     if leave_details["start_date"] is None and "start date" in question.lower():
#         leave_details["start_date"] = extract_date_from_response(question)
#     elif leave_details["end_date"] is None and "end date" in question.lower():
#         leave_details["end_date"] = extract_date_from_response(question)
#     elif leave_details["reason"] is None and "reason" in question.lower():
#         leave_details["reason"] = question.replace("The reason is", "").strip()

#     # Determine the next missing detail
#     if leave_details["start_date"] is None:
#         prompt_detail = "start date"
#     elif leave_details["end_date"] is None:
#         prompt_detail = "end date"
#     elif leave_details["reason"] is None:
#         prompt_detail = "reason for leave"
#     else:
#         # If all details are collected, call the API
#         summary = (
#             f"- **Start Date:** {leave_details['start_date']}\n"
#             f"- **End Date:** {leave_details['end_date']}\n"
#             f"- **Reason for Leave:** {leave_details['reason']}\n\n"
#             "If everything looks correct, I will proceed with submitting your leave request."
#         )
#         print(summary)
#         return call_leave_request_api()

#     # Prepare the assistant's question
#     assistant_question = f"Could you please provide the {prompt_detail} for your leave request?"

#     # Log history and return the assistant's next question
#     chat_history.append(f"User: {question}")
#     chat_history.append(f"Assistant: {assistant_question}")
#     return assistant_question

# def extract_date_from_response(response):
#     """
#     Extracts a date from the response text using a regex.
#     """
#     date_pattern = r"\d{4}-\d{2}-\d{2}"  # Matches YYYY-MM-DD format
#     match = re.search(date_pattern, response)
#     return match.group(0) if match else None


# def call_leave_request_api():
#     """
#     Calls the API to create a leave request using the collected details.
#     """
#     global leave_details

#     # Check if all details are available
#     if not leave_details["start_date"] or not leave_details["end_date"] or not leave_details["reason"]:
#         return "Cannot proceed with the API call. Missing required details."
    
#     project_key = "PPT"
#     summary = "Leave Request"
#     issue_type = "Leave Request"

#     # Call the API function
#     return create_leave_request_issue(
#         leave_details["start_date"],
#         leave_details["end_date"],
#         leave_details["reason"],
#         project_key,
#         summary,
#         issue_type
#     )

# def create_leave_request_issue(start_date_str, end_date_str, description, project_key="PPT", summary="Leave Request", issue_type="Leave Request"):
#     """
#     Creates a leave request issue in Jira.
#     """
#     # Validate the date format
#     try:
#         start_date = datetime.strptime(start_date_str, "%Y-%m-%d").strftime("%Y-%m-%dT%H:%M:%S.000+0000")
#         end_date = datetime.strptime(end_date_str, "%Y-%m-%d").strftime("%Y-%m-%dT%H:%M:%S.000+0000")
#     except ValueError:
#         return "Incorrect date format, should be YYYY-MM-DD"

#     leave_request_data = {
#         "fields": {
#             "project": {
#                 "key": project_key
#             },
#             "summary": summary,
#             "description": f"{description} from {start_date} to {end_date}.",
#             "issuetype": {
#                 "name": issue_type
#             },
#             "customfield_10038": start_date, 
#             "customfield_10037": end_date    
#         }
#     }
    
#     create_issue_url = "https://obaidsajidkhan.atlassian.net/rest/api/2/issue"

#     # Base64 encode your email and API token for Basic Authorization
#     username = "obaidsajidkhan@gmail.com"
#     api_token = "ATATT3xFfGF0fa9J56mtLuBDl-RkVPcZCHmsmtJQGpnMYn8zMc5W8_SU1ox4-renTvggLwPX-7b3spMRLtcwfAz5XKToTtcUZSKCCr8c0IBUYDu_cV_s3CL3obadqA_OKdIb-XGukoValddEkzp2IWHEDVa2u7Ok-ti4_kryDnIcjzZnHR4DT2Y=B56400C5"

#     # Define the headers for the API request
#     headers = {
#         "Content-Type": "application/json",
#         "Authorization": f"Basic {base64.b64encode(f'{username}:{api_token}'.encode()).decode()}"
#     }

#     # Define authentication credentials (not strictly necessary since Authorization header is used)
#     auth = (username, api_token)

#     # Send API request
#     response = requests.post(create_issue_url, headers=headers, auth=auth, data=json.dumps(leave_request_data))

#     if response.status_code == 201:
#         return "Issue created successfully!"
#     else:
#         return f"Failed to create issue. Status code: {response.status_code}. Response content: {response.content}"





# # Unified response function
# def get_response(question, userid):
#     """
#     Generate responses dynamically based on the type of query.
#     """
#     try:
#         # Determine the type of question
#         question_type = classify_question(question)

#         if question_type == "company":
#             context = retrieve(question)
#             response = generate(question, context)
#         elif question_type == "leave request":
#             response = request_additional_info_with_history(question)
#         else:
#             response = "Unable to classify the query. Please try rephrasing your question."

#         return response

#     except Exception as e:
#         return str(e)

# # Ensure all components are initialized
# initialize_chroma_db()
# initialize_llm()
# initialize_classification_prompt()
# initialize_main_prompt()
# # Remove Streamlit testing
# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8001)
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import chromadb
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import TokenTextSplitter
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.prompts import PromptTemplate
from typing_extensions import List, TypedDict
from fastapi.middleware.cors import CORSMiddleware
import os
import requests
from datetime import datetime
import base64
import json
import re

# Initialize FastAPI
app = FastAPI()

# Add CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
)

@app.get("/")
def read_root():
    return {"message": "Welcome to the HR Assistant API"}

@app.get("/favicon.ico")
def favicon():
    return {"message": "Favicon not available"}

os.environ["OPENAI_API_KEY"] = "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"

# Input model
class Inputs(BaseModel):
    question: str
    userid: str
    chatid: str

class QueryRequest(BaseModel):
    question: str
    userid: str = "default_user"  # Added default value
    chatid: str = "default_chat"  # Added default value

class QueryResponse(BaseModel):
    answer: str

# Global variables to store initialized components
# Global variables to track leave request progress and details
leave_request_progress = {"status": "initial"}  # Tracks progress of leave request
leave_details = {"start_date": None, "end_date": None, "reason": None}  # Stores leave details

vectDB = None
llm = None
classification_prompt = None
main_prompt = None

# Initialize ChromaDB client
def initialize_chroma_db():
    global vectDB
    if vectDB is None:
        try:
            client = chromadb.Client()

            # Load and split PDF document
            loader = PyPDFLoader("Policies.pdf")
            pdfData = loader.load()

            text_splitter = TokenTextSplitter(chunk_size=300, chunk_overlap=100)
            splitData = text_splitter.split_documents(pdfData)

            # Define storage for Chroma database
            collection_name = "Ubi"
            local_directory = "hammadrizwan"
            persist_directory = os.path.join(os.getcwd(), local_directory)

            embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))

            # Create Chroma vector database
            vectDB = Chroma.from_documents(
                splitData,
                embeddings,
                collection_name=collection_name,
                persist_directory=persist_directory
            )
        except Exception as e:
            print(f"DEBUG: Failed to initialize vectDB: {str(e)}")
            raise RuntimeError(f"Error initializing vectDB: {str(e)}")


# Initialize LLM
def initialize_llm():
    global llm
    if llm is None:
        api_key = os.getenv("OPENAI_API_KEY")
        llm = ChatOpenAI(
            openai_api_key=api_key,
            model_name="gpt-4o",
            temperature=0.7
        )

# Create classification prompt
def initialize_classification_prompt():
    global classification_prompt
    if classification_prompt is None:
        classification_template = """You are an HR assistant. Classify the following query into one of two categories: 'Company' or 'Leave Request'.
        If the query is about Company related questions, classify it as 'Company'.
        If the query is about a Leave Request interaction, classify it as 'Leave Request'.
        If there are dates provided without context or showing start and end dates it should classify as 'Leave Request'

        Query: {input}

        Provide only one word as the classification: either 'Company' or 'Leave Request'."""

        classification_prompt = PromptTemplate(
            template=classification_template,
            input_variables=["input"],
        )

# Create main prompt template
def initialize_main_prompt():
    global main_prompt
    if main_prompt is None:
        template = """  
        You are a HR Assistant, a knowledgeable and professional HR Assistant chatbot for Production Precision Trading, dedicated to providing exceptional support to employees. With a focus on precision, efficiency, and clarity, you ensure every interaction is tailored to meet the questioner's needs, including assistance with policy documents, leave requests, company information, and other formal employee-related needs.

        When responding to inquiries about yourself, you explain that your primary role is to assist employees by providing quick access to company policies, facilitating leave applications, sharing company-related information, and addressing formal employee needs. You emphasize your commitment to making HR processes seamless and efficient while supporting employees in their day-to-day requirements.

        When responding:

        Use a friendly yet professional tone, as if speaking directly to a valued client.
        Provide clear, concise, and accurate answers while ensuring the response is engaging and helpful.
        Maintain a tone of professionalism and reliability without excessive formality.
        Context:
        {context}

        Question:
        {question}

        Your Response:
        """  

        main_prompt = PromptTemplate(
            input_variables=["context", "question"],
            template=template
        )

# Deduplicate results
def deduplicate_results(results):
    unique_results = []
    seen_content = set()
    for doc in results:
        if doc.page_content not in seen_content:
            unique_results.append(doc)
            seen_content.add(doc.page_content)
    return unique_results

# Retrieve documents
def retrieve(question):
    retrieved_docs = vectDB.similarity_search(question, k=7)
    filtered_docs = deduplicate_results(retrieved_docs)
    return filtered_docs

# Generate response
def generate(question, context):
    docs_content = "\n\n".join(doc.page_content for doc in context)
    messages = main_prompt.invoke({"question": question, "context": docs_content})
    response = llm.invoke(messages)

    # Extract the text content from the response
    response_text = response.content if hasattr(response, "content") else str(response)
    return response_text.strip()

# Classification function
def classify_question(question):
    """
    Classifies the question into 'Company' or 'Leave Request'.
    """
    global classification_prompt

    print(f"DEBUG: Classifying question: {question}")

    try:
        # Create the input prompt for classification
        classification_input = classification_prompt.format(input=question)

        # Invoke the LLM
        response = llm.invoke(classification_input)

        # Extract the text content from the response
        result_text = response.content if hasattr(response, "content") else str(response)
        result_text = result_text.strip().lower()

        # Debug the output
        print(f"DEBUG: Classification Response: {result_text}")

        # Validate output
        if result_text in ["company", "leave request"]:
            return result_text
        else:
            print("DEBUG: Unexpected classification result.")
            return None
    except Exception as e:
        print(f"DEBUG: Error in classification: {str(e)}")
        return None


# Handle general queries like "hello"
def handle_general_query(question: str) -> str:
    if question.lower() in ["hello", "hi", "hey"]:
        return "Hello! How can I assist you today?"
    return "I'm here to help with your queries. Please provide more details."

def request_additional_info_with_history(question):
    """
    Interactively collect leave details and call the API once all information is gathered.
    """
    global leave_details, leave_request_progress

    print(f"DEBUG: Current progress: {leave_request_progress}")
    print(f"DEBUG: Leave details so far: {leave_details}")
    print(f"DEBUG: Received question: {question}")

    try:
        if leave_request_progress["status"] == "initial":
            leave_request_progress["status"] = "asking_start_date"
            return "Could you please provide the start date for your leave? (e.g., 'Start date is 2025-01-15')"

        if leave_request_progress["status"] == "asking_start_date":
            if "start date" in question.lower():
                leave_details["start_date"] = extract_date_from_response(question)
                leave_request_progress["status"] = "asking_end_date"
                return "Got it! Now, could you please provide the end date for your leave? (e.g., 'End date is 2025-01-20')"
            else:
                return "I couldn't find a start date. Please provide it in the format 'Start date is YYYY-MM-DD'."

        if leave_request_progress["status"] == "asking_end_date":
            if "end date" in question.lower():
                leave_details["end_date"] = extract_date_from_response(question)
                leave_request_progress["status"] = "asking_reason"
                return "Great! Lastly, could you tell me the reason for your leave? (e.g., 'Reason is personal work')"
            else:
                return "I couldn't find an end date. Please provide it in the format 'End date is YYYY-MM-DD'."

        if leave_request_progress["status"] == "asking_reason":
            if "reason" in question.lower():
                leave_details["reason"] = question.split("reason is")[-1].strip()
                leave_request_progress["status"] = "completed"
                print(f"DEBUG: Final leave details: {leave_details}")
                response = call_leave_request_api()
                leave_request_progress["status"] = "initial"  # Reset progress after completion
                leave_details = {"start_date": None, "end_date": None, "reason": None}  # Reset leave details
                return response
            else:
                return "I couldn't find a reason. Please provide it in the format 'Reason is ...'."

    except Exception as e:
        print(f"DEBUG: Exception occurred: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error in leave request process: {str(e)}")

    return "I'm not sure what you're trying to do. Could you rephrase your query?"



# # Request additional info with history
# def request_additional_info_with_history(question):
#     """
#     Collect missing leave details and call the API once all information is gathered.
#     """
#     global chat_history, leave_details

#     if leave_details["start_date"] is None and "start date" in question.lower():
#         leave_details["start_date"] = extract_date_from_response(question)
#     elif leave_details["end_date"] is None and "end date" in question.lower():
#         leave_details["end_date"] = extract_date_from_response(question)
#     elif leave_details["reason"] is None and "reason" in question.lower():
#         leave_details["reason"] = question.replace("The reason is", "").strip()

#     if leave_details["start_date"] is None:
#         prompt_detail = "start date"
#     elif leave_details["end_date"] is None:
#         prompt_detail = "end date"
#     elif leave_details["reason"] is None:
#         prompt_detail = "reason for leave"
#     else:
#         summary = (
#             f"- **Start Date:** {leave_details['start_date']}\n"
#             f"- **End Date:** {leave_details['end_date']}\n"
#             f"- **Reason for Leave:** {leave_details['reason']}\n\n"
#             "If everything looks correct, I will proceed with submitting your leave request."
#         )
#         return summary

#     assistant_question = f"Could you please provide the {prompt_detail} for your leave request?"
#     return assistant_question

def extract_date_from_response(response):
    """
    Extracts a date from the response text using a regex.
    """
    print(f"DEBUG: Extracting date from response: {response}")
    date_pattern = r"\d{4}-\d{2}-\d{2}"  # Matches YYYY-MM-DD format
    match = re.search(date_pattern, response)
    if match:
        print(f"DEBUG: Extracted date: {match.group(0)}")
        return match.group(0)
    else:
        print("DEBUG: No date found in response")
        return None


# def call_leave_request_api():
#     """
#     Calls the API to create a leave request using the collected details.
#     """
#     global leave_details

#     print(f"DEBUG: Preparing to call leave request API with details: {leave_details}")


#     # Check if all details are available
#     if not leave_details["start_date"] or not leave_details["end_date"] or not leave_details["reason"]:
#         print("DEBUG: Missing details for API call.")
#         return "Cannot proceed with the API call. Missing required details."
    
#     project_key = "PPT"
#     summary = "Leave Request"
#     issue_type = "Leave Request"

#     # Call the API function
#     return create_leave_request_issue(
#         leave_details["start_date"],
#         leave_details["end_date"],
#         leave_details["reason"],
#         project_key,
#         summary,
#         issue_type
#     )

def call_leave_request_api():
    """
    Calls the API to create a leave request using the collected details.
    """
    global leave_details

    print(f"DEBUG: Preparing to call leave request API with details: {leave_details}")

    if not leave_details["start_date"] or not leave_details["end_date"] or not leave_details["reason"]:
        print("DEBUG: Missing details for API call.")
        return "Cannot proceed with the API call. Missing required details."
    
    project_key = "PPT"
    summary = "Leave Request"
    issue_type = "Leave Request"

    try:
        response = create_leave_request_issue(
            leave_details["start_date"],
            leave_details["end_date"],
            leave_details["reason"],
            project_key,
            summary,
            issue_type
        )
        print(f"DEBUG: API response: {response}")
        return response
    except requests.RequestException as e:
        print(f"DEBUG: HTTP request exception: {str(e)}")
        return f"API request failed: {str(e)}"
    except Exception as e:
        print(f"DEBUG: General exception in API call: {str(e)}")
        return f"Unexpected error in API call: {str(e)}"



def create_leave_request_issue(start_date_str, end_date_str, description, project_key="PPT", summary="Leave Request", issue_type="Leave Request"):
    """
    Creates a leave request issue in Jira.
    """
    # Validate the date format
    try:
        start_date = datetime.strptime(start_date_str, "%Y-%m-%d").strftime("%Y-%m-%dT%H:%M:%S.000+0000")
        end_date = datetime.strptime(end_date_str, "%Y-%m-%d").strftime("%Y-%m-%dT%H:%M:%S.000+0000")
    except ValueError:
        return "Incorrect date format, should be YYYY-MM-DD"

    leave_request_data = {
        "fields": {
            "project": {
                "key": project_key
            },
            "summary": summary,
            "description": f"{description} from {start_date} to {end_date}.",
            "issuetype": {
                "name": issue_type
            },
            "customfield_10038": start_date, 
            "customfield_10037": end_date    
        }
    }
    
    create_issue_url = "https://obaidsajidkhan.atlassian.net/rest/api/2/issue"

    # Base64 encode your email and API token for Basic Authorization
    username = "obaidsajidkhan@gmail.com"
    api_token = "ATATT3xFfGF0fa9J56mtLuBDl-RkVPcZCHmsmtJQGpnMYn8zMc5W8_SU1ox4-renTvggLwPX-7b3spMRLtcwfAz5XKToTtcUZSKCCr8c0IBUYDu_cV_s3CL3obadqA_OKdIb-XGukoValddEkzp2IWHEDVa2u7Ok-ti4_kryDnIcjzZnHR4DT2Y=B56400C5"


    
    # Define the headers for the API request
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Basic {base64.b64encode(f'{username}:{api_token}'.encode()).decode()}"
    }

    # Define authentication credentials (not strictly necessary since Authorization header is used)
    auth = (username, api_token)

    # Send API request
    response = requests.post(create_issue_url, headers=headers, auth=auth, data=json.dumps(leave_request_data))

    if response.status_code == 201:
        return "Issue created successfully!"
    else:
        return f"Failed to create issue. Status code: {response.status_code}. Response content: {response.content}"
    
    





# Unified response function
def get_response(question, userid):
    """
    Generate responses dynamically based on the type of query.
    """
    try:
        # Determine the type of question
        question_type = classify_question(question)

        if question_type == "company":
            context = retrieve(question)
            response = generate(question, context)
        elif question_type == "leave request":
            response = request_additional_info_with_history(question)
        else:
            response = "Unable to classify the query. Please try rephrasing your question."

        return response

    except Exception as e:
        return str(e)

@app.post("/ask", response_model=QueryResponse)
def ask_question(request: QueryRequest):
    try:
        # Handle simple greetings
        if request.question.lower() in ["hello", "hi", "hey"]:
            return QueryResponse(answer=handle_general_query(request.question))

        # Classify and process structured queries
        question_type = classify_question(request.question)

        if question_type == "company":
            context = retrieve(request.question)
            answer = generate(request.question, context)
        elif question_type == "leave request":
            answer = request_additional_info_with_history(request.question)  # Call the updated function here
        else:
            answer = "Unable to classify your query. Please rephrase your question."

        return QueryResponse(answer=answer)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


initialize_chroma_db()
initialize_llm()
initialize_classification_prompt()
initialize_main_prompt()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
