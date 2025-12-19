import os
import json
from typing import Dict, List, Any, TypedDict
from datetime import datetime
import base64
from pathlib import Path
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from google import genai
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
import io
import vertexai
from vertexai.generative_models import GenerativeModel
from functools import reduce
from google.genai import types
#import google.generativeai as genai
#from google.generativeai import types
from config import STRUCTURE_ANALYSIS_PROMPT,PATHOLOGY_ANALYSIS_PROMPT,LIST_ALL_PROBLEMS
# Import configuration
import random
from pypdf import PdfReader
import re

from config import get_config, validate_config
from google.auth.exceptions import DefaultCredentialsError

from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import shutil

import tempfile

# 1. Define the path to your credential file
credential_path = "C:\\Users\\eduar\\INSUS\\pot-sand2\\be1\\gcp_cred1.json"

# 2. Check if the file actually exists before trying to use it
if not os.path.exists(credential_path):
    print(f"❌ ERROR: Credential file not found at the specified path: {credential_path}")
    print("Please make sure the path is correct and the file is there.")
else:
    print(f"✅ Credential file found at: {credential_path}")
    
    
    # Set the environment variable for the SDK to find the credentials
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = credential_path

    # 3. Wrap the initialization in a try...except block to catch errors
    try:
        config_check = {"configurable": {"thread_id": "1"}}
        print("Attempting to initialize Vertex AI...")
        vertexai.init(
            project="pot-test-environment",
            location="us-central1",
            api_endpoint="us-central1-aiplatform.googleapis.com"
        )
        print("✅ SUCCESS: Vertex AI has been initialized successfully!")

    except DefaultCredentialsError as e:
        print(f"❌ AUTHENTICATION ERROR: The credentials are not valid or could not be found.")
        print(f"   Please check if the JSON file is correct and has the right permissions in your GCP project.")
        print(f"   Details: {e}")
        
    except Exception as e:
        print(f"❌ AN UNEXPECTED ERROR OCCURRED during Vertex AI initialization.")
        print(f"   Error type: {type(e).__name__}")
        print(f"   Details: {e}")
        print(f"   This could be due to an incorrect project ID, location, or network issues.")


llm = init_chat_model("gemini-1.5-pro-002")

# read all pdf files and return text

# Get configuration

config = get_config()



# Configure Google Gemini API

GOOGLE_API_KEY = config["api"]["google_api_key"]

# Add this new function to your script

def draw_cover_page(canvas, doc):
    """
    Draws a full-page background image for the cover.
    This page will NOT have a header or footer.
    """
    # Build the path to the cover image dynamically
    script_dir = Path(__file__).resolve().parent
    cover_image_path = script_dir / "ReportImages" / "intro.png"
    
    canvas.saveState()
    
    if os.path.exists(cover_image_path):
        # Draw the image to fill the entire page, from corner (0,0)
        canvas.drawImage(
            str(cover_image_path), 
            0, 
            0, 
            width=doc.pagesize[0], 
            height=doc.pagesize[1], 
            mask='auto'
        )
    else:
        # Provide a fallback message if the image is missing
        canvas.setFont('Helvetica-Bold', 24)
        canvas.drawCentredString(doc.pagesize[0] / 2.0, doc.pagesize[1] / 2.0, "Cover Image Not Found")

    canvas.restoreState()


def draw_header_footer(canvas, doc):
    """
    Draws the header (logo) and footer (banner) on each page.
    """

    # 1. Get the directory where this script is located.
    # __file__ is a special variable that holds the path to the current script.
    script_dir = Path(__file__).resolve().parent

    # 2. Define the relative paths from the script's directory to your images.
    #    The '/' operator in pathlib automatically handles joining paths correctly
    #    for any operating system (Windows, macOS, Linux).
    logo_path = script_dir / "ReportImages" / "PRBlogo.png"
    footer_path = script_dir / "ReportImages" / "PRBsignature.jpg"

    # Save the current state of the canvas
    canvas.saveState()
    
    # --- HEADER (Logo) ---
    # Position the logo in the top right corner
    # Co-ordinates are measured from the bottom-left corner of the page
    logo_width = 1.1 * inch
    logo_height = 1.1 * inch
    x_pos = doc.width + doc.leftMargin - logo_width + (1.0 * inch)  # Aligns to the right margin
    y_pos = doc.height + doc.topMargin - logo_height + (1.0 * inch) # Aligns to the top margin
    
    if os.path.exists(logo_path):
        canvas.drawImage(logo_path, x_pos, y_pos, width=logo_width, height=logo_height, mask='auto')
    else:
        # Fallback if logo is not found
        canvas.setFont('Helvetica', 8)
        canvas.drawString(x_pos, y_pos + 0.5 * inch, "Logo not found")

    # --- FOOTER (Banner) ---
    # 1. Define the desired width and height for your footer banner
    dim = 0.5
    footer_width = dim * 5.5 * inch    # <<< CHANGE THIS to your desired width
    footer_height = dim * inch # <<< CHANGE THIS to your desired height

    # 2. Calculate the X coordinate to center the image on the page
    x_pos_footer = (doc.pagesize[0] - footer_width) / 2.0

    # 3. Define the Y coordinate (its distance from the bottom of the page)
    y_pos_footer = 0.2 * inch # <<< CHANGE THIS to move it up or down

    if os.path.exists(footer_path):
        # 4. Draw the image using the new calculated position and dimensions
        canvas.drawImage(footer_path, x_pos_footer, y_pos_footer, width=footer_width, height=footer_height, mask='auto')
    else:
        # Fallback if footer is not found (now also centered)
        canvas.setFont('Helvetica', 8)
        # Use drawCentredString for easy text centering
        canvas.drawCentredString(doc.pagesize[0] / 2.0, y_pos_footer + (footer_height/2), "Footer banner not found")

    # --- Page Numbers ---
    # This will still be aligned to the right text margin
    page_num_text = f"Page {doc.page}"
    canvas.setFont('Helvetica', 9)
    canvas.drawRightString(doc.width + doc.leftMargin, 0.25 * inch, page_num_text)

    # Restore the canvas to its original state
    canvas.restoreState()



def get_language_instruction(language: str) -> str:
    """Returns the appropriate language instruction string."""
    if language == "french":
        return "Give the response in French."
    else:
        return ""
    
def format_llm_text(text: str) -> str:
    # Headings: Convert ## to <b><font size=14>...</font></b>
    text = re.sub(r'^## (.+)$', r'<b><font size="14">\1</font></b>', text, flags=re.MULTILINE)

    # Bold: **text** -> <b>text</b>
    text = re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', text)

    # Italic: *text* -> <i>text</i>
    text = re.sub(r'(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)', r'<i>\1</i>', text)

    # New lines: Replace \n with <br/>
    text = text.replace('\n', '<br/>')

    return text

# Define the state schema
class ReportState(TypedDict):
    """State for the report generation workflow"""
    # Basic details
    client_details: Dict[str, str]
    service_provider_details: Dict[str, str]
    project_reference: str
    date: str
    subject: str
    project_dimensions: Dict[str, str]
    prb_reference: str
    
    # Analysis results
    structure_analysis: Dict[str, Any]
    pathology_analysis: Dict[str, Any]
    solutions_analysis: Dict[str, Any]
    paint_problems : Dict[str,Any]
    
    # Images
    images: List[str]
    
    # Final report
    pdf_path: str
    messages: List[Any]

    # Language
    language: str


def encode_image_to_base64(image_path: str) -> str:
    """Encode image to base64 for API calls"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def validate_image_path(image_path: str) -> bool:
    """Validate image file path and format"""
    if not os.path.exists(image_path):
        return False
    
    # Check file extension
    file_ext = Path(image_path).suffix.lower()
    if file_ext not in config["validation"]["supported_image_formats"]:
        return False
    
    # Check file size
    file_size_mb = os.path.getsize(image_path) / (1024 * 1024)
    if file_size_mb > config["validation"]["max_image_size_mb"]:
        return False
    
    return True

def start_node(state: ReportState) -> ReportState:
    """Initialize the report generation process"""
    print("🚀 Starting report generation process...")
    
    # Initialize basic details with default values from config
    state["client_details"] = config["defaults"]["client_details"].copy()
    state["service_provider_details"] = config["defaults"]["service_provider_details"].copy()
    state["project_reference"] = "PRJ-001"
    state["date"] = datetime.now().strftime("%Y-%m-%d")
    state["subject"] = "Structural Analysis and Rehabilitation Report"
    state["project_dimensions"] = config["defaults"]["project_dimensions"].copy()
    state["prb_reference"] = "PRB-001"
    
    state["messages"] = [SystemMessage(content=config["system_message"])]
    state["language"] = "french" # Modify to change language. "french" or anything else for English
    
    print("✅ Basic details initialized")
    return state

def generate_basic_details_node(state: ReportState) -> ReportState:
    """Generate or update basic project details"""
    print("📋 Generating basic project details...")
    # Data Collected from streamlit can be used here
    # You can add LLM calls here to generate more detailed information based on project context

    print("✅ Basic details generated")
    return state

def analyze_structure_node(state: ReportState) -> ReportState:
    """Analyze images for type of existing structure using Gemini Vision"""
    print("🏗️ Analyzing existing structure...")
    
    if not state.get("images"):
        print("⚠️ No images provided for structure analysis")
        state["structure_analysis"] = {"error": config["errors"]["no_images"]}
        return state
    
    # Validate images
    valid_images = []
    for image_path in state["images"]:
        if validate_image_path(image_path):
            valid_images.append(image_path)
        else:
            print(f"⚠️ Invalid or missing image: {image_path}")
    
    if not valid_images:
        state["structure_analysis"] = {"error": config["errors"]["no_images"]}
        return state
    
    try:
        # Process each image with Gemini Vision
        client = genai.Client(
          vertexai=True,
          project="pot-test-environment",
          location="europe-central2",
        )
        total_msg=[]
        language_instruction = get_language_instruction(state["language"])
        full_prompt = f"{STRUCTURE_ANALYSIS_PROMPT}\n\n{language_instruction}"
        msg1_text1 = types.Part.from_text(text=full_prompt)
        total_msg.append(msg1_text1)
        for img in valid_images:
            with open(img, "rb") as img:
                s = base64.b64encode(img.read())
                x=s.decode("utf-8")
            total_msg.append(types.Part.from_bytes(data=base64.b64decode(f"""{x}"""),mime_type="image/jpeg"))
        model = "gemini-1.5-pro-002"
        contents = [
        types.Content(
          role="user",
          parts=total_msg
        ),
      ]
        
        
        generate_content_config = types.GenerateContentConfig(
        temperature = 0.5,
        top_p =0.5,
        max_output_tokens = 6000,
        system_instruction=[types.Part.from_text(text="""Create a comprehensive summary based on the photos , pleasea generate plain summary and not JSON""")],
      )
        t=[]
        for chunk in client.models.generate_content_stream(model = model,contents = contents,config = generate_content_config):
            t.append(chunk.text)
        summary_text = ''.join(t)
        state["structure_analysis"] = {
            "results":summary_text,
            "photos":random.sample(valid_images, 2),
            "summary": "Structure analysis completed"
        }
        
        print("✅ Structure analysis completed")
        
    except Exception as e:
        print(f"❌ Error in structure analysis: {e}")
        state["structure_analysis"] = {"error": f"{config['errors']['api_error']}: {str(e)}"}
    
    return state

def analyze_pathologies_node(state: ReportState) -> ReportState:
    """Analyze images for pathologies using Gemini Vision"""
    print("🔍 Analyzing pathologies...")
    
    if not state.get("images"):
        print("⚠️ No images provided for pathology analysis")
        state["pathology_analysis"] = {"error": config["errors"]["no_images"]}
        return state
    
    # Validate images
    valid_images = []
    for image_path in state["images"]:
        if validate_image_path(image_path):
            valid_images.append(image_path)
        else:
            print(f"⚠️ Invalid or missing image: {image_path}")
    
    if not valid_images:
        state["pathology_analysis"] = {"error": config["errors"]["no_images"]}
        return state
    
    try:
        client = genai.Client(
          vertexai=True,
          project="pot-test-environment",
          location="europe-central2",
      )
        total_msg=[]
        language_instruction = get_language_instruction(state["language"])
        full_prompt = f"{PATHOLOGY_ANALYSIS_PROMPT}\n\n{language_instruction}"
        msg1_text1 = types.Part.from_text(text=full_prompt)
        total_msg.append(msg1_text1)
        for img in valid_images:
            with open(img, "rb") as img:
                s = base64.b64encode(img.read())
                x=s.decode("utf-8")
            total_msg.append(types.Part.from_bytes(data=base64.b64decode(f"""{x}"""),mime_type="image/jpeg"))
        model = "gemini-1.5-pro-002" #
        contents = [
        types.Content(
          role="user",
          parts=total_msg
        ),
      ]
        
        
        generate_content_config = types.GenerateContentConfig(
        temperature = 0.3,
        top_p =0.3,
        max_output_tokens = 8000,
        system_instruction=[types.Part.from_text(text="""Generate a plain summary on the given points ,explain them , nothing else""")],
      )
        t=[]
        for chunk in client.models.generate_content_stream(model = model,contents = contents,config = generate_content_config):
            t.append(chunk.text)
        summary_text = ''.join(t)
        # Process each image with Gemini Vision
    
        state["pathology_analysis"] = {
            "results": summary_text,
            "photos":random.sample(valid_images, 2),
            "summary": "Pathology analysis completed"
        }
        
        print("✅ Pathology analysis completed")
        
    except Exception as e:
        print(f"❌ Error in pathology analysis: {e}")
        state["pathology_analysis"] = {"error": f"{config['errors']['api_error']}: {str(e)}"}
    
    return state


def analyse_problems(state: ReportState) -> ReportState:
    """Analyze images for Listing all the issues"""
    
    if not state.get("images"):
        print("⚠️ No images provided for pathology analysis")
        state["pathology_analysis"] = {"error": config["errors"]["no_images"]}
        return state
    
    # Validate images
    valid_images = []
    for image_path in state["images"]:
        if validate_image_path(image_path):
            valid_images.append(image_path)
        else:
            print(f"⚠️ Invalid or missing image: {image_path}")
    
    if not valid_images:
        state["pathology_analysis"] = {"error": config["errors"]["no_images"]}
        return state
    
    try:
        client = genai.Client(
          vertexai=True,
          project="pot-test-environment",
          location="europe-central2",
      )
        total_msg=[]
        language_instruction = get_language_instruction(state["language"])
        text_context_for_problem=LIST_ALL_PROBLEMS+"Structural problems :"+state.get('structure_analysis')['results'] + "pathologies problems : " + state.get('pathology_analysis')['results']
        full_prompt = f"{text_context_for_problem}\n\n{language_instruction}"
        msg1_text1 = types.Part.from_text(text=full_prompt)
        total_msg.append(msg1_text1)
        for img in valid_images:
            with open(img, "rb") as img:
                s = base64.b64encode(img.read())
                x=s.decode("utf-8")
            total_msg.append(types.Part.from_bytes(data=base64.b64decode(f"""{x}"""),mime_type="image/jpeg"))
        model = "gemini-1.5-pro-002"
        contents = [
        types.Content(
          role="user",
          parts=total_msg
        ),
      ]
        
        
        generate_content_config = types.GenerateContentConfig(
        temperature = 0.2,
        top_p =0.2,
        max_output_tokens = 2000,
        system_instruction=[types.Part.from_text(text="""Based on the text and the photos generate the problems that you see related to paints""")],
      )
        t=[]
        for chunk in client.models.generate_content_stream(model = model,contents = contents,config = generate_content_config):
            t.append(chunk.text)
        summary_text = ''.join(t)
        # Process each image with Gemini Vision
    
        state["paint_problems"] = {
            "results": summary_text,
            "summary": "Problems Generated"
        }
        
        print("✅ ALl problems related to paints are concatinated")
        
    except Exception as e:
        print(f"❌ Error in paint_problems: {e}")
        state["paint_problems"] = {"error": f"{config['errors']['api_error']}: {str(e)}"}
    
    return state


def generate_solutions_node(state: ReportState) -> ReportState:
    """Generate solutions and finishing systems based on previous analyses"""
    try:
        client = genai.Client(
          vertexai=True,
          project="pot-test-environment",
          location="europe-central2",
      )

      
        generation_config = {
    "max_output_tokens": 5000,
    "temperature": 0.3,
    "top_p": 0.2,
}
        
        #reader = PdfReader("D:\\Work\\2025\\Preco tool\\preco tool\\PRB offerings on Paint problems.pdf")        #This part will get replaced by RAG in MVP , In POC the reference pdf is just one so putting everything into context
        reader = PdfReader("C:\\Users\\eduar\\INSUS\\pot-sand2\\be1\\PRB offerings on Paint problems.pdf") 
        total_text=[]
        for page in reader.pages:
            text = page.extract_text()
            total_text.append(text)
        pdf_content=''.join(total_text)
        solution_context = pdf_content
        text_context_for_problem=state.get('paint_problems')['results']
       
        model = GenerativeModel(
        "gemini-1.5-pro-002",
        system_instruction=["""Refere to the Solution Context and create a solution for given problem"""]
        )
        chat = model.start_chat()
        language_instruction = get_language_instruction(state["language"])
        text= f"""You are a technical expert in the field of specialty building soltions , Based on the given problem , address the issues by refering to the given context 
             ##
             Given problem : {text_context_for_problem}
             ##
             Reference material : {solution_context}
             ##
             Create a list of comprehensive solution for the given problems using reference material , explain each problem point in detail on why do we want to use the given solution
             """
        full_prompt = f"{text}\n\n{language_instruction}"
        proposed_solutions=chat.send_message( full_prompt,generation_config=generation_config)
        print(proposed_solutions.text)
        print('----------------------------------------------------------------------')
        state["solutions_analysis"] = {
            "results": proposed_solutions.text,
            "summary": "Problems Generated"
        }
        
        print("✅ Solution formed")
        
    except Exception as e:
        print(f"❌ Error in solutions_analysis: {e}")
        state["solutions_analysis"] = {"error": f"{config['errors']['api_error']}: {str(e)}"}
    
    return state

def create_pdf_report_node(state: ReportState) -> ReportState:
    """Create a comprehensive PDF report"""
    print("📄 Creating PDF report...")
    
    try:
        # Create output directory
        output_dir = Path(config["output"]["reports_directory"])
        output_dir.mkdir(exist_ok=True)
        
        # Generate filename
        timestamp = datetime.now().strftime(config["output"]["timestamp_format"])
        filename = config["output"]["filename_template"].format(timestamp=timestamp)
        pdf_path = output_dir / filename
        
        # Create PDF document
        doc = SimpleDocTemplate(str(pdf_path), pagesize=A4)
        story = [PageBreak()]
        styles = getSampleStyleSheet()
        
        # Title
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=config["pdf"]["title_font_size"],
            spaceAfter=30,
            alignment=1  # Center alignment
        )
        story.append(Paragraph("STRUCTURAL ANALYSIS AND REHABILITATION REPORT", title_style))
        story.append(Spacer(1, 20))
        
        # Basic Information Table
        basic_data = [
            ['Project Reference:', state['project_reference']],
            ['Date:', state['date']],
            ['Subject:', state['subject']],
            ['PRB Reference:', state['prb_reference']]
        ]
        
        col_widths = [w * inch for w in config["pdf"]["table_col_widths"]]
        basic_table = Table(basic_data, colWidths=col_widths)
        basic_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0, -1), colors.grey),
            ('TEXTCOLOR', (0, 0), (0, -1), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), config["pdf"]["normal_font_size"]),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (1, 0), (1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        story.append(basic_table)
        story.append(Spacer(1, 20))
        
        # Client and Service Provider Information
        story.append(Paragraph("CLIENT AND SERVICE PROVIDER INFORMATION", styles['Heading2']))
        story.append(Spacer(1, 12))
        
        client_data = [
            ['Client Name:', state['client_details']['name']],
            ['Client Address:', state['client_details']['address']],
            ['Client Contact:', state['client_details']['contact']],
            ['', ''],
            ['Service Provider:', state['service_provider_details']['name']],
            ['Provider Address:', state['service_provider_details']['address']],
            ['Provider Contact:', state['service_provider_details']['contact']]
        ]
        
        client_table = Table(client_data, colWidths=col_widths)
        client_table.setStyle(TableStyle([
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), config["pdf"]["normal_font_size"]),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        story.append(client_table)
        story.append(Spacer(1, 20))
        
        # Structure Analysis Section
        story.append(Paragraph("EXISTING STRUCTURE ANALYSIS", styles['Heading2']))
        story.append(Spacer(1, 12))
        
        if state.get('structure_analysis') and not state['structure_analysis'].get('error'):
            structure_text = state.get('structure_analysis')['results']
           # print(structure_text)#"Structure analysis completed. See detailed results in the analysis data."
            story.append(Paragraph(structure_text, styles['Normal']))
            
            caption_style = ParagraphStyle(
            'Caption',
            parent=styles['Normal'],
            fontSize=10,
            alignment=1,  # Centered
            italic=True,
            textColor=colors.grey
        )
            
            
            img = Image(state.get('structure_analysis')['photos'][0], width=4*inch, height=2*inch)
            img.hAlign = 'CENTER'
            story.append(Spacer(1, 20))
            story.append(Spacer(1, 10))
            story.append(img)
            story.append(Spacer(1, 20))
            story.append(Paragraph("SITE PHOTO", caption_style))
            story.append(Spacer(1, 20))

        else:
            story.append(Paragraph("Structure analysis could not be completed.", styles['Normal']))
        
        story.append(Spacer(1, 20))
        
        # Pathology Analysis Section
        story.append(Paragraph("PATHOLOGY ANALYSIS", styles['Heading2']))
        story.append(Spacer(1, 12))
        
        if state.get('pathology_analysis') and not state['pathology_analysis'].get('error'):
            pathology_text = state.get('pathology_analysis')['results'] #"Pathology analysis completed. Critical issues identified and documented."
            formatted_text = format_llm_text(pathology_text)
            story.append(Paragraph(formatted_text, styles['Normal']))
            
            
            caption_style = ParagraphStyle(
            'Caption',
            parent=styles['Normal'],
            fontSize=10,
            alignment=1,  # Centered
            italic=True,
            textColor=colors.grey
        )
            
            
            img = Image(state.get('pathology_analysis')['photos'][0], width=4*inch, height=2*inch)
            img.hAlign = 'CENTER'
            story.append(Spacer(1, 20))
            story.append(Spacer(1, 10))
            story.append(img)
            story.append(Spacer(1, 20))
            story.append(Paragraph("SITE PHOTO", caption_style))
            story.append(Spacer(1, 20))
            
            img = Image(state.get('pathology_analysis')['photos'][1], width=4*inch, height=2*inch)
            img.hAlign = 'CENTER'
            story.append(Spacer(1, 20))
            story.append(Spacer(1, 10))
            story.append(img)
            story.append(Spacer(1, 20))
            
            story.append(Paragraph("SITE PHOTO", caption_style))
            story.append(Spacer(1, 20))

        
        else:
            story.append(Paragraph("Pathology analysis could not be completed.", styles['Normal']))
        
        story.append(Spacer(1, 20))
        
        # Solutions Section
        story.append(Paragraph("PROPOSED SOLUTIONS AND FINISHING SYSTEMS", styles['Heading2']))
        story.append(Spacer(1, 12))
        
        if state.get('solutions_analysis') and not state['solutions_analysis'].get('error'):
            solutions_text = state.get('solutions_analysis')['results']#"Comprehensive solutions and finishing systems have been developed based on the analysis."
            formatted_text = format_llm_text(solutions_text)
            story.append(Paragraph(formatted_text, styles['Normal']))
        else:
            story.append(Paragraph("Solutions could not be generated.", styles['Normal']))
        
        story.append(Spacer(1, 20))
        
        # Build PDF
        doc.build(story, onFirstPage=draw_cover_page, onLaterPages=draw_header_footer)
        
        state["pdf_path"] = str(pdf_path)
        print(f"✅ PDF report created: {pdf_path}")
        
    except Exception as e:
        print(f"❌ Error creating PDF report: {e}")
        state["pdf_path"] = f"Error: {config['errors']['pdf_error']}: {str(e)}"
    
    return state




def should_continue(state: ReportState) -> str:
    """Determine the next step in the workflow"""
    if "error" in state.get("structure_analysis", {}):
        return "end"
    if "error" in state.get("pathology_analysis", {}):
        return "end"
    return "continue"

# Create the workflow graph
def create_report_graph():
    """Create the report generation workflow graph"""
    
    # Create the graph
    workflow = StateGraph(ReportState)
    
    # Add nodes
    workflow.add_node("start", start_node)
    workflow.add_node("basic_details", generate_basic_details_node)
    workflow.add_node("analyze_structure", analyze_structure_node)
    workflow.add_node("analyze_pathologies", analyze_pathologies_node)
    workflow.add_node("analyse_problems",analyse_problems)
    workflow.add_node("generate_solutions", generate_solutions_node)
    workflow.add_node("create_pdf", create_pdf_report_node)
    
    # Add edges
    workflow.set_entry_point("start")
    workflow.add_edge("start", "basic_details")
    workflow.add_edge("basic_details", "analyze_structure")
    workflow.add_edge("analyze_structure", "analyze_pathologies")
    workflow.add_edge("analyze_pathologies", "analyse_problems")
    workflow.add_edge("analyse_problems","generate_solutions")
    workflow.add_edge("generate_solutions", "create_pdf")
    workflow.add_edge("create_pdf", END)
    
    # Compile the graph
    app = workflow.compile(checkpointer=MemorySaver())
    
    return app

# Main function to run the report generation
def generate_report(
    image_paths: List[str],
    project_reference: str,
    date: str,
    subject: str,
    prb_reference: str,
    client_details: Dict[str, str],
    service_provider_details: Dict[str, str]
):
    """Generate a structural analysis report"""
    
    if image_paths is None:
        image_paths = []
    
    # Validate configuration
    if not validate_config():
        print("⚠️  Configuration validation failed. Please check your settings.")
    
    # Validate image count
    if len(image_paths) > config["validation"]["max_images_per_report"]:
        print(f"⚠️  Warning: Too many images provided. Maximum allowed: {config['validation']['max_images_per_report']}")
        image_paths = image_paths[:config["validation"]["max_images_per_report"]]
    
    # Create the graph
    app = create_report_graph()
    
    # Initialize state
    initial_state = {
        "images": image_paths,
        "client_details": client_details,
        "service_provider_details": service_provider_details,
        "project_reference": project_reference,
        "date": date,
        "subject": subject,
        "project_dimensions": {},  # or fill as needed
        "prb_reference": prb_reference,
        "structure_analysis": {},
        "pathology_analysis": {},
        "solutions_analysis": {},
        "paint_problems": {},
        "pdf_path": "",
        "messages": [],
        "language": "",
    }
    
    # Run the workflow
    print("🔄 Starting report generation workflow...")
    #result = app.invoke(initial_state)
    final_result_dicts=[]
    for event in app.stream(initial_state,{"configurable": {"thread_id": "1"}}):
        for value in event.values():
            final_result_dicts.append(value)
    result = reduce(lambda d1, d2: {**d1, **d2}, final_result_dicts)
    
    print("🎉 Report generation completed!")
    print(f"📄 PDF Report: {result.get('pdf_path', 'Not created')}")
    
    return result


app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/generate-report")
async def generate_report_api(
    project_reference: str = Form(...),
    date: str = Form(...),
    subject: str = Form(...),
    prb_reference: str = Form(...),
    client_name: str = Form(...),
    client_address: str = Form(...),
    client_contact: str = Form(...),
    provider_name: str = Form(...),
    provider_address: str = Form(...),
    provider_contact: str = Form(...),
    images: list[UploadFile] = File(...)
):
    # Save uploaded images to temp files
    temp_dir = tempfile.mkdtemp()
    image_paths = []
    for img in images:
        img_path = os.path.join(temp_dir, img.filename)
        with open(img_path, "wb") as buffer:
            shutil.copyfileobj(img.file, buffer)
        image_paths.append(img_path)

    # Prepare state for report
    # (You may want to update generate_report to accept all details, but for now, just pass images)
    client_details = {
        "name": client_name,
        "address": client_address,
        "contact": client_contact,
    }
    service_provider_details = {
        "name": provider_name,
        "address": provider_address,
        "contact": provider_contact,
    }
    result = generate_report(
        image_paths,
        project_reference,
        date,
        subject,
        prb_reference,
        client_details,
        service_provider_details
    )
    # Optionally, clean up temp_dir after use
    return JSONResponse(content=result)