import streamlit as st
import requests
from datetime import date
from PIL import Image
from fastapi.responses import FileResponse
import os
# --- Sidebar Branding ---
st.sidebar.image(Image.open('PRBLogo.jpg'), width=5, use_column_width=True)
st.sidebar.markdown('<br><br>', unsafe_allow_html=True)
st.sidebar.markdown('**POC developed by**')
st.sidebar.image(Image.open('StrategicPortfolio.png'), width=120, use_column_width=True)



# --- Main Title and PRB Logo ---
st.title('STRUCTURAL ANALYSIS AND REHABILITATION REPORT')

# --- Section 1: Project Details ---
st.header('1. Project Details')
col1, col2 = st.columns(2)
with col1:
    project_reference = st.text_input('Project Reference', 'PRJ-001')
    date_value = st.date_input('Date', date(2025, 6, 30))
with col2:
    subject = st.text_input('Subject', 'Structural Analysis and Rehabilitation Report')
    prb_reference = st.text_input('PRB Reference', 'PRB-001')

# --- Section 2: Client Details ---
st.header('2. Client Details')
client_name = st.text_input('Client Name', 'Default Client')
client_address = st.text_input('Client Address', 'Default Address')
client_contact = st.text_input('Client Contact', 'Default Contact')

# --- Section 3: Service Provider Details ---
st.header('3. Service Provider Details')
provider_name = st.text_input('Service Provider', 'Default Service Provider')
provider_address = st.text_input('Provider Address', 'Default Address')
provider_contact = st.text_input('Provider Contact', 'Default Contact')

# --- Run Graph Button ---
st.markdown('---')
st.subheader('Run Report Generation')

# --- Image Upload Section ---
st.markdown('---')
st.subheader('Upload Sample Images (min 2 required)')
uploaded_images = st.file_uploader('Choose images', type=['png', 'jpg', 'jpeg'], accept_multiple_files=True)

if uploaded_images is not None and len(uploaded_images) > 0:
    st.write(f"{len(uploaded_images)} image(s) selected.")

# --- Button and API Call ---
if st.button('Generate Report'):
    if not uploaded_images or len(uploaded_images) < 2:
        st.error('Please upload at least 2 images before running the report.')
    else:
        # Prepare form data for FastAPI endpoint
        form_data = {
            'project_reference': project_reference,
            'date': date_value.strftime('%Y-%m-%d'),
            'subject': subject,
            'prb_reference': prb_reference,
            'client_name': client_name,
            'client_address': client_address,
            'client_contact': client_contact,
            'provider_name': provider_name,
            'provider_address': provider_address,
            'provider_contact': provider_contact,
        }
        files = [('images', (img.name, img, img.type)) for img in uploaded_images]
        try:
            response = requests.post(
                'http://localhost:8000/generate-report',
                data=form_data,
                files=files
            )
            if response.status_code == 200:
                st.success('Report generated successfully!')
                result = response.json()
                st.json(result)
                pdf_path = result.get('pdf_path')
                if pdf_path:
                    # Fetch the PDF file from the backend
                    download_url = f"http://localhost:8000/download-report?path={pdf_path}"
                    pdf_response = requests.get(download_url)
                    if pdf_response.status_code == 200:
                        st.download_button(
                            label="Download PDF Report",
                            data=pdf_response.content,
                            file_name=os.path.basename(pdf_path),
                            mime="application/pdf"
                        )
                    else:
                        st.error("Could not fetch the PDF for download.")
            else:
                st.error(f'Error: {response.status_code} - {response.text}')
        except Exception as e:
            st.error(f'Failed to connect to backend: {e}') 