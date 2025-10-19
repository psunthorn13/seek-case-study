import pandas as pd
import streamlit as st
import PyPDF2
import docx
import io

from vector_database.pinecone_handler import PineconeHandler
from llm.job_analyser_llm import JobAnalyser

# Set page configuration
st.set_page_config(layout="wide", page_title="Job Browser")
st.header("Job Browser")

# Initialize session state for job data and selected job
if 'job_df' not in st.session_state:
    st.session_state.job_df = None
if 'selected_job_id' not in st.session_state:
    st.session_state.selected_job_id = None
if 'resume_text' not in st.session_state:
    st.session_state.resume_text = None
if 'search_mode' not in st.session_state:
    st.session_state.search_mode = None
if 'job_analysis' not in st.session_state:
    st.session_state.job_analysis = None
if 'show_job_description' not in st.session_state:
    st.session_state.show_job_description = True


def extract_text_from_pdf(pdf_file):
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text


def extract_text_from_docx(docx_file):
    doc = docx.Document(docx_file)
    text = ""
    for paragraph in doc.paragraphs:
        text += paragraph.text + "\n"
    return text


def escape_markdown(text):
    if pd.notna(text):
        return text.replace('$', '\\$').replace('*', '\\*').replace('_', '\\_')
    return ""


# Create tabs for different search methods
tab1, tab2 = st.tabs(["Search with Resume", "Search by Query"])

# Resume upload tab
with tab1:
    st.subheader("Upload your resume to find matching jobs")
    resume = st.file_uploader("Upload Your Resume", type=["pdf", "docx"], key="file_uploader")

    if resume is not None:
        # Save the file contents in a BytesIO object
        file_bytes = io.BytesIO(resume.read())

        try:
            # Extract text based on file type
            if resume.type == "application/pdf":
                st.session_state.resume_text = extract_text_from_pdf(file_bytes)
            elif resume.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
                st.session_state.resume_text = extract_text_from_docx(file_bytes)

            # Display confirmation
            st.success("Resume uploaded and processed successfully!")

            # Optionally show a preview of the text
            with st.expander("Preview Resume Text"):
                st.text(st.session_state.resume_text)

            st.session_state.search_mode = "resume"
        except Exception as e:
            st.error(f"Error processing resume: {e}")

    resume_search_button = st.button("SEEK Jobs", key="resume_seek")

    if resume_search_button:
        if st.session_state.resume_text:
            pinecone_query = f"Resume: {st.session_state.resume_text}"

            handler = PineconeHandler(
                index_name='seek-ads'
            )
            st.session_state.job_df = handler.search(
                namespace="job-description-namespace",
                query=pinecone_query,
                top_k=20
            )
        else:
            st.warning("Please upload your resume first")

# Keyword search tab
with tab2:
    st.subheader("Search for jobs by keywords")
    keyword_search_query = st.text_input("Enter keywords or job title", key="keyword_search_input")
    keyword_search_button = st.button("SEEK Jobs", key="keyword_seek")

    if keyword_search_button and keyword_search_query:
        st.session_state.search_mode = "keyword"
        handler = PineconeHandler(
            index_name='seek-ads'
        )
        st.session_state.job_df = handler.search(
            namespace="job-description-namespace",
            query=keyword_search_query,
            top_k=20
        )
    elif keyword_search_button:
        st.warning("Please enter search keywords")

# Create two columns for the layout
col1, col2 = st.columns(2)

# Job cards in the left column
with col1:
    if st.session_state.job_df is not None:
        st.markdown("---")
        for i, row in st.session_state.job_df.iterrows():
            with st.container():
                card = st.container()
                with card:
                    st.markdown(f"#### {escape_markdown(row['title'])}")
                    if pd.notna(row['metadata.additionalSalaryText']):
                        st.markdown(f"**Salary:** {escape_markdown(row['metadata.additionalSalaryText'])}")
                    if pd.notna(row['metadata.location.name']):
                        st.markdown(f"**Location:** {escape_markdown(row['metadata.location.name'])}")
                    for bullet in ['metadata.standout.bullet1', 'metadata.standout.bullet2',
                                   'metadata.standout.bullet3']:
                        if pd.notna(row[bullet]):
                            st.markdown(f"- {escape_markdown(row[bullet])}")
                    if st.button(f"View Details", key=f"view_{row['id']}"):
                        st.session_state.selected_job_id = row['id']
                        # Clear any existing analysis when viewing job details
                        st.session_state.job_analysis = None
                        st.session_state.show_job_description = True

                st.markdown("---")

# Job details in the right column
with col2:
    st.markdown("---")
    if st.session_state.selected_job_id and st.session_state.job_df is not None:
        job = st.session_state.job_df[st.session_state.job_df['id'] == st.session_state.selected_job_id].iloc[0]
        st.markdown(f"#### {escape_markdown(job['title'])}")


        # Add the Analyse Resume Fit button here in the right column
        if st.button(f"Analyse Resume Fit", key=f"analyse_{st.session_state.selected_job_id}"):
            st.markdown("---")

            # Check if we have a resume to analyze
            if st.session_state.resume_text:
                job_description = job['content']

                with st.spinner("Analysing resume fit..."):
                    job_analyser = JobAnalyser(
                        user_resume=st.session_state.resume_text,
                        job_description=job_description
                    )
                    analysis = job_analyser.generate_analysis()
                    st.session_state.job_analysis = analysis
                    # Set flag to hide job description when showing analysis
                    st.session_state.show_job_description = False
            else:
                st.warning("Please upload your resume first to analyze fit")

        # Display job analysis if available
        if st.session_state.job_analysis:
            st.markdown("#### AI-Generated Resume Fit Analysis")
            st.markdown(st.session_state.job_analysis)

        # Display job description only when not showing analysis
        if st.session_state.show_job_description:
            st.markdown("---")
            st.markdown(job['content'], unsafe_allow_html=True)
