import io

import PyPDF2
import docx
import pandas as pd
import streamlit as st

from helpers.constant import FEATURE_OPTIONS
from helpers.job_analyser_llm import JobAnalyser
from helpers.pinecone_handler import PineconeHandler


class DocumentProcessor:
    """Class to handle document processing and text extraction"""

    @staticmethod
    def extract_text_from_pdf(pdf_file):
        # Extract text from PDF file
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        return "".join([page.extract_text() for page in pdf_reader.pages])

    @staticmethod
    def extract_text_from_docx(docx_file):
        # Extract text from DOCX file
        doc = docx.Document(docx_file)
        return "\n".join([paragraph.text for paragraph in doc.paragraphs])

    @staticmethod
    def process_resume(resume_file):
        """Process uploaded resume file and extract text"""
        file_bytes = io.BytesIO(resume_file.read())

        if resume_file.type == "application/pdf":
            return DocumentProcessor.extract_text_from_pdf(file_bytes)
        elif resume_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            return DocumentProcessor.extract_text_from_docx(file_bytes)
        else:
            raise ValueError(f"Unsupported file type: {resume_file.type}")


class JobSearchApp:
    """Main application class for Smart Job Search"""

    def __init__(self):
        self._initialise_session_state()
        self._setup_page_layout()

    @staticmethod
    def _initialise_session_state():
        """initialise session state variables"""
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
        if 'search_query' not in st.session_state:
            st.session_state.search_query = None

    @staticmethod
    def _setup_page_layout():
        """Configure the page layout and title"""
        st.set_page_config(layout="wide", page_title="Smart Job Search")
        st.header("Smart Job Search")

    def run(self):
        """Main method to run the application"""
        # Create tabs for different search methods
        tab1, tab2 = st.tabs(["Search with Resume", "Search by Query"])

        # Handle each tab's content
        with tab1:
            self._display_resume_search_tab()

        with tab2:
            self._display_keyword_search_tab()

        # Display filters
        self._display_search_filters()

        # Search button outside the tabs
        if st.button("SEEK Jobs"):

            # Reset selected job
            st.session_state.selected_job_id = None

            # Check if both search methods are empty
            has_resume = st.session_state.resume_text is not None
            has_keywords = st.session_state.search_query is not None and st.session_state.search_query.strip() != ""

            # Handle different search scenarios
            if not has_resume and not has_keywords:
                st.warning("Please either upload your resume or enter search keywords")
            elif has_resume and not has_keywords:
                pinecone_query = f"""Resume: {st.session_state.resume_text} Query: {st.session_state.search_query}"""
                self._perform_job_search(pinecone_query)
            else:
                pinecone_query = st.session_state.resume_text or st.session_state.search_query
                self._perform_job_search(pinecone_query)

        # Create two columns for displaying jobs and details
        col1, col2 = st.columns(2)

        with col1:
            self._display_job_listings()

        with col2:
            self._display_job_details()

    @staticmethod
    def _display_resume_search_tab():
        """Display and handle resume upload tab"""
        st.subheader("Upload your resume to find matching jobs")
        resume = st.file_uploader("Upload Your Resume", type=["pdf", "docx"], key="file_uploader")

        # Check if the file has been removed (resume is None)
        if resume is None and st.session_state.resume_text is not None:

            # Reset the resume text when file is removed
            st.session_state.resume_text = None

        elif resume is not None:
            try:
                # Process the uploaded resume
                st.session_state.resume_text = DocumentProcessor.process_resume(resume)
                st.success("Resume uploaded and processed successfully!")

                with st.expander("Preview Resume Text"):
                    st.text(st.session_state.resume_text)

            except Exception as e:
                st.error(f"Error processing resume: {e}")

    @staticmethod
    def _display_keyword_search_tab():
        """Display and handle keyword search tab"""
        st.subheader("Describe what job you're looking for")
        keyword_search_query = st.text_input("Describe what job you're looking for", key="keyword_search_input")
        st.session_state.search_query = keyword_search_query

    @staticmethod
    def _display_search_filters():
        """Display job search filters in 3 columns"""
        col1, col2, col3 = st.columns(3)

        # Let the multiselect widgets handle their session state connection automatically through keys
        col1.multiselect(
            "Location",
            options=FEATURE_OPTIONS['Location'],
            key="location_filter"
        )

        col2.multiselect(
            "Work Type",
            options=FEATURE_OPTIONS['work_type'],
            key="work_type_filter"
        )

        col3.multiselect(
            "Classification",
            options=FEATURE_OPTIONS['classification'],
            key="classification_filter"
        )

    @staticmethod
    def _perform_job_search(query, top_k=20):
        """Perform search using Pinecone vector database"""

        # Build filter dict for metadata filtering
        filter_dict = {}

        if st.session_state.location_filter:
            filter_dict["metadata.location.name"] = {"$in": st.session_state.location_filter}

        if st.session_state.work_type_filter:
            filter_dict["metadata.workType.name"] = {"$in": st.session_state.work_type_filter}

        if st.session_state.classification_filter:
            filter_dict["metadata.classification.name"] = {"$in": st.session_state.classification_filter}

        # Perform the search with filters
        handler = PineconeHandler(index_name='seek-ads')
        st.session_state.job_df = handler.search(
            namespace="job-description-namespace",
            query=query,
            top_k=top_k,
            filter_dict=filter_dict
        )

    def _display_job_listings(self):
        """Display job listing cards in the left column"""
        if st.session_state.job_df is not None:
            st.markdown("---")
            for i, row in st.session_state.job_df.iterrows():
                self._render_job_card(row)

    def _render_job_card(self, job):
        """Render a single job card"""
        with st.container():
            card = st.container()
            with card:
                st.markdown(f"#### {self._escape_markdown(job['title'])}")

                if pd.notna(job['metadata.additionalSalaryText']):
                    st.markdown(f"**Salary:** {self._escape_markdown(job['metadata.additionalSalaryText'])}")

                if pd.notna(job['metadata.location.name']):
                    st.markdown(f"**Location:** {self._escape_markdown(job['metadata.location.name'])}")

                # Display bullet points if available
                for bullet in ['metadata.standout.bullet1', 'metadata.standout.bullet2', 'metadata.standout.bullet3']:
                    if pd.notna(job[bullet]):
                        st.markdown(f"- {self._escape_markdown(job[bullet])}")

                # View details button
                if st.button(f"View Details", key=f"view_{job['id']}"):
                    st.session_state.selected_job_id = job['id']
                    st.session_state.job_analysis = None
                    st.session_state.show_job_description = True

            st.markdown("---")

    def _display_job_details(self):
        """Display job details in the right column"""
        if st.session_state.selected_job_id and st.session_state.job_df is not None:
            job = st.session_state.job_df[st.session_state.job_df['id'] == st.session_state.selected_job_id].iloc[0]
            st.markdown(f"#### {self._escape_markdown(job['title'])}")

            # Analyze Resume Fit button
            if st.button(f"Analyse Resume Fit", key=f"analyse_{st.session_state.selected_job_id}"):
                self._analyse_resume_fit(job)

            # Display either job analysis or job description
            if st.session_state.job_analysis and not st.session_state.show_job_description:
                st.markdown("#### AI-Generated Resume Fit Analysis")
                st.markdown(st.session_state.job_analysis)

            if st.session_state.show_job_description:
                st.markdown("---")
                st.markdown(job['content'], unsafe_allow_html=True)

    @staticmethod
    def _analyse_resume_fit(job):
        """Analyze resume fit for the selected job"""
        st.markdown("---")

        if st.session_state.resume_text:
            job_description = job['content']

            with st.spinner("Analysing resume fit..."):
                job_analyser = JobAnalyser(
                    user_resume=st.session_state.resume_text,
                    job_description=job_description
                )
                analysis = job_analyser.generate_analysis()
                st.session_state.job_analysis = analysis
                st.session_state.show_job_description = False
        else:
            st.warning("Please upload your resume first to analyse fit")

    @staticmethod
    def _escape_markdown(text):
        """Escape special characters in Markdown text"""
        if pd.notna(text):
            return text.replace('$', '\\$').replace('*', '\\*').replace('_', '\\_')
        return ""


# Main entry point
if __name__ == "__main__":
    app = JobSearchApp()
    app.run()
