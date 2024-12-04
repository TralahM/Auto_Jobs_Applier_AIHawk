import os
import tempfile
import textwrap
import time
from src.ai_hawk.libs.resume_and_cover_builder.utils import LoggerChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed
from loguru import logger
from pathlib import Path
from langchain_core.prompt_values import StringPromptValue
from langchain_core.runnables import RunnablePassthrough
from langchain_text_splitters import TokenTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from lib_resume_builder_AIHawk.config import global_config
from langchain_community.document_loaders import TextLoader
import logging
import re  # Per la parsing regex, soprattutto in `parse_wait_time_from_error_message`
from requests.exceptions import HTTPError as HTTPStatusError  # Gestione degli errori HTTP
import openai

# Carica le variabili d'ambiente dal file .env
load_dotenv()

# Configura il file di log
log_folder = 'log/resume/gpt_resume'
if not os.path.exists(log_folder):
    os.makedirs(log_folder)
log_path = Path(log_folder).resolve()
logger.add(log_path / "gpt_resume.log", rotation="1 day", compression="zip", retention="7 days", level="DEBUG")


class LLMResumer:
    def __init__(self, openai_api_key, strings):
        self.llm_cheap = LoggerChatModel(
            ChatOpenAI(
                model_name="gpt-4o-mini", openai_api_key=openai_api_key, temperature=0.4
            )
        )
        self.strings = strings
        self.llm_embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)  # Inizializza gli embeddings

    @staticmethod
    def _preprocess_template_string(template: str) -> str:
        """
        Preprocessa la stringa del template rimuovendo gli spazi bianchi iniziali e l'indentazione.
        Args:
            template (str): La stringa del template da preprocessare.
        Returns:
            str: La stringa del template preprocessata.
        """
        return textwrap.dedent(template)
    
    def get_job_description_from_url(self, url_job_description):
        from lib_resume_builder_AIHawk.utils import create_driver_selenium
        driver = create_driver_selenium()
        driver.get(url_job_description)
        time.sleep(3)
        body_element = driver.find_element("tag name", "body")
        response = body_element.get_attribute("outerHTML")
        driver.quit()
        with tempfile.NamedTemporaryFile(delete=False, suffix=".html", mode="w", encoding="utf-8") as temp_file:
            temp_file.write(response)
            temp_file_path = temp_file.name
        try:
            loader = TextLoader(temp_file_path, encoding="utf-8", autodetect_encoding=True)
            document = loader.load()
        finally:
            os.remove(temp_file_path)
        text_splitter = TokenTextSplitter(chunk_size=500, chunk_overlap=50)
        all_splits = text_splitter.split_documents(document)
        vectorstore = FAISS.from_documents(documents=all_splits, embedding=self.llm_embeddings)
        prompt = PromptTemplate(
            template="""
            You are an expert job description analyst. Your role is to meticulously analyze and interpret job descriptions. 
            After analyzing the job description, answer the following question in a clear, and informative manner.
            
            Question: {question}
            Job Description: {context}
            Answer:
            """,
            input_variables=["question", "context"]
        )
        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)
        context_formatter = vectorstore.as_retriever() | format_docs
        question_passthrough = RunnablePassthrough()
        chain_job_description = prompt | self.llm_cheap | StrOutputParser()
        summarize_prompt_template = self._preprocess_template_string(self.strings.summarize_prompt_template)
        prompt_summarize = ChatPromptTemplate.from_template(summarize_prompt_template)
        chain_summarize = prompt_summarize | self.llm_cheap | StrOutputParser()
        qa_chain = (
            {
                "context": context_formatter,
                "question": question_passthrough,
            }
            | chain_job_description
            | (lambda output: {"text": output})
            | chain_summarize
        )
        result = qa_chain.invoke("Provide, full job description")
        self.job_description = result

    def extract_company_name(self):
        """
        Estrae il nome dell'azienda dalla descrizione del lavoro.
        Returns:
            str: Il nome dell'azienda estratto.
        """
        return self._extract_information("What is the company name in this job description?")

    def extract_role(self):
        """
        Estrae il ruolo/titolo ricercato dalla descrizione del lavoro.
        Returns:
            str: Il ruolo/titolo estratto.
        """
        return self._extract_information("What is the role or title being sought in this job description?")

    def extract_location(self):
        """
        Estrae la località dalla descrizione del lavoro.
        Returns:
            str: La località estratta.
        """
        return self._extract_information("What is the location mentioned in this job description?")

    def extract_recruiter_email(self):
        """
        Estrae l'email del recruiter dalla descrizione del lavoro.
        Returns:
            str: L'email del recruiter estratta.
        """
        return self._extract_information("What is the recruiter's email address in this job description?")

    def _extract_information(self, question):
        """
        Metodo generico per estrarre informazioni specifiche basate sulla domanda fornita.
        Args:
            question (str): La domanda da porre al LLM per l'estrazione.
        Returns:
            str: L'informazione estratta.
        """
        if not hasattr(self, 'job_description'):
            raise ValueError("Job description not found. Please run get_job_description_from_url first.")

        prompt = PromptTemplate(
            template="""
            You are an expert in extracting specific information from job descriptions. 
            Carefully read the job description below and provide a clear and concise answer to the question.

            Job Description: {job_description}

            Question: {question}
            Answer:
            """,
            input_variables=["job_description", "question"]
        )

        chain = prompt | self.llm_cheap | StrOutputParser()
        result = chain.invoke({
            "job_description": self.job_description,
            "question": question
        })
        return result.strip()

    def extract_all_details(self):
        """
        Estrae il nome dell'azienda, il ruolo, la località e l'email del recruiter dalla descrizione del lavoro.
        Returns:
            dict: Un dizionario contenente tutti i dettagli estratti.
        """
        details = {}
        details['company_name'] = self.extract_company_name()
        details['role'] = self.extract_role()
        details['location'] = self.extract_location()
        details['recruiter_email'] = self.extract_recruiter_email()
        return details
