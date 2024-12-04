"""
This module contains the FacadeManager class, which is responsible for managing the interaction between the user and the other components of the application.
"""
# app/libs/resume_and_cover_builder/manager_facade.py
import logging
import os

import inquirer
from pathlib import Path

from src.utils.chrome_utils import HTML_to_PDF
from .config import global_config

class ResumeFacade:
    def __init__(self, api_key, style_manager, resume_generator, resume_object, output_path):
        """
        Initialize the FacadeManager with the given API key, style manager, resume generator, resume object, and log path.
        Args:
            api_key (str): The OpenAI API key to be used for generating text.
            style_manager (StyleManager): The StyleManager instance to manage the styles.
            resume_generator (ResumeGenerator): The ResumeGenerator instance to generate resumes and cover letters.
            resume_object (str): The resume object to be used for generating resumes and cover letters.
            output_path (str): The path to the log file.
        """
        lib_directory = Path(__file__).resolve().parent
        global_config.STRINGS_MODULE_RESUME_PATH = lib_directory / "resume_prompt/strings_feder-cr.py"
        global_config.STRINGS_MODULE_RESUME_JOB_DESCRIPTION_PATH = lib_directory / "resume_job_description_prompt/strings_feder-cr.py"
        global_config.STRINGS_MODULE_COVER_LETTER_JOB_DESCRIPTION_PATH = lib_directory / "cover_letter_prompt/strings_feder-cr.py"
        global_config.STRINGS_MODULE_NAME = "strings_feder_cr"
        global_config.STYLES_DIRECTORY = lib_directory / "resume_style"
        global_config.LOG_OUTPUT_FILE_PATH = output_path
        global_config.API_KEY = api_key
        self.style_manager = style_manager
        self.resume_generator = resume_generator
        self.resume_generator.set_resume_object(resume_object)
        self.selected_style = None  # Proprietà per memorizzare lo stile selezionato
    
    def set_driver(self, driver):
         self.driver = driver

    def prompt_user(self, choices: list[str], message: str) -> str:
        """
        Prompt the user with the given message and choices.
        Args:
            choices (list[str]): The list of choices to present to the user.
            message (str): The message to display to the user.
        Returns:
            str: The choice selected by the user.
        """
        questions = [
            inquirer.List('selection', message=message, choices=choices),
        ]
        return inquirer.prompt(questions)['selection']

    def prompt_for_text(self, message: str) -> str:
        """
        Prompt the user to enter text with the given message.
        Args:
            message (str): The message to display to the user.
        Returns:
            str: The text entered by the user.
        """
        questions = [
            inquirer.Text('text', message=message),
        ]
        return inquirer.prompt(questions)['text']

    def choose_style(self) -> None:
        """
        Prompt the user to choose a style for the resume.
        """
        styles = self.style_manager.get_styles()
        if not styles:
            print("No styles available")
            return None
        formatted_choices = self.style_manager.format_choices(styles)
        selected_choice = self.prompt_user(formatted_choices, "Which style would you like to adopt?")
        self.selected_style = selected_choice.split(' (')[0]

    def create_resume_pdf(self, job_description_text=None) -> bytes:
        """
        Create a resume PDF using the selected style and the given job description text.
        Args:
            job_description_text (str): The job description text to include in the resume.
        Returns:
            bytes: The PDF content as bytes.
        """
        if self.selected_style is None:
            raise ValueError("Devi scegliere uno stile prima di generare il PDF.")
        
        style_path = self.style_manager.get_style_path(self.selected_style)

        if job_description_text is None:
            html_resume = self.resume_generator.create_resume(style_path)
        else:
            html_resume = self.resume_generator.create_resume_job_description_text(style_path, job_description_text)
        result = HTML_to_PDF(html_resume, self.driver)
        self.driver.quit()
        return result
    
    def create_cover_letter(self, job_description_text: str) -> None:
        """
        Create a cover letter based on the given job description text and format.
        Args:
            job_description_text (str): The job description text to include in the cover letter.
        """
        style_path = self.style_manager.get_style_path()
        cover_letter_html = self.resume_generator.create_cover_letter_job_description(style_path, job_description_text)
        result = HTML_to_PDF(cover_letter_html, self.driver)
        self.driver.quit()
        return result
