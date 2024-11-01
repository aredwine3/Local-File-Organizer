import os

from setuptools import find_packages, setup
from setuptools.command.install import install


class PostInstallCommand(install):
    """Post-installation for installation mode."""

    def run(self):
        # Call the standard install method
        install.run(self)
        # Create the .env file
        env_content = """# Set to 'true' to use OpenAI API, 'false' to use local models
USE_OPENAI=false
# Your OpenAI API key
OPENAI_API_KEY=your_openai_api_key
# Maximum number of tokens
MAX_TOKENS=5000
# Maximum length for filenames and folder names
MAX_FILENAME_LENGTH=5
MAX_FOLDERNAME_LENGTH=4
"""
        with open(".env", "w") as f:
            f.write(env_content)


setup(
    name="local-file-organizer",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "cmake",
        "pytesseract",
        "PyMuPDF",
        "python-docx",
        "pandas",
        "openpyxl",
        "xlrd",
        "nltk",
        "rich",
        "python-pptx",
        "openai",
        "python-dotenv",
        "opencv-python",
    ],
    cmdclass={
        "install": PostInstallCommand,
    },
)
