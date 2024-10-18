from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="qat-system-app",
    version="0.1.0",
    author="Abiola Omolaja",
    author_email="biolaomolaja@gmail.com",
    description="A Flask app for QAT system",
    long_description="QAT system app is a comprehensive QAT system Flask application that provides various features such as text processing, PDF handling, and natural language processing (NLP). The application is designed to streamline and enhance learning experiences. It integrates various libraries and frameworks to provide robust functionalities for text processing, data parsing, web interactions, and more, making it an essential tool for students, researchers and even faculty members.",
    long_description_content_type="text/markdown",
    url="https://github.com/techgrandmaster/qat-system-app",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requires=[
        "Flask",
        "Jinja2",
        "PyYAML",
        "Werkzeug",
        "click",
        "lxml",
        "nltk",
        "numpy",
        "pillow",
        "requests",

    ],
    entry_points={
        'console_scripts': [
            'qat-system-app=qat_system_app.main:main',
        ],
    },
)
