# QAT System Application

## Overview
**qat-system-app** is a comprehensive QAT system Flask application that provides various features such as text processing, PDF handling, and natural language processing (NLP). The application is designed to streamline and enhance learning experiences. It integrates various libraries and frameworks to provide robust functionalities for text processing, data parsing, web interactions, and more, making it an essential tool for students, researchers and even faculty members.

## Features
- **Web Framework**: Build web applications and interfaces using `Flask`.
- **AI/ML**: Machine Learning capabilities with PyTorch, text generation and language understanding using Hugging Face's `transformers`
- **NLP**: Natural language processing with NLTK and TextBlob
- **Text Processing**: Utilize `nltk` for advanced natural language processing.
- **Data Parsing and Manipulation**: Leverage `lxml` for XML and HTML parsing.
- **Image Processing**: Handle image files with `Pillow`.
- **HTTP Requests**: Simplify HTTP interactions with `requests`.
- **Template Rendering**: Use `Jinja2` for versatile template rendering.
- DOCX file creation and manipulation with `docx` and `python-docx`
- Web server built with Flask

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/techgrandmaster/qat-sys-app.git
   cd qat-sys-app
   ```

2. Create a virtual environment and activate it:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```
   
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Dependencies

- **Flask~=3.0.3**: A micro web framework written in Python. It is easy to set up and extendable via various plugins.

- **PyPDF2~=3.0.1**: A library for reading and manipulating PDF files. It supports tasks such as splitting and merging PDFs, extracting text, and more.

- **docx~=0.2.4**: A library for creating and updating .docx files. It simplifies the creation of Word documents programmatically.

- **python-docx~=0.8.11**: Another library for creating and updating Word (.docx) files. It includes more comprehensive features compared to `docx`.

- **nltk~=3.9.1**: The Natural Language Toolkit is a leading platform for building Python programs to work with human language data. It provides easy-to-use interfaces to over 50 corpora and lexical resources.

- **textblob~=0.18.0.post0**: A library for processing textual data. It provides a simple API for diving into common natural language processing (NLP) tasks such as part-of-speech tagging, noun phrase extraction, sentiment analysis, classification, translation, and more.

- **transformers~=4.45.2**: Hugging Face's Transformers library provides thousands of pre-trained models to perform tasks on texts such as classification, information extraction, question answering, summarization, translation, and text generation in over 100 languages.

- **setuptools~=75.1.0**: A package development and distribution library. It supplies utilities to build and distribute Python packages, especially those that have dependencies.

- **torch~=2.4.1**: PyTorch is an optimized tensor library for deep learning using GPUs and CPUs. It enables efficient numerical computation and is widely used in the machine learning community for building and training neural networks.

## Usage

1. Run the Flask application:
   ```bash
   export FLASK_APP=app.py
   flask run
   ```
   The server will start running at `http://127.0.0.1:5000/`.

2. Access the application via your web browser at the provided URL.

## Contributing

Feel free to fork this repository and submit pull requests. For any issues, please open an issue on GitHub.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contact

For any questions or suggestions, please contact [biolaomolaja@gmail.com](mailto:biolaomolaja@gmail.com).