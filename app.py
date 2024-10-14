import os
import random
import sqlite3
import uuid
from difflib import SequenceMatcher

import PyPDF2
import docx
import nltk
from flask import Flask, jsonify, render_template
from flask import session
from flask_wtf import FlaskForm
from textblob import TextBlob
from transformers import pipeline
from werkzeug.utils import secure_filename
from wtforms import FileField, SubmitField
from wtforms.fields.simple import TextAreaField
from wtforms.validators import InputRequired

app = Flask(__name__)
app.config['SECRET_KEY'] = 'superstructure'
app.config['UPLOAD_FOLDER'] = 'static/files'


class UploadFileForm(FlaskForm):
    file = FileField('File', validators=[InputRequired()])
    submit = SubmitField('Upload file')


class QueryForm(FlaskForm):
    question = TextAreaField('Enter your query below', validators=[InputRequired()])
    submit = SubmitField('Go')


class EvaluateForm(FlaskForm):
    user_answer = TextAreaField('Enter your answer below', validators=[InputRequired()])
    submit = SubmitField('Evaluate')


# Download the Punkt tokenizer model for sentence splitting
nltk.download('punkt')
nltk.download('punkt_tab')

nltk.download('averaged_perceptron_tagger')  # Download the part-of-speech tagger

# Initialize the question answering and summarization pipelines with specified model
question_answering_pipeline = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")
summarization_pipeline = pipeline("summarization", model="facebook/bart-large-cnn")

# Database setup
conn = sqlite3.connect('qat_database.db')
cursor = conn.cursor()
cursor.execute('''
    CREATE TABLE IF NOT EXISTS test_questions (
        id TEXT PRIMARY KEY,
        question TEXT,
        answer TEXT,
        test TEXT
    )
''')
conn.commit()
conn.close()


@app.route('/upload/', methods=['GET', 'POST'])
def upload_document():
    """
    Handle the upload and processing of documents.

    :return: If the request method is GET, renders the upload form template.
             If the request method is POST and the file is successfully uploaded and processed,
             returns a JSON response with a success message.
             If the file type is unsupported or an error occurs during processing,
             returns a JSON response with an error message and appropriate status code.
    """
    upload_form = UploadFileForm()
    if upload_form.validate_on_submit():
        uploaded_file = upload_form.file.data  # Grab the file
        # Save the file in the UPLOAD_FOLDER
        filename = secure_filename(uploaded_file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        uploaded_file.save(file_path)  # Save the file to disk

        # Process the file according to its type (PDF or DOCX)
        try:
            if filename.endswith('.pdf'):
                extracted_text_file = process_pdf(uploaded_file)
            elif filename.endswith('.docx'):
                extracted_text_file = process_docx(uploaded_file)
            else:
                return jsonify({'error': 'Unsupported file type'}), 400
            session['extracted_text_file_name'] = extracted_text_file  # Store the file name in the session

            return jsonify({'message': f'Document processed and saved as {extracted_text_file}'}), 200

        except Exception as e:
            return jsonify({'error': f'Error processing document: {str(e)}'}), 500

    return render_template('upload.html', form=upload_form)


@app.route('/query/', methods=['GET', 'POST'])
def query():
    """
    Handles HTTP GET and POST requests to the /query/ endpoint.
    On a GET request, renders the query form.
    On a POST request, validates the form, reads the extracted text from a file,
    generates an answer using a question-answering model, and then creates bullet points
    and a test question based on the answer.
    The results are stored in the database and returned as a JSON response.

    :return: JSON response with the answer, bullet points, test question, and test question ID or an error message
    and status code if something goes wrong.
    """
    query_form = QueryForm()
    extracted_text_file_name = session.get('extracted_text_file_name')  # Retrieve the file name from the session
    if extracted_text_file_name is None:
        return jsonify({'error': 'No file uploaded or processed'}), 400  # Handle missing file

    if query_form.validate_on_submit():
        question = query_form.question.data  # Grab the submitted question from the form
        # Specify the path to the extracted text file
        extracted_text_file = os.path.join(app.config['UPLOAD_FOLDER'], extracted_text_file_name)

        try:
            # Open and read the extracted text file
            with open(extracted_text_file, 'r') as file:
                context = file.read()
        except Exception as e:
            return jsonify({'error': f'Error reading extracted text: {str(e)}'}), 500

        # Use the extracted text as context for the question-answering model
        result = question_answering_pipeline({'question': question, 'context': context})
        # print(f"Answer: '{result['answer']}'")
        answer = result['answer']

        # Generate bullet points
        bullet_points = generate_bullet_points(answer)
        # Generate test question
        test_question, test_answer = generate_test_question_and_answer(answer)

        # Save test question and answer
        test_question_id = str(uuid.uuid4())
        query_conn = sqlite3.connect('qat_database.db')
        query_cursor = query_conn.cursor()
        query_cursor.execute('''
            INSERT INTO test_questions (id, question, answer)
            VALUES (?, ?, ?)
        ''', (test_question_id, test_question, test_answer))
        conn.commit()
        conn.close()

        session['test_question'] = test_question  # Store the test question in the session
        session['test_question_id'] = test_question_id  # Store the test question id in the session

        return jsonify({
            'answer': answer,
            'bullet_points': bullet_points,
            'test_question': test_question,
            'test_question_id': test_question_id
        })

    return render_template('query.html', form=query_form)


@app.route('/evaluate/', methods=['GET', 'POST'])
def evaluate():
    """
    Handles the evaluation of a user's answer to a test question.

    A form is used to capture the user's answer, which is then validated
    and compared to the correct answer stored in the database.
    If the submitted answer is correct, relevant metrics are calculated and returned as a JSON response.
    If the form submission is invalid or if the provided question ID is not found, appropriate error messages are returned.

    :return:
        On successful evaluation - JSON response containing `knowledge_understood` and `knowledge_confidence`.
        On invalid form submission or missing question ID - Relevant error messages in JSON format.
    """
    eval_form = EvaluateForm()
    # Retrieve the test question and its id from the session
    test_question = session.get('test_question')
    test_question_id = session.get('test_question_id')

    if eval_form.validate_on_submit():
        user_answer = eval_form.user_answer.data  # Grab the submitted answer from the form

        # Retrieve the correct answer from the database
        eval_conn = sqlite3.connect('qat_database.db')
        eval_cursor = eval_conn.cursor()
        eval_cursor.execute('''
            SELECT answer FROM test_questions WHERE id = ?
        ''', (test_question_id,))
        result = eval_cursor.fetchone()
        eval_cursor.close()

        if result is None:
            return jsonify({'error': 'Invalid test_question_id'}), 400

        correct_answer = result[0]

        # Evaluate the answer
        knowledge_understood, knowledge_confidence = evaluate_answer(user_answer, correct_answer)

        return jsonify({
            'knowledge_understood': knowledge_understood,
            'knowledge_confidence': knowledge_confidence
        })

    return render_template('evaluate.html', question=test_question, form=eval_form)


@app.route('/test_questions/', methods=['GET'])
def get_test_questions():
    """
    :return: A JSON response containing all test questions
    retrieved from the 'test_questions' table in the 'qat_database.db' SQLite database.
    """
    tq_conn = sqlite3.connect('qat_database.db')
    tq_cursor = tq_conn.cursor()
    tq_cursor.execute('SELECT * FROM test_questions')
    rows = tq_cursor.fetchall()
    test_questions = []
    for row in rows:
        test_questions.append({
            'id': row[0],
            'question': row[1],
            'answer': row[2]
        })
    tq_cursor.close()
    return jsonify(test_questions)


@app.route('/test_questions/<test_question_id>', methods=['DELETE'])
def delete_test_question(test_question_id):
    """
    :param test_question_id: The ID of the test question to be deleted.
    :return: A JSON response indicating the result of the deletion operation.
    Possible responses include a success message with a 200 status code if the deletion is successful,
    an error message with a 404 status code if the test question is not found,
    and an error message with a 500 status code if there is an internal server error.
    """
    try:
        tq_conn = sqlite3.connect('qat_database.db')
        tq_cursor = tq_conn.cursor()
        tq_cursor.execute('DELETE FROM test_questions WHERE id=?', (test_question_id,))
        tq_conn.commit()
        if tq_cursor.rowcount == 0:
            return jsonify({'error': 'Test question not found'}), 404
        tq_cursor.close()
        return jsonify({'message': 'Test question deleted successfully'}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500


def process_pdf(file):
    """
    :param file: File object representing the PDF to be processed.
    :return: The filename where the extracted text has been saved.
    """
    # Read the PDF directly from the file object
    pdf_reader = PyPDF2.PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()

    # Save the extracted text to a file for further use
    extracted_text_filename = f'extracted_text_{os.path.splitext(secure_filename(file.filename))[0]}.txt'
    with open(os.path.join(app.config['UPLOAD_FOLDER'], extracted_text_filename), 'w') as text_file:
        text_file.write(text)

    print(f"Extracted text from PDF: {text}")
    return extracted_text_filename  # Return the filename for further processing


def process_docx(file):
    """
    :param file: The DOCX file object to be processed.
    :return: The filename where the extracted text is saved.
    """
    # Read the DOCX file from the file object
    doc = docx.Document(file)
    text = ""
    for paragraph in doc.paragraphs:
        text += paragraph.text

    # Save the extracted text to a file for further use
    extracted_text_filename = f'extracted_text_{os.path.splitext(secure_filename(file.filename))[0]}.txt'
    with open(os.path.join(app.config['UPLOAD_FOLDER'], extracted_text_filename), 'w') as text_file:
        text_file.write(text)

    print(f"Extracted text from DOCX: {text}")
    return extracted_text_filename  # Return the filename for further processing


def generate_bullet_points(answer):
    """
    :param answer: The text input that needs to be summarized and converted into bullet points.
    :return: A list of bullet points extracted from the summarized text.
    """
    # Summarize the answer to extract key information
    summary_text = summarization_pipeline(answer, max_length=150, min_length=50, do_sample=False)[0]['summary_text']

    # Split the summary into sentences
    sentences = nltk.sent_tokenize(summary_text)

    # Extract key phrases from each sentence
    bullet_points = []
    for sentence in sentences:
        # Perform part-of-speech tagging
        words = nltk.word_tokenize(sentence)
        tagged_words = nltk.pos_tag(words)

        # Extract noun phrases (e.g., "key concepts", "important factors")
        for i in range(len(tagged_words) - 1):
            if tagged_words[i][1].startswith('NN') and tagged_words[i + 1][1].startswith('NN'):
                bullet_points.append(f"{tagged_words[i][0]} {tagged_words[i + 1][0]}")

        # Extract verbs with their objects (e.g., "explain the process", "identify the issues")
        for i in range(len(tagged_words) - 2):
            if (tagged_words[i][1].startswith('VB') and tagged_words[i + 1][1].startswith('DT')
                    and tagged_words[i + 2][1].startswith('NN')):
                bullet_points.append(f"{tagged_words[i][0]} {tagged_words[i + 1][0]} {tagged_words[i + 2][0]}")

    # Remove duplicates and return the bullet points
    return list(set(bullet_points))


def generate_test_question_and_answer(answer):
    """
    :param answer: A string containing the text from which questions should be generated.
    :return: A tuple containing a generated question and the corresponding answer.
    """
    questions = []

    # Ask the model to generate questions directly
    result = question_answering_pipeline({
        'question': "Generate a question about this text.",
        'context': answer
    })
    questions.append(result['answer'])

    # Use summarization to identify key sentences and turn them into questions
    summary_text = summarization_pipeline(answer, max_length=150, min_length=50, do_sample=False)[0]['summary_text']
    sentences = summary_text.split('. ')
    for sentence in sentences:
        # Use a simple heuristic to turn declarative sentences into questions
        if sentence.startswith("This"):
            question = "What does this refer to?"
            questions.append(question)
        elif " is " in sentence:
            parts = sentence.split(" is ")
            question = f"What is {parts[0]}?"
            questions.append(question)

    # Select a test question randomly
    test_question = random.choice(questions)
    # Generate the test answer
    test_answer = answer
    return test_question, test_answer


def evaluate_answer(user_answer, correct_answer):
    """
    :param user_answer: The answer provided by the user as a string.
    :param correct_answer: The correct answer to compare against the user's answer.
    :return: A tuple containing a boolean indicating if the answer is understood (based on a similarity threshold
    and sentiment match) and an integer representing the confidence percentage.
    """
    # Compare user answer and correct answer using string similarity
    similarity_ratio = SequenceMatcher(None, user_answer, correct_answer).ratio()

    # Analyze the sentiment of both answers
    user_answer_blob = TextBlob(user_answer)
    correct_answer_blob = TextBlob(correct_answer)

    # Compare the sentiment polarity
    sentiment_match = user_answer_blob.sentiment.polarity == correct_answer_blob.sentiment.polarity

    # Determine if the answer is correct based on similarity threshold and sentiment match
    knowledge_understood = similarity_ratio > 0.7 and sentiment_match
    knowledge_confidence = int(similarity_ratio * 100)

    return knowledge_understood, knowledge_confidence


if __name__ == '__main__':
    app.run(debug=True)
