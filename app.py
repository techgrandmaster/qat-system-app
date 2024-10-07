import random
import sqlite3
import uuid
from difflib import SequenceMatcher

import PyPDF2
import docx
import nltk
from flask import Flask, request, jsonify
from flask_uploads import UploadSet, configure_uploads, DOCUMENTS
from textblob import TextBlob
from transformers import pipeline

app = Flask(__name__)

nltk.download('punkt')  # Download the Punkt sentence tokenizer
nltk.download('averaged_perceptron_tagger')  # Download the part-of-speech tagger

question_answering_pipeline = pipeline("question-answering")
summarization_pipeline = pipeline("summarization") # Use a summarization model

# Configure file uploads
documents = UploadSet('documents', DOCUMENTS)
app.config['UPLOADED_DOCUMENTS_DEST'] = 'uploads'
configure_uploads(app, documents)

# Initialize the question answering pipeline
qat_pipeline = pipeline("question-answering")

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


@app.route('/')
def index():
    return "Welcome to the QAT system!"


@app.route('/upload/', methods=['POST'])
def upload_document():
    if 'document' not in request.files:
        return jsonify({'error': 'No document part'}), 400

    file = request.files['document']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file:
        filename = documents.save(file)
        # Further processing of the uploaded document
        try:
            if filename.endswith('.pdf'):
                process_pdf(filename)
            elif filename.endswith('.docx'):
                process_docx(filename)
            # Add more file types as needed
        except Exception as e:
            return jsonify({'error': f'Error processing document: {str(e)}'}), 500

        return jsonify({'message': f'Document uploaded as {filename}'}), 200


@app.route('/query/', methods=['POST'])
def query():
    data = request.get_json()
    question = data.get('question')
    document_path = data.get('document_path')  # Path to the uploaded document

    if not question or not document_path:
        return jsonify({'error': 'Missing question or document_path'}), 400

    # Answer the question
    with open(document_path, 'r') as file:
        context = file.read()
    result = qat_pipeline({'question': question, 'context': context})
    answer = result['answer']

    # Generate bullet points
    bullet_points = generate_bullet_points(answer)

    # Generate test question
    test_question, test_answer = generate_test_question_and_answer(answer)

    # Save test question and answer
    test_question_id = str(uuid.uuid4())
    cursor.execute('''
        INSERT INTO test_questions (id, question, answer)
        VALUES (?, ?, ?)
    ''', (test_question_id, test_question, test_answer))
    conn.commit()

    return jsonify({
        'answer': answer,
        'bullet_points': bullet_points,
        'test_question': test_question,
        'test_question_id': test_question_id
    })


@app.route('/evaluate/', methods=['POST'])
def evaluate():
    data = request.get_json()
    test_question_id = data.get('test_question_id')
    user_answer = data.get('user_answer')

    if not test_question_id or not user_answer:
        return jsonify({'error': 'Missing test_question_id or user_answer'}), 400

    # Retrieve the correct answer from the database
    cursor.execute('''
        SELECT answer FROM test_questions WHERE id = ?
    ''', (test_question_id,))
    result = cursor.fetchone()

    if result is None:
        return jsonify({'error': 'Invalid test_question_id'}), 400

    correct_answer = result[0]

    # Evaluate the answer
    knowledge_understood, knowledge_confidence = evaluate_answer(user_answer, correct_answer)

    return jsonify({
        'knowledge_understood': knowledge_understood,
        'knowledge_confidence': knowledge_confidence
    })


@app.route('/test_questions/', methods=['GET'])
def get_test_questions():
    cursor.execute('SELECT * FROM test_questions')
    rows = cursor.fetchall()
    test_questions = []
    for row in rows:
        test_questions.append({
            'id': row[0],
            'question': row[1],
            'answer': row[2]
        })
    return jsonify(test_questions)


@app.route('/test_questions/<test_question_id>', methods=['DELETE'])
def delete_test_question(test_question_id):
    try:
        cursor.execute('DELETE FROM test_questions WHERE id=?', (test_question_id,))
        conn.commit()
        if cursor.rowcount == 0:
            return jsonify({'error': 'Test question not found'}), 404
        return jsonify({'message': 'Test question deleted successfully'}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500


def process_pdf(filename):
    with open(f'uploads/{filename}', 'rb') as pdf_file:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        # Save the extracted text to a file
        with open(f'outputs/extracted_text_{filename}.txt', 'w') as text_file:
            text_file.write(text)
        print(f"Extracted text from PDF: {text}")


def process_docx(filename):
    doc = docx.Document(f'uploads/{filename}')
    text = ""
    for paragraph in doc.paragraphs:
        text += paragraph.text
    # Save the extracted text to a file
    with open(f'outputs/extracted_text_{filename}.txt', 'w') as text_file:
        text_file.write(text)
    print(f"Extracted text from DOCX: {text}")


def generate_bullet_points(answer):
    # Summarize the answer to extract key information
    summary_text = summarization_pipeline(answer, max_length=100, min_length=30, do_sample=False)[0]['summary_text']

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
            if tagged_words[i][1].startswith('VB') and tagged_words[i + 1][1].startswith('DT') and tagged_words[i + 2][1].startswith('NN'):
                bullet_points.append(f"{tagged_words[i][0]} {tagged_words[i + 1][0]} {tagged_words[i + 2][0]}")

    # Remove duplicates and return the bullet points
    return list(set(bullet_points))


def generate_test_question_and_answer(answer):
    # Generate potential test questions
    questions = []

    # Ask the model to generate questions directly
    result = question_answering_pipeline({
        'question': "Generate a question about this text.",
        'context': answer
    })
    questions.append(result['answer'])

    # Use summarization to identify key sentences and turn them into questions
    summary_text = summarization_pipeline(answer, max_length=100, min_length=30, do_sample=False)[0]['summary_text']
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

    # Generate the test answer (use the original answer or a relevant part)
    test_answer = answer

    return test_question, test_answer


# Logic to compare user_answer and correct_answer using string similarity and sentiment analysis
def evaluate_answer(user_answer, correct_answer):

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


def update_test_question(test_question_id):
    data = request.get_json()
    new_question = data.get('question')
    new_answer = data.get('answer')

    if not new_question or not new_answer:
        return jsonify({'error': 'Missing question or answer'}), 400

    try:
        cursor.execute('''
            UPDATE test_questions
            SET question = ?, answer = ?
            WHERE id = ?
        ''', (new_question, new_answer, test_question_id))
        conn.commit()

        if cursor.rowcount == 0:
            return jsonify({'error': 'Test question not found'}), 404

        return jsonify({'message': 'Test question updated successfully'}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)
