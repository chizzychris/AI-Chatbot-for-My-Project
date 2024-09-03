from flask import Flask, render_template, request, redirect, url_for
import sqlite3

app = Flask(__name__)

# Connect to SQLite database (or create it if it doesn't exist)
def init_sqlite_db():
    conn = sqlite3.connect('students.db')
    print("Opened database successfully")
    conn.execute('CREATE TABLE IF NOT EXISTS students (id INTEGER PRIMARY KEY AUTOINCREMENT, matric_number TEXT, name TEXT, email TEXT)')
    print("Table created successfully")
    conn.close()

init_sqlite_db()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/submit-details', methods=['POST'])
def submit_details():
    if request.method == 'POST':
        try:
            matric_number = request.form['matricNumber']
            name = request.form['name']
            email = request.form['email']
            
            with sqlite3.connect('students.db') as conn:
                cur = conn.cursor()
                cur.execute("INSERT INTO students (matric_number, name, email) VALUES (?, ?, ?)", (matric_number, name, email))
                conn.commit()
                msg = "Record successfully added."
        except Exception as e:
            conn.rollback()
            msg = "Error occurred in insertion: " + str(e)
        finally:
            return redirect(url_for('complaint_submission'))
            conn.close()

@app.route('/complaint-submission')
def complaint_submission():
    return "Redirecting to the complaint submission page..."

@app.route('/chatbot')
def chatbot():
    return "Redirecting to the chatbot page..."

if __name__ == '__main__':
    app.run(debug=True)
