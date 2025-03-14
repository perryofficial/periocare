from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, session
import pandas as pd
import pickle
import warnings
import mysql.connector
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch
import openai
from flask_mail import Mail, Message
import numpy as np
from flask_bcrypt import Bcrypt
import requests
from flask_login import LoginManager, login_user, login_required, current_user, logout_user, login_manager
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime, timedelta
from flask_socketio import SocketIO, send
import eventlet


warnings.filterwarnings("ignore")

# Load the pre-trained model from the pickle file
with open('predictionmodel.pkl', 'rb') as file:
    model = pickle.load(file)

app = Flask(__name__)
bcrypt = Bcrypt(app)
socketio = SocketIO(app, cors_allowed_origins="*")
app.secret_key = 'supersecretkey'  # For session handling and flash messages

def get_db_connection():
    try:
        connection = mysql.connector.connect(
            host="localhost",
            user="root",
            password="Prajwal@12345",  # Verify this password!
            database="redmoonusers"      # Ensure this database exists
        )
        print("✅ Database connection successful!")
        return connection
    except mysql.connector.Error as err:
        print(f"❌ Database connection failed: {err}")
        return None

# Database configuration
db = mysql.connector.connect(
    host="localhost",
    user="root",  # Replace with your MySQL username
    password="Prajwal@12345",  # Replace with your MySQL password
    database="redmoonusers"  # Replace with your MySQL database name
)
cursor = db.cursor()

# Mail configuration
app.config['MAIL_SERVER'] = 'smtp.gmail.com'
app.config['MAIL_PORT'] = 465
app.config['MAIL_USERNAME'] = 'naikprajwal20@gmail.com'
app.config['MAIL_PASSWORD'] = 'vkjg yicm gmca wosw'
app.config['MAIL_USE_TLS'] = False
app.config['MAIL_USE_SSL'] = True
mail = Mail(app)

chatbot = pipeline('text-generation', model='gpt2')

# openai.api_key = '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'

FAST2SMS_API_KEY = 'K2HMqk7Qy0dVSOwLB3Fosfx8erzRUpahCgNEl1mnbDWAvX5uG6UMdsfQ1lB3cyN4xRY9erzpKFPVkAIw'
FAST2SMS_API_URL = "https://www.fast2sms.com/dev/api"

# Function to send SMS using Fast2SMS
def send_sms(phone_number, message_body):
    headers = {
        'authorization': FAST2SMS_API_KEY,
        'Content-Type': 'application/json'
    }
    payload = {
        "sender_id": "FSTSMS",
        "message": message_body,
        "language": "english",
        "route": "p",
        "numbers": phone_number
    }
    response = requests.post(FAST2SMS_API_URL, headers=headers, json=payload)
    return response

# Function to send email using Flask-Mail
def send_email(to_email, subject, message_body):
    msg = Message(subject, sender='naikprajwal20@gmail.com', recipients=[to_email])
    msg.body = message_body
    mail.send(msg)

@app.route('/')
def home():
    if "un" not in session:
        return redirect(url_for("login"))  # Redirect to login if not authenticated
    return render_template('home.html')  # Render the home page if logged in


# @app.route("/signup", methods=["GET", "POST"])
# def signup():
#     if "un" in session:
#         return redirect(url_for("home"))

#     if request.method == "POST":
#         username = request.form["username"]
#         password = request.form["password"]
#         phone_number = request.form["phone_number"]
#         email = request.form["email"]

#         # Hash the password
#         hashed_pw = bcrypt.generate_password_hash(password).decode('utf-8')

#         # Insert user into the database
#         try:
#             cursor.execute(
#                 "INSERT INTO users (username, email, phone_number, password) VALUES (%s, %s, %s, %s)",
#                 (username, email, phone_number, hashed_pw)
#             )
#             db.commit()

#             # Send SMS and email confirmations
#             send_sms(phone_number, f"Hello {username}, your account has been successfully created!")
#             send_email(email, "RedMoon Account Created Successfully", f"Hello {username}, your account has been successfully created! You can now login.")

#             return redirect(url_for("login"))
#         except mysql.connector.Error as err:
#             flash(f"Error: {err}", "error")
#             db.session.rollback()

#             return render_template("signup.html")
#     return render_template("signup.html")
@app.route("/signup", methods=["GET", "POST"])
def signup():
    if "un" in session:
        return redirect(url_for("home"))

    if request.method == "POST":
        username = request.form.get("username")
        password = request.form.get("password")
        phone_number = request.form.get("phone_number")
        email = request.form.get("email")

        print(f"📥 Form Data: {username}, {email}, {phone_number}")  # Debug log

        if not all([username, password, phone_number, email]):
            flash("All fields are required.", "error")
            return render_template("signup.html")

        hashed_pw = bcrypt.generate_password_hash(password).decode('utf-8')

        connection = get_db_connection()
        if not connection:
            flash("Database connection failed.", "error")
            return render_template("signup.html")

        cursor = connection.cursor()
        try:
            # Check for existing user
            cursor.execute("SELECT username, email FROM users WHERE username = %s OR email = %s", (username, email))
            existing = cursor.fetchone()
            if existing:
                print(f"🚫 Duplicate entry: {existing}")  # Debug log
                flash("Username or email already exists.", "error")
                return render_template("signup.html")

            # Insert new user
            insert_query = """
                INSERT INTO users (username, email, phone_number, password)
                VALUES (%s, %s, %s, %s)
            """
            cursor.execute(insert_query, (username, email, phone_number, hashed_pw))
            connection.commit()
            print(f"✅ Inserted user: {username}")  # Debug log

            # Get the last inserted ID
            cursor.execute("SELECT LAST_INSERT_ID()")
            user_id = cursor.fetchone()[0]
            print(f"🆔 New user ID: {user_id}")  # Debug log

            flash("Signup successful! Please login.", "success")
            return redirect(url_for("login"))

        except mysql.connector.Error as err:
            print(f"⚠️ DATABASE ERROR: {err}")  # Debug log
            connection.rollback()
            flash("Failed to create account. Please try again.", "error")
        finally:
            cursor.close()
            connection.close()

    return render_template("signup.html")




@app.route("/login", methods=["GET", "POST"])
def login():
    if "un" in session:
        return redirect(url_for("home"))

    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]

        connection = get_db_connection()  # Get a fresh connection
        if connection is None:
            flash("Database connection failed.", "error")
            return render_template("login.html")

        cursor = connection.cursor()  # Create a new cursor
        try:
            cursor.execute("SELECT id, password FROM users WHERE username = %s", (username,))
            result = cursor.fetchone()

            if result and bcrypt.check_password_hash(result[1], password):
                session["un"] = username
                session["user_id"] = result[0]
                return redirect(url_for("home"))
            else:
                flash("Invalid username or password", "error")
        except mysql.connector.Error as err:
            flash(f"Database error: {err}", "error")
        finally:
            cursor.close()  # Close the cursor after execution
            connection.close()  # Close the connection

    return render_template("login.html")




@app.route("/forgot_password", methods=["GET", "POST"])
def forgot_password():
    if request.method == "POST":
        username = request.form["username"]

        cursor.execute("SELECT phone_number, email FROM users WHERE username = %s", (username,))
        result = cursor.fetchone()

        if result:
            phone_number, email = result
            message_body = f"Hello {username}, use the following link to reset your password: /reset-password"
            send_sms(phone_number, message_body)
            send_email(email, "Password Reset Instructions", message_body)
            return redirect(url_for("login"))
        else:
            msg = "Username not found"
            return render_template("forgot_password.html", msg=msg)

    return render_template("forgot_password.html")

@app.route("/reset_password", methods=["GET", "POST"])
def reset_password():
    if request.method == "POST":
        username = request.form["username"]
        new_password = request.form["new_password"]

        # Query the database for the user
        cursor.execute("SELECT id FROM users WHERE username = %s", (username,))
        result = cursor.fetchone()

        if result:
            hashed_pw = bcrypt.generate_password_hash(new_password).decode('utf-8')
            cursor.execute("UPDATE users SET password = %s WHERE username = %s", (hashed_pw, username))
            db.commit()

            send_sms(result[0], f"Hello {username}, your password has been successfully reset!")
            send_email(username, "Password Reset Successfully", f"Hello {username}, your password has been successfully reset!")
            return redirect(url_for("login"))
        else:
            msg = "Username not found"
            return render_template("reset_password.html", msg=msg)

    return render_template("reset_password.html")

@app.route('/chatbot', methods=['GET', 'POST'])
def chatbot():
    if request.method == 'GET':
        return render_template('chatbot.html')

    elif request.method == 'POST':
        if not request.is_json:
            return jsonify({"response": "Invalid request. Please send JSON data."}), 415

        user_message = request.json.get('message', '').strip()
        if not user_message:
            return jsonify({"response": "Please ask a question."})

        # Store conversation history
        if 'conversation_history' not in session:
            session['conversation_history'] = []

        # Add a context to the conversation history
        # This helps to guide the model on the topic of menstrual health
        session['conversation_history'].append(f"User: {user_message}")
        conversation_history = "\n".join(session['conversation_history'])

        # Updated: Provide context and guide the model to generate relevant responses
        prompt = f"User: What is menstrual health?\nBot: Menstrual health is the state of a woman's reproductive system and refers to a healthy menstrual cycle, symptoms management, and well-being.\n{conversation_history}"

        # Get the bot's response
        bot_response = chatbot(prompt, max_length=150)[0]['generated_text']

        # Save the bot's response to the conversation history
        session['conversation_history'].append(f"Bot: {bot_response}")

        return jsonify({"response": bot_response})

# @app.route('/chatbot', methods=['GET', 'POST'])
# def chatbot():
#     if request.method == 'GET':
#         return render_template('chatbot.html')

#     elif request.method == 'POST':
#         if not request.is_json:
#             return jsonify({"response": "Invalid request. Please send JSON data."}), 415

#         user_message = request.json.get('message', '').strip()
#         if not user_message:
#             return jsonify({"response": "Please ask a question."})

#         if 'conversation_history' not in session:
#             session['conversation_history'] = []

#         session['conversation_history'].append(f"User: {user_message}")
#         conversation_history = "\n".join(session['conversation_history'])

        try:
            response = openai.Completion.create(
            model="gpt-3.5-turbo",  # Use the new model
            prompt=conversation_history,
            max_tokens=150,
            temperature=0.7,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0.6
)
            bot_response = response.choices[0].text.strip()
            session['conversation_history'].append(f"Bot: {bot_response}")
            return jsonify({"response": bot_response})
        except Exception as e:
            return jsonify({"response": f"Error generating response: {str(e)}"}), 500

# Function to predict the next period date
def predict_next_period_date(model, cycle_length, period_duration, symptoms_encoded, stress_level_encoded, exercise_level_encoded):
    try:
        input_data = [[cycle_length, period_duration, symptoms_encoded, stress_level_encoded, exercise_level_encoded]]
        predicted_days = model.predict(input_data)[0]
        return predicted_days
    except Exception as e:
        print(f"Model Prediction Error: {str(e)}")
        return None
    
#previousslyy it was

# app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql+pymysql://root:Prajwal%40021892@localhost/redmoonusers'
# app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# db = SQLAlchemy(app)
#-------------------------------------------------------------------------------------------------------
    
app.config['SQLALCHEMY_DATABASE_URI'] =  'mysql+mysqlconnector://root:Prajwal%4012345@localhost/redmoonusers'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Initialize SQLAlchemy with the Flask app
db = SQLAlchemy(app)



class Cycle(db.Model):
    __tablename__ = 'cycles'
    
    id = db.Column(db.Integer, primary_key=True)
    start_date = db.Column(db.Date, nullable=False)
    cycle_length = db.Column(db.Integer, nullable=False)

@app.route('/add_cycle', methods=['POST'])
def add_cycle():
    try:
        data = request.get_json()
        start_date = datetime.strptime(data['start_date'], "%Y-%m-%d").date()
        cycle_length = int(data['cycle_length'])

        new_cycle = Cycle(start_date=start_date, cycle_length=cycle_length)
        db.session.add(new_cycle)
        db.session.commit()

        return jsonify({"message": "Cycle added successfully!"})

    except Exception as e:
        return jsonify({"error": str(e)}), 400  # Return error if something goes wrong


@app.route('/get_predictions', methods=['GET','POST'])
def get_predictions():
    cycles = Cycle.query.order_by(Cycle.start_date.desc()).limit(6).all()
    
    if not cycles:
        return jsonify({"error": "No cycle records found."})

    last_cycle = cycles[0]
    predictions = []

    for i in range(1, 4):  # Predict for the next 3 months
        next_period_start = last_cycle.start_date + timedelta(days=last_cycle.cycle_length * i)
        predictions.append(next_period_start.strftime("%Y-%m-%d"))

    return jsonify({"predicted_dates": predictions})


@app.route('/period-resources')
def period_resources():
    resources = [
        {
            'type': 'video',
            'title': 'Girls Talk: What Are Periods and Why Do Girls Get Them?',
            'url': 'https://youtu.be/oqiNpJqsGSQ?si=QHEM9hFM8f1FnxMD'
        },
        {
            'type': 'blog',
            'title': 'Which is Better: Menstrual Cups, Tampons, or Pads?',
            'url': 'https://allmatters.com/en-eu/blogs/blog/which-is-better-menstrual-cups-tampons-and-pads#:~:text=Both%20menstrual%20cups%20and%20pads,a%20more%20environmentally%2Dfriendly%20choice.'
        }
    ]
    return render_template('period_resources.html', resources=resources)

@app.route('/submit_blog', methods=['GET', 'POST'])
def submit_blog():
    if request.method == 'POST':
        title = request.form['title']
        content = request.form['content']

        if not title or not content:
            flash("Please fill out both title and content.", "error")
            return render_template('submit_blog.html')

        try:
            conn = get_db_connection()
            cur = conn.cursor()
            cur.execute("SELECT id FROM users WHERE username = %s", (session["un"],))
            user_result = cur.fetchone()
            if not user_result:
                flash("User not found. Please log in again.", "error")
                conn.close()
                return redirect(url_for('login'))

            user_id = user_result[0]

            # Insert the blog into the database
            cur.execute(
                "INSERT INTO blogs (title, content, user_id) VALUES (%s, %s, %s)",
                (title, content, user_id)
            )
            conn.commit()
            flash("Your blog has been successfully submitted!", "success")
            conn.close()
            return redirect(url_for('submit_blog'))

        except mysql.connector.Error as err:
            flash(f"Database Error: {err}", "error")
            conn.rollback()
            conn.close()
            return render_template('submit_blog.html')

    # For GET request: fetch all blogs with their likes and comments
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("""
        SELECT blogs.id, blogs.title, blogs.content, users.username, blogs.created_at
        FROM blogs
        JOIN users ON blogs.user_id = users.id
        ORDER BY blogs.created_at DESC
    """)
    blogs = cur.fetchall()

    blog_data = []
    for blog in blogs:
        blog_id = blog[0]

        # Count likes for each blog
        cur.execute("SELECT COUNT(*) FROM blog_likes WHERE blog_id = %s", (blog_id,))
        likes = cur.fetchone()[0]

        # Fetch comments and corresponding usernames
        cur.execute("""
            SELECT comments.content, users.username 
            FROM comments 
            JOIN users ON comments.user_id = users.id 
            WHERE blog_id = %s
        """, (blog_id,))
        comments = cur.fetchall()

        blog_data.append({
            'id': blog_id,
            'title': blog[1],
            'content': blog[2],
            'author': blog[3],
            'created_at': blog[4],
            'likes': likes,
            'comments': [{'user': comment[1], 'content': comment[0]} for comment in comments]
        })
    conn.close()
    return render_template('submit_blog.html', blogs=blog_data)

@app.route('/like_blog/<int:blog_id>', methods=['POST'])
def like_blog(blog_id):
    if 'user_id' not in session:
        return jsonify({'success': False, 'message': 'Please login to like the blog.'}), 403

    try:
        # Get the database connection
        db = get_db_connection()
        cursor = db.cursor()

        # Insert like into the blog_likes table
        cursor.execute("INSERT INTO blog_likes (blog_id, user_id) VALUES (%s, %s)", (blog_id, session["user_id"]))
        db.commit()

        # Fetch the updated like count from blog_likes table
        cursor.execute("SELECT COUNT(*) FROM blog_likes WHERE blog_id = %s", (blog_id,))
        likes = cursor.fetchone()[0]

        cursor.close()
        db.close()

        return jsonify({'success': True, 'likes': likes})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

@app.route('/comment_blog/<int:blog_id>', methods=['POST'])
def comment_blog(blog_id):
    user_id = session.get("user_id")  # Use logged-in user's ID
    if not user_id:
        return jsonify({'success': False, 'message': 'Please log in to comment.'}), 400

    data = request.get_json()
    comment = data.get('comment', '').strip()

    if not comment:
        return jsonify({'success': False, 'message': 'Comment cannot be empty.'}), 400

    try:
        # Get the database connection
        db = get_db_connection()
        cursor = db.cursor()

        # Insert the comment with user_id
        cursor.execute("INSERT INTO comments (blog_id, user_id, content) VALUES (%s, %s, %s)", (blog_id, user_id, comment))
        db.commit()

        cursor.close()
        db.close()

        return jsonify({'success': True, 'message': 'Comment added successfully!'}), 200

    except mysql.connector.Error as err:
        return jsonify({'success': False, 'message': f'Database Error: {err}'}), 500

@app.route('/community')
def community():
    if "un" not in session:
        return redirect(url_for("login"))  # Redirect to login if not authenticated
    return render_template('community.html')  # Render the chat page


# WebSocket event for handling messages
@socketio.on('message')
def handle_message(msg):
    print(f"Message received: {msg}")
    send(msg, broadcast=True)  # Broadcast the message to all users

@socketio.on('connect')
def handle_connect():
    print("✅ Client connected!")

@socketio.on('message')
def handle_message(msg):
    print(f"📩 Message received: {msg}")
    send(msg, broadcast=True)

    
@app.route("/logout", methods=["POST"])
def logout():
    session.pop("un", None)
    return redirect(url_for("login"))


@socketio.on('message')
def handle_message(msg):
    print(f"📩 Message received: {msg}")

    # If user requests a doctor, respond with an automated message
    if "I need help from" in msg:
        doctor_name = msg.split("I need help from ")[1]
        response = f"{doctor_name} will assist you shortly."
        send(response, broadcast=True)
    else:
        send(msg, broadcast=True)  # Broadcast message to all users



# if __name__ == '__main__':
#     app.secret_key = "your_secret_key"
#     app.run(debug=True)
#edited 10-3-25
# if __name__ == '__main__':
#     socketio.run(app, debug=True)

if __name__ == '__main__':
    # Test the database connection at startup
    test_conn = get_db_connection()
    if test_conn:
        test_conn.close()
        print("✅ Connection test passed!")
    else:
        print("❌ Connection test failed. Check MySQL settings.")
    
    # Start the Flask app
    socketio.run(app, debug=True)