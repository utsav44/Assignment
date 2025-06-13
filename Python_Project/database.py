from flask import Flask, request, jsonify
import sqlite3
import uuid
import re
from datetime import datetime
from functools import wraps

app = Flask(__name__)

# Database setup
DATABASE = 'complaints.db'


def init_db():
    """Initialize the database with complaints table"""
    with sqlite3.connect(DATABASE) as conn:
        conn.execute('''
            CREATE TABLE IF NOT EXISTS complaints (
                complaint_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                phone_number TEXT NOT NULL,
                email TEXT NOT NULL,
                complaint_details TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        conn.commit()
    print("✅ Database initialized successfully")


def get_db_connection():
    conn = sqlite3.connect(DATABASE)
    conn.row_factory = sqlite3.Row
    return conn



# Input validation decorator
def validate_complaint_input(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        data = request.get_json()

        if not data:
            return jsonify({
                'error': 'Invalid JSON',
                'message': 'Request body must contain valid JSON'
            }), 400

        # Check required fields
        required_fields = ['name', 'phone_number', 'email', 'complaint_details']
        errors = []

        for field in required_fields:
            if field not in data or not data[field] or not str(data[field]).strip():
                errors.append(f'{field.replace("_", " ").title()} is required')

        # If basic validation fails, return early
        if errors:
            return jsonify({
                'error': 'Validation failed',
                'details': errors
            }), 400

        return f(*args, **kwargs)

    return decorated_function


# API Endpoints

@app.route('/complaints', methods=['POST'])
@validate_complaint_input
def create_complaint():
    try:
        data = request.get_json()

        # Generate unique complaint ID
        complaint_id = str(uuid.uuid4())

        # Extract and clean data
        name = str(data['name']).strip()
        phone_number = str(data['phone_number']).strip()
        email = str(data['email']).strip()
        complaint_details = str(data['complaint_details']).strip()

        # Insert into database
        with get_db_connection() as conn:
            conn.execute('''
                INSERT INTO complaints (complaint_id, name, phone_number, email, complaint_details)
                VALUES (?, ?, ?, ?, ?)
            ''', (complaint_id, name, phone_number, email, complaint_details))
            conn.commit()

        app.logger.info(f'✅ Complaint created successfully: {complaint_id}')

        return jsonify({
            'complaint_id': complaint_id,
            'message': 'Complaint created successfully'
        }), 201

    except sqlite3.Error as e:
        app.logger.error(f'❌ Database error: {e}')
        return jsonify({
            'error': 'Database error',
            'message': 'Failed to create complaint due to database issue'
        }), 500

    except Exception as e:
        app.logger.error(f'❌ Unexpected error: {e}')
        return jsonify({
            'error': 'Internal server error',
            'message': 'Something went wrong while creating the complaint'
        }), 500


@app.route('/complaints/<complaint_id>', methods=['GET'])
def get_complaint(complaint_id):
    try:

        complaint_id = complaint_id.strip()

        if not complaint_id:
            return jsonify({
                'error': 'Invalid complaint ID',
                'message': 'Complaint ID cannot be empty'
            }), 400

        with get_db_connection() as conn:
            complaint = conn.execute('''
                SELECT complaint_id, name, phone_number, email, complaint_details, created_at
                FROM complaints WHERE complaint_id = ?
            ''', (complaint_id,)).fetchone()

        if not complaint:
            return jsonify({
                'error': 'Complaint not found',
                'message': 'No complaint exists with the provided ID'
            }), 404

        app.logger.info(f'✅ Complaint retrieved successfully: {complaint_id}')

        return jsonify({
            'complaint_id': complaint['complaint_id'],
            'name': complaint['name'],
            'phone_number': complaint['phone_number'],
            'email': complaint['email'],
            'complaint_details': complaint['complaint_details'],
            'Created At': complaint['created_at']
        }), 200

    except sqlite3.Error as e:
        app.logger.error(f'❌ Database error: {e}')
        return jsonify({
            'error': 'Database error',
            'message': 'Failed to retrieve complaint due to database issue'
        }), 500

    except Exception as e:
        app.logger.error(f'❌ Unexpected error: {e}')
        return jsonify({
            'error': 'Internal server error',
            'message': 'Something went wrong while retrieving the complaint'
        }), 500


# Health check endpoint
@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    try:
        # Test database connection
        with get_db_connection() as conn:
            conn.execute('SELECT 1').fetchone()

        return jsonify({
            'status': 'healthy',
            'database': 'connected',
            'timestamp': datetime.now().isoformat()
        }), 200
    except Exception as e:
        return jsonify({
            'status': 'unhealthy',
            'database': 'disconnected',
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

if __name__ == '__main__':
    # Initialize database
    init_db()

    # Print startup information
    print("🚀 Smart Complaint Assistant API Server Starting...")
    print("📋 Available endpoints:")
    print("  POST   http://localhost:5000/complaints")
    print("  GET    http://localhost:5000/complaints/<complaint_id>")
    print("=" * 60)

    # Run the application
    app.run(debug=True, host='0.0.0.0', port=5000)