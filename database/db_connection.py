import mysql.connector
from dotenv import load_dotenv
import os

load_dotenv()

def get_db_connection():
    """Create and return database connection"""
    try:
        connection = mysql.connector.connect(
            host=os.getenv('DB_HOST', 'localhost'),
            user=os.getenv('DB_USER', 'root'),
            password=os.getenv('DB_PASSWORD', ''),
            database=os.getenv('DB_NAME', 'skin_disease_db')
        )
        return connection
    except mysql.connector.Error as e:
        print(f"Database connection error: {e}")
        return None

def initialize_database():
    """Initialize the database schema"""
    connection = get_db_connection()
    if not connection:
        print("Failed to connect to database")
        return
    
    cursor = connection.cursor()
    
    try:
        # Create database if not exists
        cursor.execute("CREATE DATABASE IF NOT EXISTS skin_disease_db")
        cursor.execute("USE skin_disease_db")
        
        # Create tables (same as in app.py)
        tables_sql = [
            """
            CREATE TABLE IF NOT EXISTS doctor_verifications (
                id INT AUTO_INCREMENT PRIMARY KEY,
                original_diagnosis VARCHAR(255),
                verified_diagnosis VARCHAR(255),
                doctor_id VARCHAR(255),
                image_id VARCHAR(255),
                is_correct BOOLEAN,
                confidence_score FLOAT DEFAULT 0,
                notes TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS analysis_history (
                id INT AUTO_INCREMENT PRIMARY KEY,
                user_id VARCHAR(255),
                image_path TEXT,
                diagnosis VARCHAR(255),
                confidence FLOAT,
                is_cancer BOOLEAN,
                cancer_status VARCHAR(100),
                explanations JSON,
                doctor_verified BOOLEAN DEFAULT FALSE,
                doctor_correction VARCHAR(255),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS community_insights (
                id INT AUTO_INCREMENT PRIMARY KEY,
                date DATE NOT NULL,
                total_scans INT DEFAULT 0,
                benign_count INT DEFAULT 0,
                malignant_count INT DEFAULT 0,
                disease_breakdown JSON,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE KEY unique_date (date)
            )
            """
        ]
        
        for sql in tables_sql:
            cursor.execute(sql)
        
        connection.commit()
        print("Database initialized successfully")
        
    except mysql.connector.Error as e:
        print(f"Database initialization error: {e}")
    finally:
        cursor.close()
        connection.close()

if __name__ == "__main__":
    initialize_database()