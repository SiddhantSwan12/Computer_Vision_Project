import sqlite3
import pandas as pd
import os
import glob
import datetime
from pathlib import Path
import json
import shutil

class DefectDatabase:
    """Database for storing and retrieving defect detection data"""
    
    def __init__(self, db_path="output/defect_database.db"):
        """Initialize the database"""
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        
        self.db_path = db_path
        self.connection = None
        self.cursor = None
        
        # Connect to database
        self._connect()
        
        # Create tables if they don't exist
        self._create_tables()
    
    def _connect(self):
        """Connect to the SQLite database"""
        self.connection = sqlite3.connect(self.db_path)
        self.connection.row_factory = sqlite3.Row  # Return rows as dictionaries
        self.cursor = self.connection.cursor()
    
    def _create_tables(self):
        """Create database tables if they don't exist"""
        # Defects table
        self.cursor.execute('''
        CREATE TABLE IF NOT EXISTS defects (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            object_id INTEGER NOT NULL,
            defect_type TEXT NOT NULL,
            severity REAL NOT NULL,
            timestamp TEXT NOT NULL,
            session_id TEXT NOT NULL,
            image_path TEXT
        )
        ''')
        
        # Sessions table
        self.cursor.execute('''
        CREATE TABLE IF NOT EXISTS sessions (
            id TEXT PRIMARY KEY,
            start_time TEXT NOT NULL,
            end_time TEXT,
            total_objects INTEGER DEFAULT 0,
            defect_objects INTEGER DEFAULT 0,
            total_defects INTEGER DEFAULT 0,
            metadata TEXT
        )
        ''')
        
        # Products table
        self.cursor.execute('''
        CREATE TABLE IF NOT EXISTS products (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            object_id INTEGER NOT NULL,
            session_id TEXT NOT NULL,
            shape TEXT,
            color TEXT,
            timestamp TEXT NOT NULL,
            has_defect BOOLEAN NOT NULL,
            FOREIGN KEY (session_id) REFERENCES sessions (id)
        )
        ''')
        
        # Daily stats table
        self.cursor.execute('''
        CREATE TABLE IF NOT EXISTS daily_stats (
            date TEXT PRIMARY KEY,
            total_objects INTEGER DEFAULT 0,
            defect_objects INTEGER DEFAULT 0,
            total_defects INTEGER DEFAULT 0,
            quality_score REAL
        )
        ''')
        
        self.connection.commit()
    
    def create_session(self, metadata=None):
        """Create a new session for recording defects"""
        session_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        start_time = datetime.datetime.now().isoformat()
        
        metadata_json = json.dumps(metadata) if metadata else None
        
        self.cursor.execute(
            "INSERT INTO sessions (id, start_time, metadata) VALUES (?, ?, ?)",
            (session_id, start_time, metadata_json)
        )
        self.connection.commit()
        
        return session_id
    
    def end_session(self, session_id, stats):
        """End a session and update its statistics"""
        end_time = datetime.datetime.now().isoformat()
        
        self.cursor.execute(
            """UPDATE sessions SET 
                end_time = ?, 
                total_objects = ?, 
                defect_objects = ?, 
                total_defects = ?
            WHERE id = ?""",
            (end_time, stats['total_objects'], stats['defect_objects'], 
             stats['total_defects'], session_id)
        )
        self.connection.commit()
    
    def add_defect(self, defect_data, session_id, image_path=None):
        """Add a defect record to the database"""
        # Insert defect record
        self.cursor.execute(
            """INSERT INTO defects 
                (object_id, defect_type, severity, timestamp, session_id, image_path) 
            VALUES (?, ?, ?, ?, ?, ?)""",
            (defect_data['object_id'], defect_data['defect_type'], 
             defect_data['severity'], defect_data['timestamp'], 
             session_id, image_path)
        )
        self.connection.commit()
    
    def add_product(self, product_data, session_id):
        """Add a product/object record to the database"""
        # Insert product record
        self.cursor.execute(
            """INSERT INTO products 
                (object_id, session_id, shape, color, timestamp, has_defect) 
            VALUES (?, ?, ?, ?, ?, ?)""",
            (product_data['id'], session_id, 
             product_data.get('shape', 'Unknown'), 
             product_data.get('color', 'Unknown'), 
             product_data['timestamp'], 
             product_data['defect']['has_defect'])
        )
        self.connection.commit()
    
    def update_daily_stats(self, date=None):
        """Update daily statistics table"""
        if date is None:
            date = datetime.datetime.now().strftime("%Y-%m-%d")
        
        # Get defect counts for the specified date
        self.cursor.execute(
            """SELECT COUNT(DISTINCT object_id) as defect_objects,
                     COUNT(*) as total_defects
              FROM defects 
              WHERE date(timestamp) = ?""",
            (date,)
        )
        defect_stats = dict(self.cursor.fetchone())
        
        # Get total object count
        self.cursor.execute(
            """SELECT COUNT(*) as total_objects
              FROM products 
              WHERE date(timestamp) = ?""",
            (date,)
        )
        result = self.cursor.fetchone()
        total_objects = dict(result)['total_objects'] if result else 0
        
        # Calculate quality score (0-100)
        quality_score = 100.0
        if total_objects > 0:
            defect_ratio = defect_stats['defect_objects'] / total_objects
            quality_score = 100 * (1 - defect_ratio)
        
        # Insert or update daily stats
        self.cursor.execute(
            """INSERT OR REPLACE INTO daily_stats 
                (date, total_objects, defect_objects, total_defects, quality_score) 
            VALUES (?, ?, ?, ?, ?)""",
            (date, total_objects, defect_stats['defect_objects'], 
             defect_stats['total_defects'], quality_score)
        )
        self.connection.commit()
    
    def import_csv_report(self, csv_path, image_dir=None, session_id=None):
        """Import a CSV defect report into the database"""
        if not os.path.exists(csv_path):
            print(f"Error: CSV file not found: {csv_path}")
            return False
        
        # Read CSV file
        df = pd.read_csv(csv_path)
        
        # Create a new session if not provided
        if session_id is None:
            metadata = {"imported_from": csv_path}
            session_id = self.create_session(metadata)
        
        # Import defects
        image_count = 0
        for _, row in df.iterrows():
            defect_data = {
                'object_id': row['object_id'],
                'defect_type': row['defect_type'],
                'severity': row['severity'],
                'timestamp': row['timestamp']
            }
            
            # Look for matching image
            image_path = None
            if image_dir and os.path.exists(image_dir):
                possible_images = glob.glob(os.path.join(
                    image_dir, f"defect_{row['object_id']}_{row['defect_type'].replace(' ', '_')}*.jpg"))
                if possible_images:
                    # Use the most recent image
                    newest_image = max(possible_images, key=os.path.getctime)
                    
                    # Copy to a more organized structure
                    date_part = datetime.datetime.strptime(
                        row['timestamp'].split()[0], "%Y-%m-%d").strftime("%Y%m%d")
                    
                    new_image_dir = os.path.join("output", "defect_library", 
                                               date_part, row['defect_type'].replace(' ', '_'))
                    os.makedirs(new_image_dir, exist_ok=True)
                    
                    new_image_path = os.path.join(
                        new_image_dir, 
                        f"obj_{row['object_id']}_{Path(newest_image).name}"
                    )
                    
                    # Only copy if it doesn't exist
                    if not os.path.exists(new_image_path):
                        shutil.copy(newest_image, new_image_path)
                        image_count += 1
                    
                    image_path = new_image_path
            
            # Add defect to database
            self.add_defect(defect_data, session_id, image_path)
        
        # Calculate statistics
        stats = {
            'total_objects': df['object_id'].nunique(),
            'defect_objects': df['object_id'].nunique(),
            'total_defects': len(df)
        }
        
        # End session with statistics
        self.end_session(session_id, stats)
        
        # Update daily statistics for all days in the dataset
        all_dates = pd.to_datetime(df['timestamp']).dt.date.unique()
        for date in all_dates:
            self.update_daily_stats(date.strftime("%Y-%m-%d"))
        
        print(f"Imported {len(df)} defect records from {csv_path}")
        print(f"Copied {image_count} defect images to the library")
        return session_id
    
    def get_defect_types(self):
        """Get all defect types and their counts"""
        self.cursor.execute(
            """SELECT defect_type, COUNT(*) as count
            FROM defects
            GROUP BY defect_type
            ORDER BY count DESC"""
        )
        return [dict(row) for row in self.cursor.fetchall()]
    
    def get_defect_stats_by_day(self, days=30):
        """Get defect statistics by day for the last N days"""
        end_date = datetime.datetime.now()
        start_date = end_date - datetime.timedelta(days=days)
        
        self.cursor.execute(
            """SELECT date, total_objects, defect_objects, total_defects, quality_score
            FROM daily_stats
            WHERE date BETWEEN ? AND ?
            ORDER BY date""",
            (start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"))
        )
        return [dict(row) for row in self.cursor.fetchall()]
    
    def get_defect_stats_by_type(self, start_date=None, end_date=None):
        """Get defect statistics by type within a date range"""
        query = """SELECT defect_type, 
                          COUNT(*) as count, 
                          AVG(severity) as avg_severity
                   FROM defects"""
        
        params = []
        if start_date or end_date:
            query += " WHERE "
            conditions = []
            
            if start_date:
                conditions.append("date(timestamp) >= ?")
                params.append(start_date)
            
            if end_date:
                conditions.append("date(timestamp) <= ?")
                params.append(end_date)
            
            query += " AND ".join(conditions)
        
        query += " GROUP BY defect_type ORDER BY count DESC"
        
        self.cursor.execute(query, params)
        return [dict(row) for row in self.cursor.fetchall()]
    
    def get_defect_images(self, defect_type=None, limit=20):
        """Get paths to defect images"""
        query = "SELECT image_path FROM defects WHERE image_path IS NOT NULL"
        params = []
        
        if defect_type:
            query += " AND defect_type = ?"
            params.append(defect_type)
        
        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)
        
        self.cursor.execute(query, params)
        results = self.cursor.fetchall()
        return [dict(row)['image_path'] for row in results if dict(row)['image_path']]
    
    def get_object_history(self, object_id):
        """Get defect history for a specific object"""
        self.cursor.execute(
            """SELECT * FROM defects
            WHERE object_id = ?
            ORDER BY timestamp""",
            (object_id,)
        )
        return [dict(row) for row in self.cursor.fetchall()]
    
    def search_defects(self, criteria):
        """Search defects based on various criteria"""
        query = "SELECT * FROM defects WHERE 1=1"
        params = []
        
        if 'object_id' in criteria:
            query += " AND object_id = ?"
            params.append(criteria['object_id'])
        
        if 'defect_type' in criteria:
            query += " AND defect_type = ?"
            params.append(criteria['defect_type'])
        
        if 'min_severity' in criteria:
            query += " AND severity >= ?"
            params.append(criteria['min_severity'])
        
        if 'max_severity' in criteria:
            query += " AND severity <= ?"
            params.append(criteria['max_severity'])
        
        if 'start_date' in criteria:
            query += " AND date(timestamp) >= ?"
            params.append(criteria['start_date'])
        
        if 'end_date' in criteria:
            query += " AND date(timestamp) <= ?"
            params.append(criteria['end_date'])
        
        if 'session_id' in criteria:
            query += " AND session_id = ?"
            params.append(criteria['session_id'])
        
        query += " ORDER BY timestamp DESC"
        
        if 'limit' in criteria:
            query += " LIMIT ?"
            params.append(criteria['limit'])
        
        self.cursor.execute(query, params)
        return [dict(row) for row in self.cursor.fetchall()]
    
    def close(self):
        """Close the database connection"""
        if self.connection:
            self.connection.close()
            self.connection = None
            self.cursor = None 