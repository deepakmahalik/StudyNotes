import sqlite3
import os
import json

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'database.db')
ATTACHMENTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'NodeAttachements'))

def get_db_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db_connection()
    c = conn.cursor()
    
    # Create Users table
    c.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            email TEXT NOT NULL,
            password TEXT NOT NULL,
            is_admin BOOLEAN NOT NULL DEFAULT 0
        )
    ''')
    
    # Safely add new columns if they don't exist
    try:
        c.execute('ALTER TABLE users ADD COLUMN full_name TEXT')
    except sqlite3.OperationalError:
        pass
    try:
        c.execute('ALTER TABLE users ADD COLUMN first_name TEXT')
    except sqlite3.OperationalError:
        pass
    try:
        c.execute('ALTER TABLE users ADD COLUMN last_name TEXT')
    except sqlite3.OperationalError:
        pass
    try:
        c.execute('ALTER TABLE users ADD COLUMN location TEXT')
    except sqlite3.OperationalError:
        pass

    # Migrate full_name to first_name and last_name if not done already
    c.execute('SELECT id, full_name, first_name FROM users')
    users_to_migrate = c.fetchall()
    for u in users_to_migrate:
        u_id = u['id']
        full_n = u['full_name']
        first_n = u['first_name']
        if first_n is None or first_n == '':
            if full_n:
                parts = full_n.strip().split(' ', 1)
                first = parts[0]
                last = parts[1] if len(parts) > 1 else ''
            else:
                first = ''
                last = ''
            c.execute('UPDATE users SET first_name = ?, last_name = ? WHERE id = ?', (first, last, u_id))

    # Create Maps table
    c.execute('''
        CREATE TABLE IF NOT EXISTS maps (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            nodes_json TEXT,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
    ''')
    
    # Create Snapshots table
    c.execute('''
        CREATE TABLE IF NOT EXISTS map_snapshots (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            nodes_json TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
    ''')

    # Node file attachments — stored on disk, only path recorded here
    c.execute('''
        CREATE TABLE IF NOT EXISTS node_attachments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            node_id TEXT NOT NULL,
            user_id INTEGER NOT NULL,
            original_name TEXT NOT NULL,
            stored_name TEXT NOT NULL,
            file_size INTEGER NOT NULL DEFAULT 0,
            mime_type TEXT NOT NULL DEFAULT '',
            uploaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
    ''')
    
    # Check if admin exists, if not seed users
    c.execute('SELECT * FROM users WHERE username = ?', ('admin',))
    if not c.fetchone():
        print("Seeding initial users...")
        # Seed admin
        c.execute('INSERT INTO users (username, email, password, is_admin, first_name, last_name, location) VALUES (?, ?, ?, ?, ?, ?, ?)',
                  ('admin', 'admin@example.com', 'admin', True, 'Admin', 'User', 'System'))
        
        # Seed 5 dummy users
        for i in range(1, 6):
            c.execute('INSERT INTO users (username, email, password, is_admin, first_name, last_name, location) VALUES (?, ?, ?, ?, ?, ?, ?)',
                      (f'user{i}', f'user{i}@dummy.com', 'password1', False, f'UserFirst{i}', f'UserLast{i}', 'Unknown'))
                      
    conn.commit()
    conn.close()

def authenticate(username, password):
    conn = get_db_connection()
    user = conn.execute('SELECT * FROM users WHERE username = ? AND password = ?', (username, password)).fetchone()
    conn.close()
    return dict(user) if user else None

def get_user_by_id(user_id):
    conn = get_db_connection()
    user = conn.execute('SELECT * FROM users WHERE id = ?', (user_id,)).fetchone()
    conn.close()
    return dict(user) if user else None

def get_all_users():
    conn = get_db_connection()
    users = conn.execute('SELECT id, username, email, is_admin, first_name, last_name, location FROM users').fetchall()
    conn.close()
    return [dict(u) for u in users]

def update_user(user_id, username, email, password, first_name, last_name, location):
    conn = get_db_connection()
    if password:
        conn.execute('UPDATE users SET username = ?, email = ?, password = ?, first_name = ?, last_name = ?, location = ? WHERE id = ?', 
                     (username, email, password, first_name, last_name, location, user_id))
    else:
        conn.execute('UPDATE users SET username = ?, email = ?, first_name = ?, last_name = ?, location = ? WHERE id = ?', 
                     (username, email, first_name, last_name, location, user_id))
    conn.commit()
    conn.close()

def get_user_maps(user_id):
    conn = get_db_connection()
    map_row = conn.execute('SELECT nodes_json FROM maps WHERE user_id = ?', (user_id,)).fetchone()
    conn.close()
    if map_row and map_row['nodes_json']:
        return json.loads(map_row['nodes_json'])
    return None

def save_user_maps(user_id, nodes_data):
    conn = get_db_connection()
    existing = conn.execute('SELECT id FROM maps WHERE user_id = ?', (user_id,)).fetchone()
    nodes_json = json.dumps(nodes_data)
    
    if existing:
        conn.execute('UPDATE maps SET nodes_json = ? WHERE id = ?', (nodes_json, existing['id']))
    else:
        conn.execute('INSERT INTO maps (user_id, nodes_json) VALUES (?, ?)', (user_id, nodes_json))
        
    # Create snapshot
    conn.execute('INSERT INTO map_snapshots (user_id, nodes_json) VALUES (?, ?)', (user_id, nodes_json))
        
    conn.commit()
    conn.close()

def get_user_snapshots(user_id):
    conn = get_db_connection()
    snapshots = conn.execute('SELECT id, created_at FROM map_snapshots WHERE user_id = ? ORDER BY created_at DESC', (user_id,)).fetchall()
    conn.close()
    return [dict(s) for s in snapshots]

def get_snapshot_data(snapshot_id, user_id):
    conn = get_db_connection()
    row = conn.execute('SELECT nodes_json FROM map_snapshots WHERE id = ? AND user_id = ?', (snapshot_id, user_id)).fetchone()
    conn.close()
    if row and row['nodes_json']:
        return json.loads(row['nodes_json'])
    return None

def delete_snapshot(snapshot_id, user_id):
    conn = get_db_connection()
    conn.execute('DELETE FROM map_snapshots WHERE id = ? AND user_id = ?', (snapshot_id, user_id))
    conn.commit()
    conn.close()

def delete_snapshots(snapshot_ids, user_id):
    if not snapshot_ids:
        return
    conn = get_db_connection()
    placeholders = ','.join('?' * len(snapshot_ids))
    query = f'DELETE FROM map_snapshots WHERE user_id = ? AND id IN ({placeholders})'
    params = [user_id] + snapshot_ids
    conn.execute(query, params)
    conn.commit()
    conn.close()

# ── Node attachment CRUD ─────────────────────────────────────────────────────

def add_attachment(node_id, user_id, original_name, stored_name, file_size, mime_type):
    conn = get_db_connection()
    cursor = conn.execute(
        'INSERT INTO node_attachments (node_id, user_id, original_name, stored_name, file_size, mime_type) VALUES (?,?,?,?,?,?)',
        (node_id, user_id, original_name, stored_name, file_size, mime_type)
    )
    att_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return att_id

def get_node_attachments(node_id, user_id):
    conn = get_db_connection()
    rows = conn.execute(
        'SELECT * FROM node_attachments WHERE node_id=? AND user_id=? ORDER BY uploaded_at ASC',
        (node_id, user_id)
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]

def get_attachment(att_id, user_id):
    conn = get_db_connection()
    row = conn.execute(
        'SELECT * FROM node_attachments WHERE id=? AND user_id=?',
        (att_id, user_id)
    ).fetchone()
    conn.close()
    return dict(row) if row else None

def delete_attachment(att_id, user_id):
    """Remove DB record and return stored_name so caller can delete the file."""
    conn = get_db_connection()
    row = conn.execute(
        'SELECT stored_name FROM node_attachments WHERE id=? AND user_id=?',
        (att_id, user_id)
    ).fetchone()
    if not row:
        conn.close()
        return None
    conn.execute('DELETE FROM node_attachments WHERE id=?', (att_id,))
    conn.commit()
    conn.close()
    return row['stored_name']

def delete_node_attachments(node_id, user_id):
    """Remove all DB records for a node; returns list of stored_names to delete from disk."""
    conn = get_db_connection()
    rows = conn.execute(
        'SELECT stored_name FROM node_attachments WHERE node_id=? AND user_id=?',
        (node_id, user_id)
    ).fetchall()
    names = [r['stored_name'] for r in rows]
    conn.execute('DELETE FROM node_attachments WHERE node_id=? AND user_id=?', (node_id, user_id))
    conn.commit()
    conn.close()
    return names

def cleanup_orphan_attachments(user_id, live_node_ids: set):
    """Delete DB records (and return stored_names) for nodes no longer in the map."""
    conn = get_db_connection()
    rows = conn.execute(
        'SELECT id, node_id, stored_name FROM node_attachments WHERE user_id=?',
        (user_id,)
    ).fetchall()
    orphans = [r for r in rows if r['node_id'] not in live_node_ids]
    names = [r['stored_name'] for r in orphans]
    ids = [r['id'] for r in orphans]
    if ids:
        conn.execute(f'DELETE FROM node_attachments WHERE id IN ({",".join("?"*len(ids))})', ids)
        conn.commit()
    conn.close()
    return names

def get_all_attachment_stored_names(user_id):
    """Return all stored_names recorded in DB for a user — used for orphan file scan."""
    conn = get_db_connection()
    rows = conn.execute(
        'SELECT stored_name FROM node_attachments WHERE user_id=?', (user_id,)
    ).fetchall()
    conn.close()
    return {r['stored_name'] for r in rows}

def get_all_users_with_attachments():
    conn = get_db_connection()
    rows = conn.execute('SELECT DISTINCT user_id FROM node_attachments').fetchall()
    conn.close()
    return [r['user_id'] for r in rows]


def create_user(username, email, password, first_name='', last_name='', location=''):
    conn = get_db_connection()
    try:
        cursor = conn.execute(
            'INSERT INTO users (username, email, password, is_admin, first_name, last_name, location) VALUES (?, ?, ?, ?, ?, ?, ?)',
            (username, email, password, False, first_name, last_name, location)
        )
        user_id = cursor.lastrowid
        conn.commit()
        return user_id
    except sqlite3.IntegrityError:
        # Username already exists (UNIQUE constraint failed)
        return None
    finally:
        conn.close()

if __name__ == '__main__':
    init_db()
    print("Database initialized successfully.")
