import logging
import os
import sys
import json
import mimetypes
import uuid
import re
from flask import Flask, request, jsonify, render_template, session, redirect, url_for, send_from_directory
from flask_cors import CORS
import database

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add Agents to path so we can import ocr_node_agent
base_dir = os.path.dirname(os.path.abspath(__file__))
agents_dir = os.path.join(base_dir, 'Agents')
rag_dir = os.path.join(base_dir, 'RAG')
if agents_dir not in sys.path:
    sys.path.append(agents_dir)
if base_dir not in sys.path:
    sys.path.insert(0, base_dir)

from ocr_node_agent import chain
from pdf_processor import process_pdf
from doc_processor import process_doc
from text_processor import process_text, process_text_content
from image_processor import process_image

IMAGE_EXTENSIONS  = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.tiff'}
PDF_EXTENSIONS    = {'.pdf'}
DOC_EXTENSIONS    = {'.doc', '.docx'}
TEXT_EXTENSIONS   = {'.txt'}
URL_REGEX         = re.compile(r'https?://[^\s]+')


app = Flask(__name__)
app.secret_key = 'super_secret_key_for_node_tree'
CORS(app)

UPLOAD_FOLDER = os.path.join(base_dir, 'OCR', 'Upload')
PROCESSED_FOLDER = os.path.join(base_dir, 'OCR', 'Processed')
ATTACHMENTS_DIR = database.ATTACHMENTS_DIR

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)
os.makedirs(ATTACHMENTS_DIR, exist_ok=True)

ALLOWED_ATTACHMENT_EXTENSIONS = {
    '.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp',
    '.pdf', '.doc', '.docx', '.txt', '.csv', '.xls', '.xlsx',
    '.ppt', '.pptx', '.zip', '.mp3', '.mp4',
}
MAX_ATTACHMENT_SIZE = 50 * 1024 * 1024  # 50 MB

# --- AUTH & DASHBOARD ROUTES ---
@app.route('/')
def index():
    if 'user_id' in session:
        return redirect(url_for('serve_app'))
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    error = None
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        user = database.authenticate(username, password)
        if user:
            session['user_id'] = user['id']
            session['is_admin'] = user['is_admin']
            logger.info(f"User '{username}' (ID: {user['id']}) logged in successfully.")
            return redirect(url_for('serve_app'))
        else:
            logger.warning(f"Failed login attempt for username '{username}'.")
            error = "Invalid username or password"
    return render_template('login.html', error=error)

@app.route('/signup', methods=['POST'])
def signup():
    username = request.form.get('username')
    email = request.form.get('email')
    password = request.form.get('password')
    first_name = request.form.get('first_name', '')
    last_name = request.form.get('last_name', '')
    location = request.form.get('location', '')
    
    if not username or not email or not password:
        return render_template('login.html', error="All fields are required for Sign Up.", mode='signup')
        
    user_id = database.create_user(username, email, password, first_name, last_name, location)
    if user_id:
        session['user_id'] = user_id
        session['is_admin'] = False
        logger.info(f"New user registered successfully: '{username}' (Email: {email}, ID: {user_id})")
        return redirect(url_for('serve_app'))
    else:
        logger.warning(f"Failed signup attempt: Username '{username}' already exists.")
        return render_template('login.html', error="Username already exists.", mode='signup')

@app.route('/logout')
def logout():
    user_id = session.get('user_id')
    session.clear()
    logger.info(f"User ID {user_id} logged out.")
    return redirect(url_for('login'))

@app.route('/app')
def serve_app():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    # Serve the node-tree html from the Frontend folder
    frontend_dir = os.path.join(os.path.dirname(base_dir), 'Frontend')
    return send_from_directory(frontend_dir, 'node-tree_11.html')

@app.route('/frontend/<path:filename>')
def static_frontend(filename):
    """Serve Frontend static files (canvas-editor.html, etc.)"""
    frontend_dir = os.path.join(os.path.dirname(base_dir), 'Frontend')
    logger.info(f"Serving frontend file: {filename} from {frontend_dir}")
    return send_from_directory(frontend_dir, filename)

@app.route('/api/me')
def api_me():
    if 'user_id' not in session:
        return jsonify({'error': 'Unauthorized'}), 401
    
    user = database.get_user_by_id(session['user_id'])
    username = user['username'] if user else 'User'
    first_name = user['first_name'] if user and user.get('first_name') else ''
    last_name = user['last_name'] if user and user.get('last_name') else ''
    if not first_name:
        first_name = username
    
    return jsonify({
        'user_id': session['user_id'],
        'username': username,
        'first_name': first_name,
        'last_name': last_name,
        'is_admin': session.get('is_admin', False)
    })

@app.route('/admin')
def admin_dashboard():
    if not session.get('is_admin'):
        return "Unauthorized", 403
    users = database.get_all_users()
    return render_template('admin.html', users=users)

@app.route('/admin/update_user/<int:user_id>', methods=['POST'])
def admin_update_user(user_id):
    if not session.get('is_admin'):
        logger.warning(f"Unauthorized admin action attempt by User ID {session.get('user_id')}")
        return "Unauthorized", 403
    username = request.form.get('username')
    email = request.form.get('email')
    password = request.form.get('password')
    first_name = request.form.get('first_name')
    last_name = request.form.get('last_name')
    location = request.form.get('location')
    database.update_user(user_id, username, email, password, first_name, last_name, location)
    logger.info(f"Admin (User ID: {session.get('user_id')}) updated details for User ID: {user_id}")
    return redirect(url_for('admin_dashboard'))

@app.route('/admin/view_map/<int:user_id>')
def admin_view_map(user_id):
    if not session.get('is_admin'):
        return "Unauthorized", 403
    return redirect(url_for('serve_app', user_id=user_id))

@app.route('/api/nodes', methods=['GET', 'POST'])
def api_nodes():
    if 'user_id' not in session:
        return jsonify({'error': 'Unauthorized'}), 401
    
    # If admin is viewing another user's map (passed via query param)
    target_user_id = request.args.get('user_id', session['user_id'])
    if int(target_user_id) != session['user_id'] and not session.get('is_admin'):
        logger.warning(f"User ID {session['user_id']} attempted unauthorized map access for User ID {target_user_id}")
        return jsonify({'error': 'Unauthorized'}), 403
        
    if request.method == 'GET':
        nodes = database.get_user_maps(target_user_id)
        logger.info(f"Node maps loaded for User ID {target_user_id} (Requested by User ID {session['user_id']})")
        return jsonify({'nodes': nodes})
    else:
        # POST save nodes
        nodes_data = request.json.get('nodes')
        database.save_user_maps(target_user_id, nodes_data)
        logger.info(f"Node maps saved for User ID {target_user_id} (Initiated by User ID {session['user_id']})")
        # Cascade-delete attachments for deleted nodes
        _orphan_cleanup_on_save(int(target_user_id), nodes_data or [])
        # Auto incremental ingest — only new/changed nodes are re-embedded
        try:
            from RAG.pipeline import incremental_ingest
            rag_result = incremental_ingest(int(target_user_id))
            logger.info(f"Auto RAG incremental ingest for user_id={target_user_id}: {rag_result}")
        except Exception as e:
            logger.warning(f"Auto RAG ingest failed (non-critical): {e}")
        return jsonify({'message': 'Saved successfully'})

@app.route('/api/snapshots', methods=['GET'])
def api_snapshots():
    if 'user_id' not in session:
        return jsonify({'error': 'Unauthorized'}), 401
    
    target_user_id = request.args.get('user_id', session['user_id'])
    if int(target_user_id) != session['user_id'] and not session.get('is_admin'):
        return jsonify({'error': 'Unauthorized'}), 403
        
    snapshots = database.get_user_snapshots(target_user_id)
    return jsonify({'snapshots': snapshots})

@app.route('/api/snapshots/bulk', methods=['DELETE'])
def api_snapshots_bulk_delete():
    if 'user_id' not in session:
        return jsonify({'error': 'Unauthorized'}), 401
        
    target_user_id = request.args.get('user_id', session['user_id'])
    if int(target_user_id) != session['user_id'] and not session.get('is_admin'):
        return jsonify({'error': 'Unauthorized'}), 403
        
    data = request.json or {}
    snapshot_ids = data.get('ids', [])
    
    if not snapshot_ids:
        return jsonify({'error': 'No snapshot IDs provided'}), 400
        
    try:
        database.delete_snapshots(snapshot_ids, int(target_user_id))
        return jsonify({'message': 'Snapshots deleted successfully'}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/snapshots/<int:snapshot_id>', methods=['GET', 'DELETE'])
def api_snapshot_data(snapshot_id):
    if 'user_id' not in session:
        return jsonify({'error': 'Unauthorized'}), 401
        
    target_user_id = request.args.get('user_id', session['user_id'])
    if int(target_user_id) != session['user_id'] and not session.get('is_admin'):
        return jsonify({'error': 'Unauthorized'}), 403
        
    if request.method == 'DELETE':
        try:
            database.delete_snapshot(snapshot_id, int(target_user_id))
            return jsonify({'message': 'Snapshot deleted successfully'}), 200
        except Exception as e:
            return jsonify({'error': str(e)}), 500
        
    data = database.get_snapshot_data(snapshot_id, target_user_id)
    if data is None:
        return jsonify({'error': 'Snapshot not found'}), 404
        
    return jsonify({'nodes': data})

def check_ocr_garbled(nodes_data):
    """
    Checks if nodes_data represents a garbled OCR error response.
    Returns (is_garbled, error_message)
    """
    if not nodes_data:
        return False, ""
    
    # Determine the list of nodes
    nodes_list = []
    if isinstance(nodes_data, dict) and "nodes" in nodes_data:
        nodes_list = nodes_data["nodes"]
    elif isinstance(nodes_data, list):
        nodes_list = nodes_data
    
    for node in nodes_list:
        if not isinstance(node, dict):
            continue
        desc = node.get("description", "")
        label = node.get("label", "")
        # Look for indicators of garbled OCR content
        if "garbled" in desc.lower() and "discernible" in desc.lower():
            return True, desc
        if "garbled" in label.lower() and "discernible" in label.lower():
            return True, label
        if "unintelligible text" in label.lower() and "garbled" in desc.lower():
            return True, desc
            
    return False, ""


@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    filename = file.filename
    ext = os.path.splitext(filename)[1].lower()
    file_path = os.path.join(UPLOAD_FOLDER, filename)
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    file.save(file_path)

    output_name = os.path.splitext(filename)[0] + "_structure.json"
    output_path = os.path.join(PROCESSED_FOLDER, output_name)
    if os.path.exists(output_path):
        os.remove(output_path)

    try:
        logger.info("User %s uploaded '%s' (type: %s)", session.get('user_id'), filename, ext)

        if ext in PDF_EXTENSIONS:
            nodes_data = process_pdf(file_path, output_path)

        elif ext in DOC_EXTENSIONS:
            nodes_data = process_doc(file_path, output_path)

        elif ext in TEXT_EXTENSIONS:
            nodes_data = process_text(file_path, output_path)

        elif ext in IMAGE_EXTENSIONS:
            nodes_data = process_image(file_path, output_path)

        else:
            return jsonify({'error': f'Unsupported file type: {ext}. Supported: images, .pdf, .doc, .docx, .txt'}), 400

        # Check if the nodes contain garbled content error message
        is_garbled, garbled_msg = check_ocr_garbled(nodes_data)
        if is_garbled:
            logger.warning("OCR processing resulted in garbled text: %s", garbled_msg)
            if os.path.exists(output_path):
                try:
                    os.remove(output_path)
                except Exception:
                    pass
            public_ocr_path = os.path.abspath(os.path.join(base_dir, "..", "MindForge", "public", "ocr_nodes.json"))
            if os.path.exists(public_ocr_path):
                try:
                    os.remove(public_ocr_path)
                except Exception:
                    pass
            return jsonify({'error': f'Image: {garbled_msg}'}), 422

        logger.info("File '%s' successfully processed into nodes.", filename)
        return jsonify({'message': 'Success', 'data': nodes_data}), 200

    except Exception as e:
        logger.error("Error processing upload '%s': %s", filename, e)
        return jsonify({'error': str(e)}), 500

@app.route('/generate-text', methods=['POST'])
def generate_text():
    data = request.json
    text = data.get('text', '') if data else ''
    if not text:
        return jsonify({'error': 'No text provided'}), 400
    
    try:
        lower_text = text.lower().strip()
        urls = URL_REGEX.findall(text)
        is_url_only = text.strip().startswith(('http://', 'https://')) and len(text.strip().split()) == 1
        has_node_keywords = any(keyword in lower_text for keyword in ["generate node", "create node", "mind map", "mindmap", "node", "structure"])

        if has_node_keywords or is_url_only:
            if urls:
                # 1. Process URL Node Generation
                url = urls[0]
                logger.info(f"User ID {session.get('user_id')} requested node generation for URL: {url}")
                output_name = f"url_{uuid.uuid4().hex[:8]}_structure.json"
                output_path = os.path.join(PROCESSED_FOLDER, output_name)
                
                from url_processor import process_url
                nodes_data = process_url(url, output_path)
            else:
                # 2. Process Text Node Generation
                logger.info(f"User ID {session.get('user_id')} requested node generation via text input.")
                output_name = f"text_{uuid.uuid4().hex[:8]}_structure.json"
                output_path = os.path.join(PROCESSED_FOLDER, output_name)
                
                from text_processor import process_text_content
                nodes_data = process_text_content(text, output_path)

            # Check if the nodes contain garbled content error message
            is_garbled, garbled_msg = check_ocr_garbled(nodes_data)
            if is_garbled:
                logger.warning("Node generation resulted in garbled text: %s", garbled_msg)
                return jsonify({'error': f'Generation error: {garbled_msg}'}), 422

            logger.info("Node generation completed successfully.")
            return jsonify({'type': 'nodes', 'message': 'Success', 'data': nodes_data}), 200
        else:
            user_id = session.get('user_id', 0)
            session_id = data.get('session_id', 'default_session')
            logger.info(f"User ID {user_id} initiated RAG/chat query.")
            try:
                from RAG.pipeline import query as rag_query
                result = rag_query(text, user_id=user_id, session_id=session_id)
                return jsonify({
                    'type': 'chat',
                    'message': result['answer'],
                    'sources': result.get('sources', []),
                    'mode': result.get('mode', 'rag'),
                    'metrics': result.get('metrics', {}),
                }), 200
            except Exception as rag_err:
                logger.warning(f"RAG pipeline unavailable ({rag_err}); falling back to chatbot.")
                from chatbot import chatbot_with_history
                response = chatbot_with_history.invoke(
                    {"input": text},
                    config={"configurable": {"session_id": session_id}}
                )
                return jsonify({'type': 'chat', 'message': response.content}), 200
    except Exception as e:
        logger.error(f"Error processing text input: {e}")
        return jsonify({'error': str(e)}), 500

# ── RAG API endpoints ─────────────────────────────────────────────────────────

@app.route('/rag')
def rag_panel():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    if not session.get('is_admin'):
        return "Unauthorized", 403
    return render_template('rag.html', user_id=session['user_id'])


@app.route('/api/rag/trace', methods=['POST'])
def rag_trace():
    """Step-by-step trace of a RAG query for debugging."""
    if 'user_id' not in session:
        return jsonify({'error': 'Unauthorized'}), 401
    question = (request.json or {}).get('question', 'ML description')
    user_id = session['user_id']
    trace = {}
    try:
        import sys, os
        sys.path.insert(0, os.path.join(base_dir))
        from RAG.embeddings import embed_query
        from RAG.vector_store import get_all_user_chunks, dense_search
        from RAG.retrieval import retrieve

        chunks = get_all_user_chunks(user_id)
        trace['chunks_in_qdrant'] = len(chunks)
        trace['sample_chunk'] = chunks[0] if chunks else None

        vec = embed_query(question)
        trace['query_embedded'] = True

        hits = dense_search(vec, user_id=user_id, top_k=3)
        trace['dense_hits'] = len(hits)
        trace['dense_results'] = [{'text': h['text'], 'score': h['score']} for h in hits]

        reranked = retrieve(question, user_id=user_id)
        trace['reranked_count'] = len(reranked)
        trace['reranked_results'] = [{'text': r['text']} for r in reranked]

    except Exception as e:
        import traceback
        trace['error'] = str(e)
        trace['traceback'] = traceback.format_exc()
    return jsonify(trace)


@app.route('/api/rag/debug')
def rag_debug():
    if 'user_id' not in session:
        return jsonify({'error': 'Unauthorized'}), 401
    user_id = session['user_id']
    nodes = database.get_user_maps(user_id)
    return jsonify({
        'session_user_id': user_id,
        'nodes_found': nodes is not None,
        'node_count': len(nodes) if nodes else 0,
        'sample': nodes[:2] if nodes else [],
    })


@app.route('/api/rag/ingest', methods=['POST'])
def rag_ingest():
    """Trigger ingestion of the current user's mind map into the RAG vector store."""
    if 'user_id' not in session:
        return jsonify({'error': 'Unauthorized'}), 401

    target_user_id = (request.json or {}).get('user_id', session['user_id'])
    if int(target_user_id) != session['user_id'] and not session.get('is_admin'):
        return jsonify({'error': 'Unauthorized'}), 403

    full = (request.json or {}).get('full', False)
    try:
        if full:
            # Full re-ingest: wipe all existing vectors then re-index from scratch.
            # Use this to purge stale/placeholder nodes like "New Node".
            from RAG.pipeline import ingest
            result = ingest(int(target_user_id))
        else:
            from RAG.pipeline import incremental_ingest
            result = incremental_ingest(int(target_user_id))
        logger.info(f"RAG ingest for user_id={target_user_id}: {result}")
        return jsonify({'message': 'Ingestion complete', **result}), 200
    except Exception as e:
        logger.error(f"RAG ingest error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/rag/stats', methods=['GET'])
def rag_stats():
    """Return RAG pipeline monitoring statistics (admin only)."""
    if not session.get('is_admin'):
        return jsonify({'error': 'Unauthorized'}), 403
    try:
        from RAG.monitoring import get_stats
        return jsonify(get_stats()), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ── Node Attachment API ───────────────────────────────────────────────────────

@app.route('/api/attachments/upload/<node_id>', methods=['POST'])
def upload_attachment(node_id):
    if 'user_id' not in session:
        return jsonify({'error': 'Unauthorized'}), 401
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']
    if not file.filename:
        return jsonify({'error': 'Empty filename'}), 400

    original_name = file.filename
    ext = os.path.splitext(original_name)[1].lower()
    if ext not in ALLOWED_ATTACHMENT_EXTENSIONS:
        return jsonify({'error': f'File type {ext} not allowed'}), 400

    # Read into memory to check size
    data = file.read()
    if len(data) > MAX_ATTACHMENT_SIZE:
        return jsonify({'error': 'File exceeds 50 MB limit'}), 400

    # Sanitize filename: nodeId_uuid_originalname
    safe_name = ''.join(c if c.isalnum() or c in '._-' else '_' for c in original_name)
    stored_name = f"{node_id}_{uuid.uuid4().hex[:8]}_{safe_name}"
    file_path = os.path.join(ATTACHMENTS_DIR, stored_name)

    with open(file_path, 'wb') as f:
        f.write(data)

    mime = mimetypes.guess_type(original_name)[0] or 'application/octet-stream'
    att_id = database.add_attachment(node_id, session['user_id'], original_name, stored_name, len(data), mime)
    logger.info("Attachment uploaded: node=%s user=%s file=%s", node_id, session['user_id'], stored_name)

    return jsonify({
        'id': att_id,
        'node_id': node_id,
        'original_name': original_name,
        'stored_name': stored_name,
        'file_size': len(data),
        'mime_type': mime,
    }), 201


@app.route('/api/attachments/<node_id>', methods=['GET'])
def list_attachments(node_id):
    if 'user_id' not in session:
        return jsonify({'error': 'Unauthorized'}), 401
    attachments = database.get_node_attachments(node_id, session['user_id'])
    return jsonify({'attachments': attachments})


@app.route('/api/attachments/download/<int:att_id>', methods=['GET'])
def download_attachment(att_id):
    if 'user_id' not in session:
        return jsonify({'error': 'Unauthorized'}), 401
    att = database.get_attachment(att_id, session['user_id'])
    if not att:
        return jsonify({'error': 'Not found'}), 404
    file_path = os.path.join(ATTACHMENTS_DIR, att['stored_name'])
    if not os.path.exists(file_path):
        return jsonify({'error': 'File missing from disk'}), 404
    return send_from_directory(ATTACHMENTS_DIR, att['stored_name'],
                               as_attachment=True,
                               download_name=att['original_name'])


@app.route('/api/attachments/view/<int:att_id>', methods=['GET'])
def view_attachment(att_id):
    if 'user_id' not in session:
        return jsonify({'error': 'Unauthorized'}), 401
    att = database.get_attachment(att_id, session['user_id'])
    if not att:
        return jsonify({'error': 'Not found'}), 404
    file_path = os.path.join(ATTACHMENTS_DIR, att['stored_name'])
    if not os.path.exists(file_path):
        return jsonify({'error': 'File missing from disk'}), 404
    mimetype = att['mime_type'] or 'text/html'
    response = send_from_directory(ATTACHMENTS_DIR, att['stored_name'], as_attachment=False)
    response.headers['Content-Type'] = mimetype
    response.headers.pop('Content-Disposition', None)
    return response


@app.route('/api/attachments/delete/<int:att_id>', methods=['DELETE'])
def delete_attachment(att_id):
    if 'user_id' not in session:
        return jsonify({'error': 'Unauthorized'}), 401
    stored_name = database.delete_attachment(att_id, session['user_id'])
    if not stored_name:
        return jsonify({'error': 'Not found'}), 404
    file_path = os.path.join(ATTACHMENTS_DIR, stored_name)
    if os.path.exists(file_path):
        os.remove(file_path)
    logger.info("Attachment deleted: id=%s user=%s", att_id, session['user_id'])
    return jsonify({'message': 'Deleted'})


def _orphan_cleanup_on_save(user_id, nodes_data):
    """Remove attachments whose node no longer exists in the saved map."""
    if not nodes_data:
        return
    live_ids = {n['id'] for n in nodes_data if isinstance(n, dict) and 'id' in n}
    orphan_names = database.cleanup_orphan_attachments(user_id, live_ids)
    for name in orphan_names:
        path = os.path.join(ATTACHMENTS_DIR, name)
        if os.path.exists(path):
            os.remove(path)
        logger.info("Orphan attachment removed: %s (user=%s)", name, user_id)


def _startup_orphan_scan():
    """On server start, delete any files in NodeAttachements/ that have no DB record."""
    try:
        if not os.path.isdir(ATTACHMENTS_DIR):
            return
        disk_files = set(os.listdir(ATTACHMENTS_DIR))
        # Collect all known stored_names across all users from DB
        from sqlite3 import connect
        conn = connect(database.DB_PATH)
        rows = conn.execute('SELECT stored_name FROM node_attachments').fetchall()
        conn.close()
        known = {r[0] for r in rows}
        orphans = disk_files - known
        for name in orphans:
            path = os.path.join(ATTACHMENTS_DIR, name)
            if os.path.isfile(path):
                os.remove(path)
                logger.info("Startup orphan scan removed: %s", name)
        if orphans:
            logger.info("Startup scan: removed %d orphan file(s)", len(orphans))
    except Exception as e:
        logger.warning("Startup orphan scan failed (non-critical): %s", e)


if __name__ == '__main__':
    database.init_db()
    _startup_orphan_scan()
    print("Starting NodeTree Backend Server on port 5000...")
    app.run(host='0.0.0.0', port=5000, debug=True)
