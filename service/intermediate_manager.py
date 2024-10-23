# file: intermediate_manager.py
# directory: .
import os
import subprocess
import threading
import time
import json
from datetime import datetime
from flask import Flask, request, jsonify, render_template, redirect, url_for, send_from_directory
from sqlalchemy import create_engine, Column, Integer, String, Boolean, DateTime, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

app = Flask(__name__, template_folder='templates')
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['FAVICON_FOLDER'] = 'favicon'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['FAVICON_FOLDER'], exist_ok=True)

# Глобальные переменные для хранения состояния парсера
parser_process = None
parser_stats = {}
parser_lock = threading.Lock()
run_history = []
current_run_id = None  # Новая глобальная переменная для хранения текущего run_id

# Конфигурация базы данных
DB_HOST = os.getenv("DB_HOST")
DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DATABASE_URL = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}/{DB_NAME}"

engine = create_engine(DATABASE_URL)
Base = declarative_base()
Session = sessionmaker(bind=engine)
session = Session()

# Определение моделей базы данных
class ParserRun(Base):
    __tablename__ = 'parser_runs'
    id = Column(Integer, primary_key=True)
    instance_id = Column(Integer, nullable=False)
    host = Column(String, nullable=False)
    start_time = Column(DateTime, default=datetime.utcnow)
    end_time = Column(DateTime, nullable=True)
    parameters = Column(JSON, nullable=True)
    stats = Column(JSON, nullable=True)

# Создание таблиц, если их нет
Base.metadata.create_all(engine)

# Получение идентификатора инстанса и информации о серверах
INSTANCE_ID = int(os.getenv("INSTANCE_ID", "0"))
TOTAL_SERVERS = int(os.getenv("TOTAL_SERVERS", "1"))
HOST_NAME = os.uname()[1]

def save_run_history(run):
    with parser_lock:
        run_history.append(run)

def update_run_in_db(run_id, end_time=None, stats=None):
    run = session.query(ParserRun).filter_by(id=run_id).first()
    if run:
        if end_time:
            run.end_time = end_time
        if stats:
            run.stats = stats
        session.commit()

def collect_stats(run_id):
    global parser_stats
    while True:
        if parser_process and parser_process.poll() is None:
            if os.path.exists('stats.json'):
                with open('stats.json', 'r') as f:
                    parser_stats = json.load(f)
                # Обновляем статистику в базе данных
                update_run_in_db(run_id, stats=parser_stats)
        else:
            break
        time.sleep(5)

@app.route('/favicon.ico')
def favicon():
    return send_from_directory(app.config['FAVICON_FOLDER'], 'favicon.ico', mimetype='image/vnd.microsoft.icon')
@app.route('/', methods=['GET'])
def index():
    status = 'stopped'
    if parser_process and parser_process.poll() is None:
        status = 'running'
    # Получаем историю запусков из базы данных
    runs = session.query(ParserRun).filter_by(instance_id=INSTANCE_ID).order_by(ParserRun.start_time.desc()).all()
    return render_template('parser_status.html', status=status, stats=parser_stats, runs=runs, instance_id=INSTANCE_ID, host_name=HOST_NAME, total_servers=TOTAL_SERVERS)

@app.route('/start_parser', methods=['POST'])
def start_parser():
    global parser_process
    global current_run_id  # Объявляем, что будем использовать глобальную переменную
    data = request.form
    start = data.get('start', 0)
    limit = data.get('limit', 50)
    if parser_process and parser_process.poll() is None:
        return jsonify({'status': 'Parser is already running'}), 400
    command = ['python', 'main.py', '-s', str(start), '-l', str(limit)]
    parser_process = subprocess.Popen(command)
    # Сохраняем информацию о запуске в базе данных
    new_run = ParserRun(
        instance_id=INSTANCE_ID,
        host=HOST_NAME,
        parameters={'start': start, 'limit': limit},
    )
    session.add(new_run)
    session.commit()
    current_run_id = new_run.id  # Сохраняем текущий run_id
    # Запускаем поток для сбора статистики
    stats_thread = threading.Thread(target=collect_stats, args=(current_run_id,))
    stats_thread.daemon = True
    stats_thread.start()
    return redirect(url_for('index'))

@app.route('/stop_parser', methods=['POST'])
def stop_parser():
    global parser_process
    global current_run_id  # Используем глобальную переменную
    if parser_process and parser_process.poll() is None:
        parser_process.terminate()
        parser_process = None
        # Обновляем информацию о завершении в базе данных
        update_run_in_db(current_run_id, end_time=datetime.utcnow())
        current_run_id = None  # Сбрасываем current_run_id
        return redirect(url_for('index'))
    else:
        return jsonify({'status': 'Parser is not running'}), 400

@app.route('/parser_status', methods=['GET'])
def parser_status():
    global parser_process
    status = 'stopped'
    if parser_process and parser_process.poll() is None:
        status = 'running'
    return jsonify({'status': status, 'stats': parser_stats}), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
