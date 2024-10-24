import os
import signal
import subprocess
import threading
import time
import json
import psutil
from datetime import datetime, timezone

import sqlalchemy
from flask import Flask, request, jsonify, render_template, redirect, url_for, send_from_directory, send_file
from sqlalchemy import create_engine, Column, Integer, String, Boolean, DateTime, JSON, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
import shutil
import tarfile
import io
import xlsxwriter
import logging
signal.signal(signal.SIGCHLD, signal.SIG_IGN)
# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

parser_state = {'status': 'stopped', 'stats': {}}
status_lock = threading.Lock()

app = Flask(__name__, template_folder='templates')
app.config['SECRET_KEY'] = 'your_secret_key'

app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['FAVICON_FOLDER'] = 'favicon'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['FAVICON_FOLDER'], exist_ok=True)

# Глобальные переменные для процесса парсера и потоков
parser_process = None
stdout_thread = None
stderr_thread = None
monitor_thread = None
parser_stats = {}
parser_lock = threading.Lock()
current_run_id = None

# Конфигурация базы данных
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_NAME = os.getenv("DB_NAME", "mydatabase")
DB_USER = os.getenv("DB_USER", "myuser")
DB_PASSWORD = os.getenv("DB_PASSWORD", "mypassword")
DATABASE_URL = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}/{DB_NAME}"

engine = create_engine(DATABASE_URL)
Base = sqlalchemy.orm.declarative_base()
Session = sessionmaker(bind=engine)

# Определение моделей базы данных
class ParserRun(Base):
    __tablename__ = 'parser_runs'
    id = Column(Integer, primary_key=True)
    instance_id = Column(Integer, nullable=False)
    host = Column(String, nullable=False)
    start_time = Column(DateTime, default=datetime.utcnow)
    end_time = Column(DateTime, nullable=True)
    parameters = Column(JSON, nullable=True)
    command_line = Column(String, nullable=True)
    log_archive = Column(String, nullable=True)
    pid = Column(Integer, nullable=True)
    stats_history = relationship("ParserRunStats", back_populates="parser_run", cascade="all, delete-orphan")

class ParserRunStats(Base):
    __tablename__ = 'parser_run_stats'
    id = Column(Integer, primary_key=True)
    run_id = Column(Integer, ForeignKey('parser_runs.id'), nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)
    stats = Column(JSON, nullable=False)
    incremental_stats = Column(JSON, nullable=True)
    parser_run = relationship("ParserRun", back_populates="stats_history")

# Создание таблиц, если их нет
Base.metadata.create_all(engine)

# Получение ID инстанса и информации о сервере
INSTANCE_ID = int(os.getenv("INSTANCE_ID", "0"))
TOTAL_SERVERS = int(os.getenv("TOTAL_SERVERS", "1"))
HOST_NAME = os.uname()[1]

def monitor_parser_process(run_id):
    global parser_process, current_run_id
    logger.info(f"Monitor thread started for run_id: {run_id}")
    i = 0
    while True:
        time.sleep(1)
        i += 1
        if parser_process is not None:
            if parser_process.is_running():
                if i % 20 == 0:
                    logger.info(f"Parser process with PID {parser_process.pid} is running.")
                continue
            else:
                logger.info(f"Parser process with PID {parser_process.pid} has terminated.")
        else:
            logger.info("Parser process is None.")

        with status_lock:
            parser_state['status'] = 'stopped'
            parser_state['stats'] = {}

        # Обновляем run в базе данных
        update_run_in_db(run_id, end_time=datetime.now(timezone.utc))

        # Перемещаем логи после завершения процесса
        collect_logs_for_run(run_id)

        # Очищаем глобальные переменные
        current_run_id = None
        parser_process = None

        # Очищаем статистику и stats.json
        if os.path.exists('stats.json'):
            os.remove('stats.json')
        with parser_lock:
            parser_stats = {}
        break


def read_output(pipe, log_file):
    with open(log_file, 'w') as f:
        for line in iter(pipe.readline, ''):
            if not line:
                break
            f.write(line)
            f.flush()
    pipe.close()

def update_run_in_db(run_id, end_time=None):
    session = Session()
    run = session.query(ParserRun).filter_by(id=run_id).first()
    if run:
        if end_time:
            run.end_time = end_time
        session.commit()
    session.close()

def calculate_incremental_stats(previous_stats, current_stats):
    incremental_stats = {}
    for key in current_stats:
        if key in previous_stats:
            if isinstance(current_stats[key], (int, float)) and isinstance(previous_stats[key], (int, float)):
                incremental_stats[key] = current_stats[key] - previous_stats[key]
            else:
                incremental_stats[key] = current_stats[key]
        else:
            incremental_stats[key] = current_stats[key]
    return incremental_stats

def collect_stats(run_id):
    global parser_stats, parser_state
    previous_stats = {}
    last_modified_time = None
    session = Session()

    try:
        while True:
            if parser_process and parser_process.is_running():
                if os.path.exists('stats.json'):
                    current_modified_time = os.path.getmtime('stats.json')
                    if last_modified_time != current_modified_time:
                        last_modified_time = current_modified_time
                        try:
                            with open('stats.json', 'r') as f:
                                data = f.read()
                            if data.strip() == '':
                                time.sleep(0.1)
                                continue

                            current_stats = json.loads(data)

                            # Обновляем глобальную статистику атомарно
                            with parser_lock:
                                parser_stats.update(current_stats)

                            if previous_stats:
                                incremental_stats = calculate_incremental_stats(previous_stats, current_stats)
                            else:
                                incremental_stats = current_stats

                            previous_stats = current_stats.copy()

                            stats_entry = ParserRunStats(
                                run_id=run_id,
                                stats=current_stats,
                                incremental_stats=incremental_stats
                            )
                            session.add(stats_entry)
                            session.commit()

                            # Обновляем общий статус парсера
                            with status_lock:
                                parser_state['stats'] = current_stats.copy()
                                parser_state['last_update'] = str(stats_entry.timestamp)

                        except json.JSONDecodeError as e:
                            print(f"JSON decode error: {e}")
                            time.sleep(0.1)
                            continue
                        except Exception as e:
                            print(f"Error in collect_stats: {str(e)}")
                            time.sleep(0.1)
                            continue
                time.sleep(1)
            else:
                break
    finally:
        session.close()

@app.route('/favicon.ico')
def favicon():
    return send_from_directory(app.config['FAVICON_FOLDER'], 'favicon.ico', mimetype='image/vnd.microsoft.icon')

@app.route('/', methods=['GET'])
def index():
    global parser_process, parser_stats
    status = 'stopped'
    if parser_process and parser_process.is_running():
        status = 'running'
    else:
        # Если парсер не запущен, очищаем stats.json
        if os.path.exists('stats.json'):
            os.remove('stats.json')
            with parser_lock:
                parser_stats = {}
    session = Session()
    runs = session.query(ParserRun).filter_by(instance_id=INSTANCE_ID).order_by(ParserRun.start_time.desc()).all()
    session.close()
    return render_template('parser_status.html', status=status, stats=parser_stats, runs=runs, instance_id=INSTANCE_ID, host_name=HOST_NAME, total_servers=TOTAL_SERVERS)


@app.route('/start_parser', methods=['POST'])
def start_parser():
    global parser_process, parser_stats
    global current_run_id
    global stdout_thread
    global stderr_thread
    global monitor_thread

    data = request.form or request.json or {}

    # Получаем все параметры из запроса
    parser_args = {
        'limit': data.get('limit', '1'),
        'start': data.get('start', '1'),
        'download_threads': data.get('download_threads', '8'),
        'batch_size': data.get('batch_size', '16'),
        'report_dir': data.get('report_dir', 'reports'),
        'stats_interval': data.get('stats_interval', '10'),
        'log_level': data.get('log_level', 'INFO'),
        'log_output': data.get('log_output', 'file'),
        'loggers': data.get('loggers', ''),
        'archive': data.get('archive', False),
        'archive_type': data.get('archive_type', ''),
        'archive_config': data.get('archive_config', ''),
        'archive_threads': data.get('archive_threads', '4'),
        'service': data.get('service', False),
        'port': data.get('port', '8070'),
        'query': data.get('query', ''),
        'mode': data.get('mode', 't')
    }

    # Проверка, запущен ли процесс парсера
    if parser_process and parser_process.is_running():
        if request.accept_mimetypes.accept_json and not request.accept_mimetypes.accept_html:
            return jsonify({'status': 'Parser is already running'}), 400
        # Если запрос через HTML, перерисовываем страницу
        return render_template('parser_status.html', status='running', stats=parser_stats)

    # Формируем команду для запуска парсера
    command = ['python', 'main.py']
    for key, value in parser_args.items():
        if isinstance(value, bool):
            if value:
                command.append(f'--{key.replace("_", "-")}')
        elif value:
            command.append(f'--{key.replace("_", "-")}')
            command.append(str(value))

    # Сохраняем параметры для записи в базу данных
    parameters = parser_args.copy()

    session = Session()
    new_run = ParserRun(
        instance_id=INSTANCE_ID,
        host=HOST_NAME,
        parameters=parameters,
        command_line=' '.join(command)
    )
    session.add(new_run)
    session.commit()
    current_run_id = new_run.id
    session.close()

    os.environ['RUN_ID'] = str(current_run_id)

    # Если файл stats.json существует и парсер не запущен, удаляем его
    if os.path.exists('stats.json'):
        os.remove('stats.json')
        with parser_lock:
            parser_stats = {}

    # Запускаем процесс парсинга
    parser_subprocess = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        bufsize=128 * 1024,
        universal_newlines=True
    )

    # Сохраняем PID в базе данных
    session = Session()
    run = session.query(ParserRun).filter_by(id=current_run_id).first()
    if run:
        run.pid = parser_subprocess.pid
        session.commit()
    session.close()

    parser_process = psutil.Process(parser_subprocess.pid)
    print("Started parser process with pid", parser_subprocess.pid)

    # Логи процесса
    stdout_log_file = f'logs/run_{current_run_id}_stdout.log'
    stderr_log_file = f'logs/run_{current_run_id}_stderr.log'

    stdout_thread = threading.Thread(target=read_output, args=(parser_subprocess.stdout, stdout_log_file))
    stderr_thread = threading.Thread(target=read_output, args=(parser_subprocess.stderr, stderr_log_file))

    stdout_thread.start()
    stderr_thread.start()

    # Статистика парсинга в отдельном потоке
    stats_thread = threading.Thread(target=collect_stats, args=(current_run_id,))
    stats_thread.daemon = True
    stats_thread.start()

    # Мониторинг процесса парсера
    monitor_thread = threading.Thread(target=monitor_parser_process, args=(current_run_id,))
    monitor_thread.daemon = True
    monitor_thread.start()

    with status_lock:
        parser_state['status'] = 'running'

    # Проверка типа запроса: если это JSON-запрос, возвращаем JSON
    if request.accept_mimetypes.accept_json and not request.accept_mimetypes.accept_html:
        return jsonify({'status': 'Parser started', 'run_id': current_run_id}), 200

    # Иначе возвращаем HTML
    return redirect(url_for('index'))


@app.route('/stop_parser', methods=['POST'])
def stop_parser():
    global parser_process, current_run_id, stdout_thread, stderr_thread

    if parser_process and parser_process.is_running():
        try:
            parser_process.terminate()
            parser_process.wait(timeout=5)
        except psutil.TimeoutExpired:
            parser_process.kill()

        parser_process = None

        if stdout_thread and stdout_thread.is_alive():
            stdout_thread.join()
        if stderr_thread and stderr_thread.is_alive():
            stderr_thread.join()

        update_run_in_db(current_run_id, end_time=datetime.utcnow())
        current_run_id = None

        # Очищаем статистику и stats.json
        if os.path.exists('stats.json'):
            os.remove('stats.json')
        with status_lock:
            parser_state['status'] = 'stopped'
            parser_state['stats'] = {}
        with parser_lock:
            parser_stats = {}

        return jsonify({'status': 'Parser stopped'}), 200
    else:
        return jsonify({'status': 'Parser is not running'}), 400

@app.route('/start_parser_api', methods=['POST'])
def start_parser_api():
    return start_parser()

@app.route('/archive_logs/<int:run_id>')
def archive_logs(run_id):
    collect_logs_for_run(run_id)
    return redirect(url_for('view_run_stats', run_id=run_id))

def collect_logs_for_run(run_id):
    session = Session()
    run = session.query(ParserRun).filter_by(id=run_id).first()
    if run and run.log_archive:
        session.close()
        return

    log_dir = 'logs'
    run_log_dir = f'logs/run_{run_id}'
    os.makedirs(run_log_dir, exist_ok=True)
    for root, dirs, files in os.walk(log_dir):
        for file in files:
            if f'_{run_id}' in file or f'_{INSTANCE_ID}' in file:
                file_path = os.path.join(root, file)
                if os.path.isfile(file_path):
                    relative_path = os.path.relpath(os.path.join(root, file), log_dir)
                    dest_path = os.path.join(run_log_dir, relative_path)
                    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
                    shutil.move(file_path, dest_path)
    tar_file = os.path.join('logs', f'run_{run_id}.tar.gz')
    with tarfile.open(tar_file, 'w:gz') as tar:
        tar.add(run_log_dir, arcname=os.path.basename(run_log_dir))
    # shutil.rmtree(run_log_dir)
    if run:
        run.log_archive = tar_file  # Сохраняем путь к архиву
        session.commit()
    session.close()



@app.route('/parser_status', methods=['GET'])
def get_parser_status():
    global parser_process
    with status_lock:
        status = parser_state.get('status', 'stopped')
        stats = parser_state.get('stats', {})
        last_update = parser_state.get('last_update', None)
    return jsonify({'status': status, 'stats': stats, 'last_update': last_update}), 200

@app.route('/view_run_stats/<int:run_id>')
def view_run_stats(run_id):
    session = Session()
    run = session.query(ParserRun).filter_by(id=run_id).first()
    if not run:
        session.close()
        return "Run not found", 404
    stats_entries = session.query(ParserRunStats).filter_by(run_id=run_id).order_by(ParserRunStats.timestamp).all()
    session.close()
    return render_template('run_stats.html', run=run, stats_entries=stats_entries, instance_id=INSTANCE_ID, host_name=HOST_NAME, total_servers=TOTAL_SERVERS)

@app.route('/get_run_stats/<int:run_id>')
def get_run_stats(run_id):
    session = Session()
    stats_entries = session.query(ParserRunStats).filter_by(run_id=run_id).order_by(ParserRunStats.timestamp).all()
    session.close()
    data = []
    for entry in stats_entries:
        data.append({
            'timestamp': entry.timestamp.isoformat(),
            'stats': entry.stats,
            'incremental_stats': entry.incremental_stats
        })
    return jsonify(data)

@app.route('/export_stats/<int:run_id>')
def export_stats(run_id):
    session = Session()
    run = session.query(ParserRun).filter_by(id=run_id).first()
    if not run:
        session.close()
        return "Run not found", 404
    stats_entries = session.query(ParserRunStats).filter_by(run_id=run_id).order_by(ParserRunStats.timestamp).all()
    session.close()
    output = io.BytesIO()
    workbook = xlsxwriter.Workbook(output)
    worksheet = workbook.add_worksheet()
    worksheet.write(0, 0, 'Timestamp')
    worksheet.write(0, 1, 'Total Files Downloaded')
    worksheet.write(0, 2, '+Δ Files Downloaded')
    worksheet.write(0, 3, 'Total Faces Found')
    worksheet.write(0, 4, '+Δ Faces Found')
    worksheet.write(0, 5, 'Total Embeddings Uploaded')
    worksheet.write(0, 6, '+Δ Embeddings Uploaded')
    row = 1
    for entry in stats_entries:
        worksheet.write(row, 0, str(entry.timestamp))
        stats = entry.stats
        inc_stats = entry.incremental_stats
        worksheet.write(row, 1, stats.get('total_files_downloaded', 0))
        worksheet.write(row, 2, inc_stats.get('total_files_downloaded', 0))
        worksheet.write(row, 3, stats.get('total_faces_found', 0))
        worksheet.write(row, 4, inc_stats.get('total_faces_found', 0))
        worksheet.write(row, 5, stats.get('total_embeddings_uploaded', 0))
        worksheet.write(row, 6, inc_stats.get('total_embeddings_uploaded', 0))
        row += 1
    workbook.close()
    output.seek(0)
    return send_file(output, attachment_filename=f'run_{run_id}_stats.xlsx', as_attachment=True)

@app.route('/delete_run/<int:run_id>')
def delete_run(run_id):
    session = Session()
    run = session.query(ParserRun).filter_by(id=run_id).first()
    if not run:
        session.close()
        return "Run not found", 404
    session.query(ParserRunStats).filter_by(run_id=run_id).delete()
    if run.log_archive and os.path.exists(run.log_archive):
        os.remove(run.log_archive)
    session.delete(run)
    session.commit()
    session.close()
    return redirect(url_for('index'))

@app.route('/download_logs/<int:run_id>', methods=['GET'])
def download_logs(run_id):
    session = Session()
    run = session.query(ParserRun).filter_by(id=run_id).first()
    if not run or not run.log_archive or not os.path.exists(run.log_archive):
        session.close()
        return "Logs not found", 404
    session.close()
    # Возвращаем сжатый файл tar.gz
    return send_file(run.log_archive, as_attachment=True)



@app.route('/view_log/<int:run_id>/<path:log_file>', methods=['GET'])
def view_log(run_id, log_file):
    session = Session()
    run = session.query(ParserRun).filter_by(id=run_id).first()
    session.close()

    if run.end_time:
        # Логи завершённого процесса
        log_path = os.path.join('logs', f'run_{run_id}', log_file)
    else:
        # Текущие логи
        log_path = os.path.join('logs', log_file)

    if not os.path.exists(log_path):
        return "Log file not found", 404
    return render_template('view_log.html', log_file=log_file, run_id=run_id)

@app.route('/get_log_content/<int:run_id>/<path:log_file>', methods=['GET'])
def get_log_content(run_id, log_file):
    session = Session()
    run = session.query(ParserRun).filter_by(id=run_id).first()
    session.close()

    if run.end_time:
        # Логи завершённого процесса
        log_path = os.path.join('logs', f'run_{run_id}', log_file)
    else:
        # Текущие логи
        log_path = os.path.join('logs', log_file)

    if not os.path.exists(log_path):
        return jsonify({"error": "Log file not found"}), 404

    with open(log_path, 'r') as file:
        content = file.read()[-10000:]
    return jsonify({"content": content})



@app.route('/list_logs/<int:run_id>', methods=['GET'])
def list_logs(run_id):
    session = Session()
    run = session.query(ParserRun).filter_by(id=run_id).first()
    session.close()

    logs = []
    if run.end_time:
        # Логи завершённого процесса находятся в logs/run_<run_id>
        log_dir = f'logs/run_{run_id}'
        if not os.path.exists(log_dir):
            return "Log directory not found", 404
        for root, dirs, files in os.walk(log_dir):
            for file in files:
                file_path = os.path.relpath(os.path.join(root, file), log_dir)
                logs.append(file_path)
    else:
        # Логи запущенного процесса находятся в исходных директориях
        log_dir = 'logs'
        for root, dirs, files in os.walk(log_dir):
            for file in files:
                if f'_{run_id}' in file or f'_{INSTANCE_ID}' in file:
                    file_path = os.path.relpath(os.path.join(root, file), log_dir)
                    logs.append(file_path)

    if not logs:
        return "No logs found for this run", 404

    return render_template('list_logs.html', logs=logs, run_id=run_id)



if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)
