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
from werkzeug.serving import make_server
import sys

print("Python executable being used:", sys.executable)
print("Virtual environment prefix:", sys.prefix)

signal.signal(signal.SIGCHLD, signal.SIG_IGN)
# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

parser_state = {'status': 'stopped', 'stats': {}, 'stop_requested': False}
status_lock = threading.Lock()

app = Flask(__name__, template_folder='templates')
app.config['SECRET_KEY'] = 'your_secret_key'

app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['FAVICON_FOLDER'] = 'favicon'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['FAVICON_FOLDER'], exist_ok=True)

# Глобальные переменные для управления процессами парсера
parser_processes = {}  # Ключ: run_id, Значение: объект процесса

parser_lock = threading.Lock()
parser_stats = {}

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

# Событие для управления завершением работы сервера
shutdown_event = threading.Event()

def monitor_parser_process(run_id, parser_proc):
    logger.info(f"Monitor thread started for run_id: {run_id}")
    i = 0
    while True:
        time.sleep(1)
        i += 1
        if parser_proc.is_running():
            if i % 20 == 0:
                logger.info(f"Parser process with PID {parser_proc.pid} is running.")
            continue
        else:
            exit_code = parser_proc.wait()
            logger.info(f"Parser process with PID {parser_proc.pid} has terminated with exit code {exit_code}.")

            # Update parser_state if not running multiple runs
            with status_lock:
                if parser_state['status'] == 'running':
                    parser_state['status'] = 'stopped'
                    logger.info("Parser status set to 'stopped'.")

            # Update run in database
            update_run_in_db(run_id, end_time=datetime.now(timezone.utc))

            # Collect logs for run
            collect_logs_for_run(run_id)

            # Remove process from the list
            with parser_lock:
                parser_processes.pop(run_id, None)
            # Clear statistics and stats.json
            if os.path.exists('stats.json'):
                os.remove('stats.json')
            with parser_lock:
                parser_stats.clear()

            logger.info("Statistics cleared after parser process termination.")
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

def collect_stats(run_id, parser_proc):
    global parser_stats, parser_state
    previous_stats = {}
    last_modified_time = None
    session = Session()

    try:
        while True:
            if parser_proc.is_running():
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
    global parser_stats
    with status_lock:
        status = parser_state.get('status', 'stopped')
    session = Session()
    runs = session.query(ParserRun).filter_by(instance_id=INSTANCE_ID).order_by(ParserRun.start_time.desc()).all()
    session.close()
    return render_template('parser_status.html', status=status, stats=parser_stats, runs=runs, instance_id=INSTANCE_ID, host_name=HOST_NAME, total_servers=TOTAL_SERVERS)

@app.route('/start_parser', methods=['POST'])
def start_parser():
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
        'archive': 'archive' in data,
        'archive_type': data.get('archive_type', ''),
        'archive_config': data.get('archive_config', ''),
        'archive_threads': data.get('archive_threads', '4'),
        'service': 'service' in data,
        'port': data.get('port', '8070'),
        'query': data.get('query', ''),
        'mode': data.get('mode', 't'),
        'machine_id': data.get('machine_id', str(INSTANCE_ID)),
        'total_machines': data.get('total_machines', str(TOTAL_SERVERS)),
        'multiple_runs': 'multiple_runs' in data,
        'total_limit': data.get('total_limit', '100000'),
        'per_run_limit': data.get('per_run_limit', '50'),
    }

    # Проверка, запущен ли процесс парсера
    with status_lock:
        if parser_state['status'] == 'running' or parser_state['status'] == 'running_multiple':
            if request.accept_mimetypes.accept_json and not request.accept_mimetypes.accept_html:
                return jsonify({'status': 'Parser is already running'}), 400
            # Если запрос через HTML, перерисовываем страницу
            return redirect(url_for('index'))

    if parser_args.get('multiple_runs'):
        # Запускаем многократный запуск в отдельном потоке
        multiple_runs_thread = threading.Thread(target=run_multiple_parsers, args=(parser_args,))
        multiple_runs_thread.start()  # Не устанавливаем daemon=True
        with status_lock:
            parser_state['status'] = 'running_multiple'
        if request.accept_mimetypes.accept_json and not request.accept_mimetypes.accept_html:
            return jsonify({'status': 'Multiple runs started'}), 200
        else:
            return redirect(url_for('index'))
    else:
        # Запуск одиночного парсера
        run_id = start_single_parser(parser_args)
        with status_lock:
            parser_state['status'] = 'running'
        if request.accept_mimetypes.accept_json and not request.accept_mimetypes.accept_html:
            return jsonify({'status': 'Parser started', 'run_id': run_id}), 200
        else:
            return redirect(url_for('index'))

def start_single_parser(parser_args):
    # Установка переменных окружения
    os.environ['MACHINE_ID'] = parser_args['machine_id']
    os.environ['TOTAL_MACHINES'] = parser_args['total_machines']

    # Формируем команду для запуска парсера
    command = [sys.executable, 'main.py']
    # command = ['python', 'main.py']
    for key, value in parser_args.items():
        if key in ['multiple_runs', 'total_limit', 'per_run_limit']:
            continue  # Эти параметры не передаются в командную строку
        if isinstance(value, bool):
            if value:
                command.append(f'--{key.replace("_", "-")}')
        elif value:
            command.append(f'--{key.replace("_", "-")}')
            command.append(str(value))

    if os.path.exists('stats.json'):
       os.remove('stats.json')

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
    run_id = new_run.id
    session.close()

    os.environ['RUN_ID'] = str(run_id)

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
    run = session.query(ParserRun).filter_by(id=run_id).first()
    if run:
        run.pid = parser_subprocess.pid
        session.commit()
    session.close()

    parser_proc = psutil.Process(parser_subprocess.pid)
    print("Started parser process with pid", parser_subprocess.pid)

    # Сохраняем процесс в глобальном словаре
    with parser_lock:
        parser_processes[run_id] = parser_proc

    # Логи процесса
    stdout_log_file = f'logs/run_{run_id}_stdout.log'
    stderr_log_file = f'logs/run_{run_id}_stderr.log'

    threading.Thread(target=read_output, args=(parser_subprocess.stdout, stdout_log_file)).start()
    threading.Thread(target=read_output, args=(parser_subprocess.stderr, stderr_log_file)).start()

    # Статистика парсинга в отдельном потоке
    threading.Thread(target=collect_stats, args=(run_id, parser_proc)).start()

    # Мониторинг процесса парсера
    threading.Thread(target=monitor_parser_process, args=(run_id, parser_proc)).start()

    return run_id

def run_multiple_parsers(parser_args):
    logger.info("Starting multiple parser runs.")
    total_limit = int(parser_args['total_limit'])
    per_run_limit = int(parser_args['per_run_limit'])
    start_position = int(parser_args.get('start', '1'))

    num_runs = (total_limit + per_run_limit - 1) // per_run_limit
    logger.info(f"Total runs to execute: {num_runs}")

    for i in range(num_runs):
        logger.info(f"Starting run {i + 1} of {num_runs}")
        with status_lock:
            if parser_state.get('stop_requested') or shutdown_event.is_set():
                logger.info("Stop requested. Exiting multiple runs.")
                break

        current_start = start_position + i * per_run_limit
        current_limit = min(per_run_limit, total_limit - i * per_run_limit)
        logger.info(f"Run {i + 1}: Start position {current_start}, Limit {current_limit}")

        parser_args_run = parser_args.copy()
        parser_args_run['start'] = str(current_start)
        parser_args_run['limit'] = str(current_limit)
        parser_args_run['multiple_runs'] = False  # Avoid recursion

        run_id = start_single_parser(parser_args_run)

        # Wait for parser to finish
        while True:
            time.sleep(1)
            with parser_lock:
                parser_proc = parser_processes.get(run_id)
            if not parser_proc or not parser_proc.is_running():
                logger.info(f"Run {i + 1} completed.")
                break
            with status_lock:
                if parser_state.get('stop_requested') or shutdown_event.is_set():
                    # Stop current process
                    logger.info("Stop requested during run. Terminating parser process.")
                    try:
                        parser_proc.terminate()
                        parser_proc.wait(timeout=5)
                    except psutil.TimeoutExpired:
                        parser_proc.kill()
                    break

    with status_lock:
        parser_state['status'] = 'stopped'
        parser_state['stop_requested'] = False
    logger.info("All runs completed. Parser status set to 'stopped'.")

def stop_parser():
    global parser_processes

    with status_lock:
        parser_state['stop_requested'] = True

    # Останавливаем все запущенные процессы
    with parser_lock:
        for run_id, parser_proc in list(parser_processes.items()):
            if parser_proc.is_running():
                try:
                    parser_proc.terminate()
                    parser_proc.wait(timeout=5)
                except psutil.TimeoutExpired:
                    parser_proc.kill()
                finally:
                    # Удаляем процесс из списка после его остановки
                    parser_processes.pop(run_id, None)
                    # Обновляем run в базе данных
                    update_run_in_db(run_id, end_time=datetime.now(timezone.utc))
                    # Собираем логи
                    collect_logs_for_run(run_id)

    with status_lock:
        parser_state['status'] = 'stopped'
        parser_state['stats'] = {}
        parser_state['stop_requested'] = False  # Сбрасываем флаг после остановки

    with parser_lock:
        parser_stats.clear()

    logger.info('Parser stopped.')

@app.route('/stop_parser', methods=['POST'])
def stop_parser_route():
    stop_parser()
    if request.accept_mimetypes.accept_json and not request.accept_mimetypes.accept_html:
        return jsonify({'status': 'Parser stopped'}), 200
    else:
        return redirect(url_for('index'))

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
    run_log_dir = os.path.join(log_dir, f'run_{run_id}')
    os.makedirs(run_log_dir, exist_ok=True)

    for root, dirs, files in os.walk(log_dir):
        # Prevent descending into run_log_dir and other run directories
        dirs[:] = [d for d in dirs if not d.startswith('run_') or d == f'run_{run_id}']

        for file in files:
            # Only move files that match the current run_id or INSTANCE_ID
            if f'_{run_id}' in file or f'_{INSTANCE_ID}' in file:
                file_path = os.path.join(root, file)
                if os.path.isfile(file_path):
                    # Calculate relative path from log_dir
                    relative_path = os.path.relpath(file_path, log_dir)
                    dest_path = os.path.join(run_log_dir, relative_path)
                    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
                    shutil.move(file_path, dest_path)
    tar_file = os.path.join('logs', f'run_{run_id}.tar.gz')
    with tarfile.open(tar_file, 'w:gz') as tar:
        tar.add(run_log_dir, arcname=os.path.basename(run_log_dir))
    # Optionally remove the run_log_dir
    # shutil.rmtree(run_log_dir)
    if run:
        run.log_archive = tar_file  # Save the path to the archive
        session.commit()
    session.close()

@app.route('/parser_status', methods=['GET'])
def get_parser_status():
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
        # Logs of a finished process
        log_path = os.path.join('logs', f'run_{run_id}', log_file)
    else:
        # Current logs
        log_path = os.path.join('logs', log_file)

    if not os.path.exists(log_path):
        return "Log file not found", 404

    return render_template('view_log.html', log_file=log_file, run_id=run_id)


@app.route('/get_log_content/<int:run_id>/<path:log_path>')
def get_log_content(run_id, log_path):
    session = Session()
    try:
        run = session.query(ParserRun).filter_by(id=run_id).first()
        if not run:
            return jsonify({'error': 'Run not found'}), 404

        if run.end_time:
            # Logs of a finished process
            log_file_path = os.path.join('logs', f'run_{run_id}', log_path)
            tail_size = None  # Return full log if the run has ended
        else:
            # Current logs
            log_file_path = os.path.join('logs', log_path)
            tail_size = 100 * 1024  # 100 KB

        if not os.path.exists(log_file_path):
            return jsonify({'error': 'Log file not found'}), 404

        if tail_size:
            with open(log_file_path, 'rb') as f:
                f.seek(0, os.SEEK_END)
                file_size = f.tell()
                f.seek(max(file_size - tail_size, 0), os.SEEK_SET)
                content = f.read().decode('utf-8', errors='replace')
        else:
            with open(log_file_path, 'r', encoding='utf-8', errors='replace') as f:
                content = f.read()

        return jsonify({'content': content})
    finally:
        session.close()



@app.route('/list_logs/<int:run_id>', methods=['GET'])
def list_logs(run_id):
    session = Session()
    run = session.query(ParserRun).filter_by(id=run_id).first()
    session.close()

    logs = []
    if run.end_time:
        # Logs of a finished process are in logs/run_<run_id>
        log_dir = os.path.join('logs', f'run_{run_id}')
        if not os.path.exists(log_dir):
            return "Log directory not found", 404
        for root, dirs, files in os.walk(log_dir):
            for file in files:
                file_path = os.path.relpath(os.path.join(root, file), log_dir)
                logs.append(file_path)
    else:
        # Logs of a running process are in the root logs directory
        log_dir = 'logs'
        # Exclude run_<run_id> directory to avoid confusion
        for root, dirs, files in os.walk(log_dir):
            dirs[:] = [d for d in dirs if d != f'run_{run_id}']
            for file in files:
                if f'_{run_id}' in file or f'_{INSTANCE_ID}' in file:
                    file_path = os.path.relpath(os.path.join(root, file), log_dir)
                    logs.append(file_path)

    if not logs:
        return "No logs found for this run", 404

    return render_template('list_logs.html', logs=logs, run_id=run_id)


@app.route('/get_run_history', methods=['GET'])
def get_run_history():
    session = Session()
    runs = session.query(ParserRun).filter_by(instance_id=INSTANCE_ID).order_by(ParserRun.start_time.desc()).all()
    session.close()
    run_list = []
    for run in runs:
        run_list.append({
            'id': run.id,
            'start_time': run.start_time.isoformat(),
            'end_time': run.end_time.isoformat() if run.end_time else 'Running',
            'parameters': run.parameters,
            'log_archive': bool(run.log_archive),
        })
    return jsonify(run_list)

# Класс для управления сервером
class ServerThread(threading.Thread):
    def __init__(self, app):
        threading.Thread.__init__(self)
        self.server = make_server('0.0.0.0', 8080, app, threaded=True)
        self.ctx = app.app_context()
        self.ctx.push()

    def run(self):
        logger.info("Starting server...")
        self.server.serve_forever()

    def shutdown(self):
        logger.info("Shutting down server...")
        self.server.shutdown()
        self.ctx.pop()

# Функция для корректного завершения при получении сигналов
def graceful_shutdown(signum, frame):
    global server
    logger.info(f"Received shutdown signal: {signum}")
    # Устанавливаем флаг завершения
    shutdown_event.set()
    # Останавливаем парсеры
    stop_parser()
    logger.info("Server shutdown initiated.")
    # Останавливаем сервер
    if server:
        server.shutdown()

# Регистрация обработчиков сигналов
signal.signal(signal.SIGINT, graceful_shutdown)
signal.signal(signal.SIGTERM, graceful_shutdown)

if __name__ == '__main__':
    server = ServerThread(app)
    server.start()

    try:
        while not shutdown_event.is_set():
            time.sleep(1)
    except KeyboardInterrupt:
        graceful_shutdown(signal.SIGINT, None)

    logger.info("Server has been stopped.")