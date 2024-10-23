# file: manager.py
# directory: azure
import os
import time
import requests
import json
from flask import Flask, request, jsonify, render_template, redirect, url_for
from azure.identity import DefaultAzureCredential
from azure.mgmt.compute import ComputeManagementClient
from azure.mgmt.resource import ResourceManagementClient
from sqlalchemy import create_engine, Column, Integer, String, Boolean, DateTime, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime

# Конфигурация Azure
subscription_id = os.getenv("AZURE_SUBSCRIPTION_ID")
credential = DefaultAzureCredential()

compute_client = ComputeManagementClient(credential, subscription_id)
resource_client = ResourceManagementClient(credential, subscription_id)

RESOURCE_GROUP_NAME = "h100-1_group"
LOCATION = "West Europe"

# Настройка базы данных
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
class Instance(Base):
    __tablename__ = 'instances'
    id = Column(Integer, primary_key=True)
    name = Column(String, unique=True, nullable=False)
    ip_address = Column(String, nullable=False)
    assigned_id = Column(Integer, nullable=False)
    is_alive = Column(Boolean, default=True)
    last_heartbeat = Column(DateTime, nullable=True)
    stats = Column(JSON, nullable=True)  # Новое поле для статистики

class InstanceStats(Base):
    __tablename__ = 'instance_stats'
    id = Column(Integer, primary_key=True)
    instance_id = Column(Integer, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)
    stats = Column(JSON, nullable=True)

# Создание таблиц, если их нет
Base.metadata.create_all(engine)

# Идентификаторы для новых инстансов
available_ids = list(range(50))
instances = {}

app = Flask(__name__, template_folder='templates')

@app.route('/get_id', methods=['GET'])
def get_id():
    vm_name = request.remote_addr
    instance = session.query(Instance).filter_by(ip_address=vm_name).first()
    if not instance:
        if available_ids:
            new_id = available_ids.pop()
            new_instance = Instance(name=vm_name, ip_address=vm_name, assigned_id=new_id, is_alive=True, last_heartbeat=datetime.utcnow())
            session.add(new_instance)
            session.commit()
            instances[vm_name] = new_id
            return jsonify({"id": new_id})
        else:
            return "No available IDs", 400
    else:
        instance.last_heartbeat = datetime.utcnow()
        session.commit()
        return jsonify({"id": instance.assigned_id})

@app.route('/heartbeat', methods=['POST'])
def heartbeat():
    data = request.json
    vm_name = request.remote_addr
    instance = session.query(Instance).filter_by(ip_address=vm_name).first()
    if instance:
        instance.last_heartbeat = datetime.utcnow()
        instance.is_alive = True
        session.commit()
        return "Heartbeat received", 200
    else:
        return "Instance not found", 404

@app.route('/stats', methods=['POST'])
def stats():
    data = request.json
    vm_name = request.remote_addr
    instance = session.query(Instance).filter_by(ip_address=vm_name).first()
    if instance:
        instance.stats = data  # Сохраняем статистику
        session.commit()
        return "Stats recorded", 200
    else:
        return "Instance not found", 404

@app.route('/', methods=['GET'])
def index():
    instances = session.query(Instance).all()
    return render_template('index.html', instances=instances)

@app.route('/instance/<int:instance_id>', methods=['GET'])
def instance_detail(instance_id):
    instance = session.query(Instance).filter_by(id=instance_id).first()
    if instance:
        return render_template('instance_detail.html', instance=instance)
    else:
        return "Instance not found", 404

@app.route('/start_parsers', methods=['POST'])
def start_parsers():
    data = request.form
    total_batches = int(data.get('total_batches', 2500))
    batches_per_parser = int(data.get('batches_per_parser', 50))
    start_position = int(data.get('start', 0))
    instances = session.query(Instance).filter_by(is_alive=True).all()
    num_instances = len(instances)
    if num_instances == 0:
        return "No alive instances", 400

    current_start = start_position
    tasks = []
    for instance in instances:
        parser_start = current_start
        parser_limit = batches_per_parser
        current_start += batches_per_parser

        url = f'http://{instance.ip_address}:8080/start_parser'
        payload = {'start': parser_start, 'limit': parser_limit}
        try:
            response = requests.post(url, json=payload, timeout=5)
            if response.status_code == 200:
                tasks.append({'instance_id': instance.id, 'start': parser_start, 'limit': parser_limit})
            else:
                print(f"Failed to start parser on {instance.name}")
        except requests.RequestException:
            print(f"Failed to connect to {instance.name}")
    return redirect(url_for('index'))

@app.route('/stop_parsers', methods=['POST'])
def stop_parsers():
    instances = session.query(Instance).filter_by(is_alive=True).all()
    for instance in instances:
        url = f'http://{instance.ip_address}:8080/stop_parser'
        try:
            response = requests.post(url, timeout=5)
            if response.status_code == 200:
                print(f"Stopped parser on {instance.name}")
            else:
                print(f"Failed to stop parser on {instance.name}")
        except requests.RequestException:
            print(f"Failed to connect to {instance.name}")
    return redirect(url_for('index'))

if __name__ == "__main__":
    # Запуск Flask-сервера
    app.run(host="0.0.0.0", port=5000)
