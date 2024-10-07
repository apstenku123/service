# file: archiver.py
# directory: .
import threading
import time
import os
import shutil
import traceback
from utils import configure_thread_logging, get_session_factory
from models import ArchivedImage, HostLog
import boto3
from azure.storage.blob import BlobServiceClient
import ftplib
import paramiko
import socket
import config  # Импортируем модуль конфигурации
def archiver_thread(archive_queue, engine, archive_type, archive_config, stats_collector, log_level, log_output):
    # global MACHINE_ID
    # Set up logger for this function
    log_filename = f'logs/archiver/archiver_{config.MACHINE_ID}.log'
    archiver_logger = configure_thread_logging('archiver', log_filename, log_level, log_output)

    SessionFactory = get_session_factory(engine)
    session = SessionFactory()
    host_name = socket.gethostname()

    # Check if HostLog exists
    existing_host_log = session.query(HostLog).filter_by(host_name=host_name, function_name='archiver', log_file=log_filename).first()
    if not existing_host_log:
        host_log = HostLog(host_name=host_name, function_name='archiver', log_file=log_filename)
        session.add(host_log)
        session.commit()
    else:
        host_log = existing_host_log

    while True:
        archive_info = archive_queue.get()
        if archive_info is None:
            archive_queue.task_done()
            break  # Termination signal

        start_time = time.time()
        batch_id = archive_info['batch_id']
        batch_dir = archive_info['batch_dir']
        filenames = archive_info['filenames']
        image_ids = archive_info['image_ids']
        filename_to_id = archive_info['filename_to_id']

        archiver_logger.info(f"Starting archiving for batch {batch_id}")

        # Archive images
        archive_urls = archive_batch(batch_dir, batch_id, archive_type, archive_config, archiver_logger)

        # Now, store the archive_urls in the database
        try:
            for filename, archive_url in archive_urls.items():
                img_id = filename_to_id.get(filename)
                if img_id:
                    archived_image = ArchivedImage(image_id=img_id, archive_url=archive_url)
                    session.add(archived_image)
            session.commit()
            archiver_logger.info(f"Archived images for batch {batch_id} stored in database.")
        except Exception as e:
            session.rollback()
            archiver_logger.error(f"Error storing archive URLs for batch {batch_id}: {e}")

        # Remove batch directory
        try:
            shutil.rmtree(batch_dir)
            archiver_logger.info(f"Removed temporary directory for batch {batch_id}")
        except Exception as e:
            archiver_logger.error(f"Error deleting batch directory {batch_dir}: {e}")

        # Update statistics
        stats_collector.increment_batches_archived()

        processing_time = time.time() - start_time
        stats_collector.add_batch_processing_time('archiver', processing_time)

        archive_queue.task_done()
        time.sleep(1)  # Small delay to reduce load
    session.close()

def archive_batch(batch_dir, batch_id, archive_type, archive_config, logger):
    if archive_type == 's3':
        return archive_batch_to_s3(batch_dir, archive_config, logger)
    elif archive_type == 'azure':
        return archive_batch_to_azure_blob(batch_dir, archive_config, logger)
    elif archive_type == 'ftp':
        return archive_batch_to_ftp(batch_dir, archive_config, logger)
    elif archive_type == 'sftp':
        return archive_batch_to_sftp(batch_dir, archive_config, logger)
    else:
        logger.error(f"Unknown archive type: {archive_type}")
        return {}

def archive_batch_to_s3(batch_dir, archive_config, logger):
    s3_endpoint = archive_config.get('s3_endpoint')
    s3_access_key = archive_config.get('s3_access_key')
    s3_secret_key = archive_config.get('s3_secret_key')
    s3_bucket = archive_config.get('s3_bucket')

    session = boto3.session.Session()
    s3_client = session.client(
        service_name='s3',
        endpoint_url=s3_endpoint,
        aws_access_key_id=s3_access_key,
        aws_secret_access_key=s3_secret_key
    )

    archive_urls = {}
    for root, dirs, files in os.walk(batch_dir):
        for file in files:
            local_path = os.path.join(root, file)
            s3_key = f"{os.path.basename(batch_dir)}/{file}"
            try:
                s3_client.upload_file(local_path, s3_bucket, s3_key)
                archive_url = f"{s3_endpoint}/{s3_bucket}/{s3_key}"
                archive_urls[file] = archive_url
                logger.info(f"Uploaded {file} to S3 at {archive_url}")
            except Exception as e:
                logger.error(f"Failed to upload {file} to S3: {e}")
    return archive_urls

def archive_batch_to_azure_blob(batch_dir, archive_config, logger):
    connection_string = archive_config.get('azure_connection_string')
    container_name = archive_config.get('azure_container_name')

    blob_service_client = BlobServiceClient.from_connection_string(connection_string)
    container_client = blob_service_client.get_container_client(container_name)

    archive_urls = {}
    for root, dirs, files in os.walk(batch_dir):
        for file in files:
            local_path = os.path.join(root, file)
            blob_name = f"{os.path.basename(batch_dir)}/{file}"
            try:
                with open(local_path, "rb") as data:
                    container_client.upload_blob(name=blob_name, data=data)
                archive_url = f"https://{blob_service_client.account_name}.blob.core.windows.net/{container_name}/{blob_name}"
                archive_urls[file] = archive_url
                logger.info(f"Uploaded {file} to Azure Blob at {archive_url}")
            except Exception as e:
                logger.error(f"Failed to upload {file} to Azure Blob: {e}")
    return archive_urls

def archive_batch_to_ftp(batch_dir, archive_config, logger):
    ftp_host = archive_config.get('ftp_host')
    ftp_port = archive_config.get('ftp_port', 21)
    ftp_username = archive_config.get('ftp_username')
    ftp_password = archive_config.get('ftp_password')
    ftp_directory = archive_config.get('ftp_directory', '/')

    archive_urls = {}
    try:
        ftp = ftplib.FTP()
        ftp.connect(ftp_host, ftp_port)
        ftp.login(ftp_username, ftp_password)
        ftp.cwd(ftp_directory)

        for root, dirs, files in os.walk(batch_dir):
            for file in files:
                local_path = os.path.join(root, file)
                with open(local_path, 'rb') as f:
                    ftp.storbinary(f'STOR {file}', f)
                archive_url = f"ftp://{ftp_host}{ftp_directory}/{file}"
                archive_urls[file] = archive_url
                logger.info(f"Uploaded {file} to FTP at {archive_url}")
        ftp.quit()
    except Exception as e:
        logger.error(f"Failed to upload files to FTP: {e}")
    return archive_urls

def archive_batch_to_sftp(batch_dir, archive_config, logger):
    sftp_host = archive_config.get('sftp_host')
    sftp_port = archive_config.get('sftp_port', 22)
    sftp_username = archive_config.get('sftp_username')
    sftp_password = archive_config.get('sftp_password')
    sftp_directory = archive_config.get('sftp_directory', '/')

    archive_urls = {}
    try:
        transport = paramiko.Transport((sftp_host, sftp_port))
        transport.connect(username=sftp_username, password=sftp_password)
        sftp = paramiko.SFTPClient.from_transport(transport)

        # Ensure the directory exists
        try:
            sftp.chdir(sftp_directory)
        except IOError:
            sftp.mkdir(sftp_directory)
            sftp.chdir(sftp_directory)

        for root, dirs, files in os.walk(batch_dir):
            for file in files:
                local_path = os.path.join(root, file)
                remote_path = f"{sftp_directory}/{os.path.basename(batch_dir)}/{file}"
                # Ensure remote directory exists
                try:
                    sftp.stat(os.path.dirname(remote_path))
                except IOError:
                    sftp.mkdir(os.path.dirname(remote_path))
                sftp.put(local_path, remote_path)
                archive_url = f"sftp://{sftp_host}{remote_path}"
                archive_urls[file] = archive_url
                logger.info(f"Uploaded {file} to SFTP at {archive_url}")
        sftp.close()
        transport.close()
    except Exception as e:
        logger.error(f"Failed to upload files to SFTP: {e}")
    return archive_urls
