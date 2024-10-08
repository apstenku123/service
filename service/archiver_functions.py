# file: archiver_functions.py
# directory: .

import os
import logging

import boto3
from azure.storage.blob import BlobServiceClient
import ftplib
import paramiko


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
