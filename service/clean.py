# file: clean.py
# directory: .
import os
import psycopg2
from psycopg2 import sql

# Database connection details
DB_HOST = os.environ.get('DB_HOST')
DB_NAME = os.environ.get('DB_NAME')
DB_USER = os.environ.get('DB_USER')
DB_PASSWORD = os.environ.get('DB_PASSWORD')

# Azure Service Bus connection details
SERVICE_BUS_CONNECTION_STR = os.environ.get('SERVICE_BUS_CONNECTION_STR')
QUEUE_NAME = os.environ.get('QUEUE_NAME')

def drop_specific_tables():
    tables_to_drop = [
        'parser_run_stats',
        'parser_runs',
        'image_embeddings',
        'host_logs',
        'batch_logs',
        'sites',
        'base_image_urls',
        'images',
        'batches',
        'archived_images',
        'batch_images',
        'checkpoints'
    ]
    conn = psycopg2.connect(
        host=DB_HOST,
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD,
        sslmode='require'
    )
    conn.autocommit = True
    cur = conn.cursor()

    # Disable foreign key constraints
    cur.execute("SET session_replication_role = 'replica';")

    # Drop specific tables
    for table_name in tables_to_drop:
        print(f"Dropping table {table_name}...")
        cur.execute(sql.SQL("DROP TABLE IF EXISTS {} CASCADE;").format(sql.Identifier(table_name)))

    # Re-enable foreign key constraints
    cur.execute("SET session_replication_role = 'origin';")

    cur.close()
    conn.close()
    print("Specified tables have been dropped.")

def clear_queue():
    admin_client = ServiceBusAdministrationClient.from_connection_string(SERVICE_BUS_CONNECTION_STR)
    print(f"Deleting queue '{QUEUE_NAME}'...")
    try:
        admin_client.delete_queue(QUEUE_NAME)
        print(f"Queue '{QUEUE_NAME}' has been deleted.")
        # Recreate the queue with the same name
        admin_client.create_queue(QUEUE_NAME)
        print(f"Queue '{QUEUE_NAME}' has been recreated.")
    except Exception as e:
        print(f"An error occurred while resetting the queue: {e}")

if __name__ == "__main__":
#    print("⚠ Warning: This script will delete specific tables from your database and reset your Azure Service Bus queue.")
#    confirm = input("Are you sure you want to proceed? This action cannot be undone. (yes/no): ")
#    if confirm.lower() == 'yes':
     print("Dropping specified tables from the database...")
     drop_specific_tables()
                           
