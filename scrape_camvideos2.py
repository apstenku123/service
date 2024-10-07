import os
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import argparse
import threading
from queue import Queue

def download_image(image_url, save_path):
    response = requests.get(image_url, stream=True)
    if response.status_code == 200:
        with open(save_path, 'wb') as out_file:
            for chunk in response.iter_content(1024):
                out_file.write(chunk)
        print(f"Downloaded: {save_path}")
    else:
        print(f"Failed to download: {image_url}")

def worker(queue, end_event):
    while not end_event.is_set() or not queue.empty():
        try:
            task = queue.get(timeout=1)  # Get task with a timeout to check end_event
            image_url, save_path = task
            download_image(image_url, save_path)
        except Queue.Empty:
            continue
        finally:
            queue.task_done()

def process_page(page_url, image_dir, unique_entries, nickname_counts, output_file, task_queue):
    response = requests.get(page_url)
    soup = BeautifulSoup(response.text, 'html.parser')

    posts = soup.find_all('div', class_='post-container')
    for post in posts:
        # Extract nickname, site, and gender
        post_name = post.find('a', class_='post-name').text.strip()
        details = post_name.split(' ')
        nickname = details[0]
        site = details[-2]
        gender = details[-1]

        # Create unique entry identifier
        unique_entry = (nickname, site, gender)
        if unique_entry not in unique_entries:
            unique_entries.add(unique_entry)
            nickname_counts[nickname] = 0

            # Write the unique entry to the output file
            output_file.write(f"{nickname}, {site}, {gender}\n")

        # Increment the counter for the current nickname
        nickname_counts[nickname] += 1

        # Get image URL and modify it to remove ".th"
        img_tag = post.find('img')
        if img_tag:
            img_url = img_tag['src'].replace('.th', '')
            img_name = f"{nickname}_{gender}_{site}_{nickname_counts[nickname]}.jpg"
            save_path = os.path.join(image_dir, img_name)
            
            # Add download task to the queue
            task_queue.put((img_url, save_path))

def get_max_pages(soup):
    nav = soup.find('nav', class_='footer')
    if nav:
        pages = nav.find_all('a', href=True)
        last_page = max([int(a.text) for a in pages if a.text.isdigit()])
        return last_page
    return 1

def scrape_website(start_page, limit, image_dir, output_filename, num_threads):
    base_url = "http://camvideos.me"
    unique_entries = set()
    nickname_counts = {}

    os.makedirs(image_dir, exist_ok=True)

    task_queue = Queue()
    end_event = threading.Event()

    # Start worker threads for downloading images
    workers = []
    for _ in range(num_threads):
        worker_thread = threading.Thread(target=worker, args=(task_queue, end_event))
        worker_thread.start()
        workers.append(worker_thread)

    end_page = start_page + limit - 1

    # Open the output file
    with open(output_filename, 'w') as output_file:
        for page_num in range(start_page, start_page + limit):
            page_url = f"{base_url}/?page={page_num}"
            print(f"Processing page {page_num}: {page_url}")
            response = requests.get(page_url)
            soup = BeautifulSoup(response.text, 'html.parser')

            process_page(page_url, image_dir, unique_entries, nickname_counts, output_file, task_queue)

            max_pages = get_max_pages(soup)
            if page_num >= max_pages:
                print(f"Reached the last page: {max_pages}")
                break

    # Wait until all tasks are done
    task_queue.join()

    # Signal workers to exit
    end_event.set()

    # Wait for all workers to finish
    for worker_thread in workers:
        worker_thread.join()

    print(f"Scraping completed. Results saved in {output_filename}")
    print("Unique entries found:")
    for entry in unique_entries:
        print(f"Nickname: {entry[0]}, Site: {entry[1]}, Gender: {entry[2]}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Scrape camvideos.me')
    parser.add_argument('-l', '--limit', type=int, default=1, help='Number of pages to process')
    parser.add_argument('-s', '--start', type=int, default=1, help='Starting page number')
    parser.add_argument('-f', '--file', type=str, help='Output file for unique entries', default=None)
    parser.add_argument('-t', '--threads', type=int, help='Number of threads for downloading', default=10)

    args = parser.parse_args()

    start_page = args.start
    end_page = start_page + args.limit - 1
    output_filename = args.file if args.file else f"nick_report_page_{start_page}_page_{end_page}.txt"

    IMAGE_DIR = "images"
    scrape_website(start_page=start_page, limit=args.limit, image_dir=IMAGE_DIR, output_filename=output_filename, num_threads=args.threads)

