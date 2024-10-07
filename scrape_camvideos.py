import os
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import argparse

def download_image(image_url, save_path):
    response = requests.get(image_url, stream=True)
    if response.status_code == 200:
        with open(save_path, 'wb') as out_file:
            for chunk in response.iter_content(1024):
                out_file.write(chunk)
        print(f"Downloaded: {save_path}")
    else:
        print(f"Failed to download: {image_url}")

def process_page(page_url, image_dir, unique_entries, nickname_counts):
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

        # Increment the counter for the current nickname
        nickname_counts[nickname] += 1

        # Get image URL and modify it to remove ".th"
        img_tag = post.find('img')
        if img_tag:
            img_url = img_tag['src'].replace('.th', '')
            img_name = f"{nickname}_{gender}_{site}_{nickname_counts[nickname]}.jpg"
            save_path = os.path.join(image_dir, img_name)
            download_image(img_url, save_path)

def get_max_pages(soup):
    nav = soup.find('nav', class_='footer')
    if nav:
        pages = nav.find_all('a', href=True)
        last_page = max([int(a.text) for a in pages if a.text.isdigit()])
        return last_page
    return 1

def scrape_website(start_page, limit, image_dir):
    base_url = "http://camvideos.me"
    unique_entries = set()
    nickname_counts = {}

    os.makedirs(image_dir, exist_ok=True)

    for page_num in range(start_page, start_page + limit):
        page_url = f"{base_url}/?page={page_num}"
        print(f"Processing page {page_num}: {page_url}")
        response = requests.get(page_url)
        soup = BeautifulSoup(response.text, 'html.parser')
        
        process_page(page_url, image_dir, unique_entries, nickname_counts)
        
        max_pages = get_max_pages(soup)
        if page_num >= max_pages:
            print(f"Reached the last page: {max_pages}")
            break

    print("Scraping completed.")
    print("Unique entries found:")
    for entry in unique_entries:
        print(f"Nickname: {entry[0]}, Site: {entry[1]}, Gender: {entry[2]}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Scrape camvideos.me')
    parser.add_argument('-l', '--limit', type=int, default=1, help='Number of pages to process')
    parser.add_argument('-s', '--start', type=int, default=1, help='Starting page number')
    
    args = parser.parse_args()

    IMAGE_DIR = "images"
    scrape_website(start_page=args.start, limit=args.limit, image_dir=IMAGE_DIR)

