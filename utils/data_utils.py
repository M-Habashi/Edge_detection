import os
import requests
from boxsdk import OAuth2, Client

# Box API credentials
CLIENT_ID = "6ndnryupe5qjah7y6tv0ydt35pz55ouw"
CLIENT_SECRET = "E6dYlLzVPu1pKzOJXNC7jsEoiq4OEvV5"
ACCESS_TOKEN = "94yKk2rrIfVTuv4gMVA0elY6sSGTTMeh"

# Local data folder
DATA_FOLDER = "data"

def list_folders_files(folder_id):
    client = Client(OAuth2(None, None, access_token=ACCESS_TOKEN))
    folder = client.folder(folder_id).get()
    print("Folders:")
    for item in folder.get_items(limit=100, offset=0):
        if item.type == 'folder':
            print(f"Folder: {item.name}")
        elif item.type == 'file':
            print(f"File: {item.name}")

def create_directory(path):
    if not os.path.exists(path):
        os.makedirs(path)
        return 1

    return 0

def total_contents_count(folder_id, client=None, count=None):
    client = Client(OAuth2(None, None, access_token=ACCESS_TOKEN)) if client is None else client
    folder = client.folder(folder_id).get()
    items = folder.get_items(limit=1000, offset=0)
    count = 0 if count is None else count

    for item in items:
        if item.type == 'file':
            count += 1

        elif item.type == 'folder':
            count += total_contents_count(item.id, client)
    return count

def download_folder_contents(folder_id, destination_path, count=None, client=None, total=None):
    if create_directory(destination_path)==0:
        print("Data is downloaded")
        return

    client = Client(OAuth2(None, None, access_token=ACCESS_TOKEN)) if client is None else client
    folder = client.folder(folder_id).get()
    items = folder.get_items(limit=1000, offset=0)
    count = 0 if count is None else count
    total = total_contents_count(folder_id) if total is None else total
    for item in items:
        if item.type == 'file':
            file_path = os.path.join(destination_path, item.name)
            with open(file_path, 'wb') as file:
                file.write(item.content())
                count += 1
                print(f"\rDownloading ---[{count}/{total}]---", flush=True, end="")

        elif item.type == 'folder':
            subfolder_path = os.path.join(destination_path, item.name)
            count += download_folder_contents(item.id, subfolder_path, count=count, total=total, client=client)

    if count == total:
        print('\n')
    return count



