import os
import zipfile
import shutil

import requests


def download_file_from_google_drive(id, destination):
    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params={'id': id}, stream=True)
    token = get_confirm_token(response)

    if token:
        params = {'id': id, 'confirm': token}
        response = session.get(URL, params=params, stream=True)

    save_response_content(response, destination)


def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value

    return None


def save_response_content(response, destination):
    CHUNK_SIZE = 32768

    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk:  # filter out keep-alive new chunks
                f.write(chunk)


def download(file_id, out_dir: str, unpack=True, zip_name='') -> None:
    download_file_from_google_drive(
        file_id,
        './temp_do_not_remove.zip'
    )

    if unpack:
        with zipfile.ZipFile('./temp_do_not_remove.zip', 'r') as zip_ref:
            zip_ref.extractall(out_dir)

        os.remove('./temp_do_not_remove.zip')
    else:
        os.rename('./temp_do_not_remove.zip', zip_name)
        if out_dir.endswith('/'):
            out_dir = out_dir[0:-1]
        shutil.move('./'+zip_name, out_dir)
