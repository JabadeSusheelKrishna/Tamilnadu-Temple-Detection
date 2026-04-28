import os
import requests

def download_file(url, folder):
    if not os.path.exists(folder):
        os.makedirs(folder)
    local_filename = os.path.join(folder, url.split('/')[-1])
    if os.path.exists(local_filename):
        print(f"File {local_filename} already exists.")
        return local_filename
    print(f"Downloading {url}...")
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'}
    with requests.get(url, stream=True, headers=headers) as r:
        r.raise_for_status()
        with open(local_filename, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
    return local_filename

samples = [
    "https://upload.wikimedia.org/wikipedia/commons/4/4e/Meenakshi_Amman_Temple_Madurai.jpg",
    "https://upload.wikimedia.org/wikipedia/commons/4/42/Brihadisvara_Temple_Thanjavur_2019.jpg",
    "https://upload.wikimedia.org/wikipedia/commons/b/b3/Shore_Temple_Mahabalipuram.jpg",
    "https://upload.wikimedia.org/wikipedia/commons/6/67/Kapaleeshwarar_Temple_Mylapore_Chennai.jpg"
]

if __name__ == "__main__":
    for s in samples:
        try:
            download_file(s, "sample_images")
        except Exception as e:
            print(f"Failed to download {s}: {e}")
