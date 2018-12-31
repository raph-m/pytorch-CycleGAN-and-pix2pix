import requests
import zipfile


# python script to download a file from a google drive public link


def download_file_from_google_drive(id, destination):
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

    url = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(url, params={'id': id}, stream=True)
    token = get_confirm_token(response)

    if token:
        params = {'id': id, 'confirm': token}
        response = session.get(url, params=params, stream=True)

    save_response_content(response, destination)


if __name__ == "__main__":

    # share link of dataset:
    # "https://drive.google.com/file/d/1z1g6fLu-s8-zsfXiTuFgiC7hog0_Km1x/view?usp=sharing"
    # new link:
    # https://drive.google.com/file/d/1gO7iejmS0wqfQN31iUrhjV_Cb7UD73Zh/view?usp=sharing
    # new new link:
    # https://drive.google.com/file/d/1eb6EltjSy157hmaYN4bH19-Wq89j9Elt/view?usp=sharing

    destination = "gdrive"
    file_id = "1eb6EltjSy157hmaYN4bH19-Wq89j9Elt"

    download_file_from_google_drive(file_id, destination + ".zip")

    with zipfile.ZipFile(destination + ".zip", 'r') as zip_ref:
        zip_ref.extractall("")


















