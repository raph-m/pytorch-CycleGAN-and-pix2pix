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

    # rar link:
    # https://drive.google.com/file/d/1sw5-5Cqy2k2MhxXm5NQxTl_LykAq9ggr/view?usp=sharing

    destination = "gdrive"
    file_id = "1sw5-5Cqy2k2MhxXm5NQxTl_LykAq9ggr"

    download_file_from_google_drive(file_id, destination + ".zip")

    # with zipfile.ZipFile(destination + ".zip", 'r') as zip_ref:
    #   zip_ref.extractall("")



















