import requests
from bs4 import BeautifulSoup


def load_website(url):

    response = requests.get(url)

    soup = BeautifulSoup(response.text, "html.parser")

    paragraphs = soup.find_all("p")

    text = " ".join(p.get_text() for p in paragraphs)

    return text