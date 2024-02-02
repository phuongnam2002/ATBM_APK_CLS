import string


def preprocessing(text: str):
    text = text.lower()
    text = text.strip(string.punctuation)

    return text
