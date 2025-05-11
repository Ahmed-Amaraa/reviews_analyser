LABEL_MAP = {
    '__label__1': 'negative',
    '__label__2': 'positive',
    '__label__3': 'neutral'
}

from textblob import TextBlob
import concurrent.futures


def get_label_and_text(text):
    polarity = TextBlob(text).sentiment.polarity
    if polarity > 0.3:
        label = '__label__2'  # positive
    elif polarity < -0.3:
        label = '__label__1'  # negative
    else:
        label = '__label__3'  # neutral
    return label, text.strip()


def process_texts(_texts):
    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = list(executor.map(get_label_and_text, _texts))
    return results


if __name__ == '__main__':
    _texts = []

    with open('train.ft.txt/train.ft.txt', 'r', encoding='utf-8') as f:
        for line in f:
            # Remove the original label (if present)
            if line.startswith('__label__'):
                line = ' '.join(line.split()[1:])
            _texts.append(line)

    with open('test.ft.txt/test.ft.txt', 'r', encoding='utf-8') as f:
        for line in f:
            # Remove the original label (if present)
            if line.startswith('__label__'):
                line = ' '.join(line.split()[1:])
            _texts.append(line)

    # Process texts in parallel
    results = process_texts(_texts)

    labels = []
    texts = []

    for label, text in results:
        labels.append(label)
        texts.append(text)

    # Write the processed data to a file
    with open('dataset.txt', 'w', encoding='utf-8') as f:
        for label, text in zip(labels, texts):
            f.write(f"{label} {text}\n")
