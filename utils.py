import json
import matplotlib.pyplot as plt

def read_json_content(file):
    with open(file) as json_file:
        return json.load(json_file)


def convert_string_to_json(string):
    return json.loads(string)


def write_to_txt(content, file):
    with open(file, 'w') as fp:
        fp.write(content)
    fp.close()


def read_text(file):
    with open(file, 'r') as f:
        return f.readlines()


def write_to_file(path, filename, content):
    with open(path + '/' + filename, 'w') as fp:
        if isinstance(content, str):
            fp.write(content)
        elif isinstance(content, list):
            for item in content:
                fp.write(item)
    fp.close()


def show_plots(dataset, color, group_by):
    dataset.groupby(group_by).size().plot(kind='bar', colormap=color).set_ylabel('Count')
    plt.show()