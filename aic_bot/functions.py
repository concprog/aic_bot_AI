import os
from time import strftime, time
from glob import glob


from . import models
from . import pipelines

timestr = lambda: strftime("%Y%m%d-%H%M%S")
DATA_PATH = "data/user_doc"


# utilities

def get_file_path(file_name: str):
    file_path: str = os.path.abspath(file_name)  # Join the directory and file name
    return file_path


def get_flie_path_from_name(file_name: str):
    files = glob(os.path.join(DATA_PATH, "**", file_name), recursive=True)
    return get_file_path(files[0])



def parse_code_from_md(data: str):
    ls1 = data.split("```")
    code: list[str] = []
    ct = 1
    for i in ls1:
        if ct % 2 != 1 and not (i.isspace()) and i != "":
            code.append(i)

        ct += 1

    return code

def get_content_from_messages(messages: list[models.Message]):
    messages: list[str]  = list(map(lambda x: x.author+ ':'+ x.content,messages))
    return messages

# AI Functions

def respond_to_query(message: models.Message, context: list[models.Message]):
    if context is not None:
        context = get_content_from_messages(context)
    query = message.content
    # implement rag response pipeline with context
    # (query, context) -> (response, source)
    # channel/role based data filtering?
    response = query
    return response

def index_data(data: models.Conversation):
    # calls index pipeline
    pass

def index_file(file_path: str):
    # use unstructured to load file
    # and then pass content to index pipline
    pass

def summarize(messages: list[models.Message]):
    data = get_content_from_messages(messages)
    data = "\n".join(data)
    # implement summarization pipeline
    

