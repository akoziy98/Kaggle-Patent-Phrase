import bs4
import lxml
import os
from bs4 import BeautifulSoup as bs
from tqdm import tqdm


def process_context_id(filename):
    try:
        splt1 = filename.split("-")
        splt2 = splt1[2].split(".")
        id_of_xml = splt2[0]
        if not id_of_xml.isalpha():
            res = id_of_xml[:-1]
            return res
        else:
            return None
    except:
        return None

def load_single_xml(path, filename):
    single_path = os.path.join(path, filename)
    with open(single_path, "r") as file:
        lines = file.readlines()

    texts_list = []
    for single_line in lines:
        bs_small_content = bs(single_line)
        res = bs_small_content.find_all("text")
        cur_str = []
        for el in res:
            cur_str.append(el.text)
        s = " ".join(cur_str)
        if cur_str != []:
            texts_list.append(s)

    context_id = process_context_id(filename)

    return context_id, texts_list


if __name__ == "__main__":
    xml_path = "xml_data"
    xml_name = "cpc-scheme-A01B.xml"
    path = os.path.join(xml_path, xml_name)

    processed_contexts = {}
    for filename in tqdm(os.listdir(xml_path)):
        context_id, texts_list = load_single_xml(xml_path, filename)
        if not context_id is None:
            if processed_contexts.get(context_id) is None:
                processed_contexts[context_id] = texts_list
            else:
                processed_contexts[context_id].extend(texts_list)