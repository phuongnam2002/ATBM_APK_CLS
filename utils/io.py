import os
import json
from androguard.core.apk import APK
from androguard.core.dex import DEX
from typing import Dict, Text, List


def load_apk(file_path: str):
    apk = APK(file_path)

    # Extract static data
    package = apk.get_package()
    receivers = apk.get_receivers()
    providers = apk.get_providers()
    activities = apk.get_activities()
    permissions = apk.get_permissions()

    # Extract code
    codes = []

    classes_dex = apk.get_dex()

    dvm = DEX(classes_dex)

    classes = dvm.get_classes()

    for x in classes:
        methods = dvm.get_methods_class(x)
        codes.extend(methods)

    data = []

    data.extend(package)
    data.extend(activities)
    data.extend(permissions)
    data.extend(providers)
    data.extend(receivers)
    data.extend(codes)

    return " ".join(data)


def load_file(file_path: str):
    with open(file_path, 'r') as f:
        data = f.readlines()

    data = [x.strip() for x in data]
    return data


def write_file(file_path: str, data):
    with open(file_path, 'w+') as f:
        for attribute in data:
            f.writelines(attribute + '\n')


def find_folders(folder_path: str):
    sub_folders = []
    for root, dirs, files in os.walk(folder_path):
        for dir_name in dirs:
            sub_folder_path = os.path.join(root, dir_name)
            sub_folders.append(sub_folder_path)

    return sub_folders


def load_jsonl(file_path: Text) -> List[Dict]:
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))
    return data


def write_jsonl(file_path, data):
    with open(file_path, 'w') as pf:
        for item in data:
            obj = json.dumps(item, ensure_ascii=False)
            pf.write(obj + '\n')
