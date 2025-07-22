import os
import re
import tempfile
import uuid
import zipfile

def parse_and_create_files(model_output):
    temp_dir = tempfile.mkdtemp(prefix="springboot_")
    current_file_path = None
    buffer = []

    for line in model_output.splitlines():
        file_match = re.match(r"File:\s*(.+)", line)
        if file_match:
            if current_file_path and buffer:
                with open(current_file_path, "w", encoding="utf-8") as f:
                    f.write('\n'.join(buffer))
                buffer = []
            relative_path = file_match.group(1).strip().replace("\\", "/")
            relative_path = re.sub(r"[^a-zA-Z0-9_/.\-]", "", relative_path)
            current_file_path = os.path.join(temp_dir, relative_path)
            os.makedirs(os.path.dirname(current_file_path), exist_ok=True)
        elif current_file_path:
            buffer.append(line)

    if current_file_path and buffer:
        with open(current_file_path, "w", encoding="utf-8") as f:
            f.write('\n'.join(buffer))

    return temp_dir

def zip_folder(folder_path):
    zip_name = f"springboot_app_{uuid.uuid4().hex[:8]}.zip"
    zip_path = os.path.join(tempfile.gettempdir(), zip_name)
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        for root, _, files in os.walk(folder_path):
            for file in files:
                full_path = os.path.join(root, file)
                arcname = os.path.relpath(full_path, folder_path)
                zipf.write(full_path, arcname)
    return zip_path
