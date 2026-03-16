import os
def execute_plan(plan_code:str,target_folder:str,file_name:str="new_feature.py"):
    os.makedirs(target_folder, exist_ok=True)
    file_path = os.path.join(target_folder, file_name)
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(plan_code)
    return file_path