structure = {
    "2_Understanding_Large_Language_Models": [
        "2.1_What_is_a_LLM",
        "2.2_Applications_of_LLMs",
        "2.3_Stages_of_building_and_using_LLMs",
        "2.4_Utilizing_large_datasets",
        "2.5_A_closer_look_at_the_GPT_architecture"
    ],
    "3_Working_with_Text_Data": [
        "3.1_Understanding_word_embeddings",
        "3.2_Tokenizing_text",
        "3.3_Converting_tokens_into_token_IDs",
        "3.4_Adding_special_context_tokens",
        "3.5_Byte_pair_encoding"
    ],
    "4_Coding_Attention_Mechanisms": [
        "4.1_Introduction_to_attention_mechanisms",
        "4.2_Self-attention_mechanisms",
        "4.3_Multi-head_attention_mechanisms",
        "4.4_Encoder_and_decoder_architectures",
        "4.5_Implementing_self-attention_mechanisms"
    ],
    "5_Implementing_a_GPT_model_from_Scratch_To_Generate_Text": [
        "5.1_Basics_of_GPT_model",
        "5.2_Implementing_GPT_model",
        "5.3_Training_GPT_model",
        "5.4_Applications_of_GPT_model"
    ],
    "6_Pretraining_on_Unlabeled_Data": [
        "6.1_Concept_of_pretraining",
        "6.2_Data_preparation",
        "6.3_Pretraining_process",
        "6.4_Evaluating_pretrained_model"
    ],
}


import os
from typing import Dict, Any

def create_directories_and_files(
        base_path: str, 
        structure: Dict[str, Any], 
        readme_file, 
        parent_path: str = "", 
        level: int = 1
    ):
    heading = "#" * level

    for key, value in structure.items():
        current_path = os.path.join(base_path, key.replace(" ", "_").replace("/", "_").replace("-", "_"))

        # 创建目录
        os.makedirs(current_path, exist_ok=True)

        # 在README中添加章节标题
        if parent_path:
            readme_file.write(f"{heading} {parent_path}/{key}\n\n")
        else:
            readme_file.write(f"{heading} {key}\n\n")

        # 递归调用创建子目录和文件
        if isinstance(value, dict) and value:
            create_directories_and_files(
                current_path, 
                value, 
                readme_file, 
                parent_path + "/" + key if parent_path else key, 
                level + 1
            )
        elif isinstance(value, list) and value:
            for idx, item in enumerate(value):
                if isinstance(item, dict) and item:
                    create_directories_and_files(
                        current_path, 
                        item, 
                        readme_file, 
                        parent_path + "/" + key if parent_path else key, 
                        level + 1
                    )
                else:
                    item = f"{idx:02d}_{item}"
                    file_name = item.replace(" ", "_").replace("/", "_").replace("-", "_") + ".py"
                    file_path = os.path.join(current_path, file_name)
                    with open(file_path, 'w', encoding='utf-8') as file:
                        file.write(f"# {item}\n\n")
                        file.write(f'"""\nLecture: {parent_path}/{key}\nContent: {item}\n"""\n\n')

                    # 在README中添加文件链接
                    item_clean = item.replace(" ", "_").replace("/", "_").replace("-", "_")
                    parent_clean = parent_path.replace(" ", "_").replace("/", "_").replace("-", "_")
                    key_clean = key.replace(" ", "_").replace("/", "_").replace("-", "_")
                    readme_file.write(f"- [{item}](./{parent_clean}/{key_clean}/{item_clean}.py)\n")
                    
                    
                    file_name = item.replace(" ", "_").replace("/", "_").replace("-", "_") + ".md"
                    file_path = os.path.join(current_path, file_name)
                    with open(file_path, 'w', encoding='utf-8') as file:
                        file.write(f"# {item}\n\n")
                        file.write(f'"""\nLecture: {parent_path}/{key}\nContent: {item}\n"""\n\n')

                    # 在README中添加文件链接
                    item_clean = item.replace(" ", "_").replace("/", "_").replace("-", "_")
                    parent_clean = parent_path.replace(" ", "_").replace("/", "_").replace("-", "_")
                    key_clean = key.replace(" ", "_").replace("/", "_").replace("-", "_")
                    readme_file.write(f"- [{item}](./{parent_clean}/{key_clean}/{item_clean}.md)\n")
                    
        else:
            # 创建文件并写入初始内容
            file_name = key.replace(" ", "_").replace("/", "_").replace("-", "_") + ".py"
            file_path = os.path.join(current_path, file_name)
            with open(file_path, 'w', encoding='utf-8') as file:
                file.write(f"# {key}\n\n")
                file.write(f'"""\nLecture: {parent_path}/{key}\nContent: {key}\n"""\n\n')

            # 在README中添加文件链接
            parent_clean = parent_path.replace(" ", "_").replace("/", "_").replace("-", "_")
            key_clean = key.replace(" ", "_").replace("/", "_").replace("-", "_")
            readme_file.write(f"- [{key}](./{parent_clean}/{key_clean}/{file_name})\n")
            
            
            file_name = key.replace(" ", "_").replace("/", "_").replace("-", "_") + ".md"
            file_path = os.path.join(current_path, file_name)
            with open(file_path, 'w', encoding='utf-8') as file:
                file.write(f"# {key}\n\n")
                file.write(f'"""\nLecture: {parent_path}/{key}\nContent: {key}\n"""\n\n')

            # 在README中添加文件链接
            parent_clean = parent_path.replace(" ", "_").replace("/", "_").replace("-", "_")
            key_clean = key.replace(" ", "_").replace("/", "_").replace("-", "_")
            readme_file.write(f"- [{key}](./{parent_clean}/{key_clean}/{file_name})\n")
            

        # 添加空行以分隔不同的章节
        readme_file.write("\n")

def main():
    root_dir = './'
    # 创建根目录
    os.makedirs(root_dir, exist_ok=True)

    # 创建 README.md 文件
    with open(os.path.join(root_dir, "README.md"), 'w', encoding='utf-8') as readme_file:
        readme_file.write("# Build Large Language Model\n\n")
        readme_file.write("这是一个关于Build Large Language Model的目录结构。\n\n")
        create_directories_and_files(root_dir, structure, readme_file)

    print("目录和文件结构已生成，并创建 README.md 文件。")

if __name__ == "__main__":
    main()