def download_hf_model_to_dir(model_name, save_dir):
    """
    下载huggingface上的文件到本地文件
    :param model_name: 指定模型文件名，huggingface上的名字 如 bert-base-uncased
    :param save_dir:  存到本地的文件夹路径。
    :return: 
    """
    from transformers import AutoModel, AutoTokenizer
    
    model = AutoModel.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    print(f"{model_name} 已经下载完毕，已保存至{save_dir}")

download_hf_model_to_dir('lmsys/vicuna-7b-v1.3', '/root/autodl-tmp/zhc/llm-attacks/data/vicuna')