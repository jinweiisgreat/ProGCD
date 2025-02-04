from tqdm import tqdm
from loguru import logger
import clip
import numpy as np
import pickle
import torch


def encode_descrip_txt(file_name):
    model, preprocess = clip.load('/home/Lmodel/ViT-L-14.pt') 
    model.cuda().eval()

    with open(file_name,'rb')as f:
        text_data = pickle.load(f)
    f.close()
  
    result = None
    first = True
    
    for img_descrip in tqdm(text_data):
        
        text_tokens = clip.tokenize(img_descrip, context_length=77, truncate=True).cuda()
        #[batch_size,77]
        
        with torch.no_grad():
            text_features = model.encode_text(text_tokens).float()
            text_features /= text_features.norm(dim = -1, keepdim = True)

        if first == True:
            result = text_features.unsqueeze(0)
            first = False
        else:
            result = torch.cat((result.cpu(),text_features.unsqueeze(0).cpu()), dim=0)

    print(f'result.shape:{result.shape}')
    
    return result
    
        
@logger.catch
def main():    

    dataset_name = 'cifar100'
    
    file_name = f"./pkl/{dataset_name}_a_photo_of_label.pkl"

    result = encode_descrip_txt(file_name)

    torch.save(result,f"./pth/{dataset_name}_a_photo_of_label.pth")

if __name__ == "__main__":
    main()

    

    