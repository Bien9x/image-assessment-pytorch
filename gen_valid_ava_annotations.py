import json
import os
def process(ava_anno_folder,filename,img_dir,save_folder):
    with open(os.path.join(ava_anno_folder,filename),'r') as f:
        data = json.load(f)

    data = list(filter(lambda x: os.path.exists(os.path.join(img_dir, x['image_id'] + ".jpg")),data))

    with open(os.path.join(save_folder,filename),'w') as f:
        json.dump(data, f)
    print('Done',filename,len(data))

if __name__ == '__main__':
    img_dir = r"D:\dataset\AVA_dataset\images\images"
    save_folder = "data/AVA_valid"
    ava_folder = "data/AVA/"
    #process(ava_folder,"ava_labels_test.json",img_dir,save_folder)
    process(ava_folder,"ava_labels_train.json",img_dir,save_folder)