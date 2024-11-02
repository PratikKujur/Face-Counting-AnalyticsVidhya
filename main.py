
from ultralytics import YOLO
import shutil
import os
import torch
import math
import pandas as pd
from google.colab import files


def fine_tunning(YOLO_pretrained_path,yamal_path):
    model=YOLO("YOLO_pretrained_path")
    # Train the model
    train_results = model.train(
        data="yamal_path",  # path to dataset YAML
        epochs=5,  # number of training epochs
        imgsz=4586,  # training image size
        device=0,  # device to run on, i.e. device=0 or device=0,1,2,3 or device=cpu
    )
    
    shutil.make_archive("runs", 'zip', "runs")
    files.download("runs.zip")

def model_prediction(best_model_path,predict_path):
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    img_paths = [os.path.join(predict_path, x) for x in os.listdir(predict_path)]

    batch_size = 4

    model=YOLO(best_model_path)

    results_data=[]
    processed_count = 0
    total_images = len(img_paths)
    
    def process_in_batches(img_paths, batch_size):
        global processed_count
        num_batches = math.ceil(len(img_paths) / batch_size)

        with torch.no_grad():
            for batch_idx in range(num_batches):
                
                batch_imgs = img_paths[batch_idx * batch_size : (batch_idx + 1) * batch_size]

                
                results = model(batch_imgs, save=True, stream=True, batch=batch_size)

                
                for img_path, result in zip(batch_imgs, results):
                    
                    head_count = len(result.boxes) if result.boxes else 0

                    
                    img_name = os.path.basename(img_path)
                    results_data.append({"Name": img_name, "HeadCount": head_count})

                
                    processed_count += 1
                    print(f"Processed {processed_count}/{total_images} images")

                    #result.show()  # Display result on screen
                # result.save(filename=f"result_{img_name}")  
                torch.cuda.empty_cache()
    
def Data_preprocessing(csv_path,original_images_dir,new_images_dir,labels_dir,test_images,test_csv):
    

    os.makedirs(new_images_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)
    annotations = pd.read_csv(csv_path)
    grouped_annotations = annotations.groupby("Name")

    for image_name, group in grouped_annotations:
    
        image_path = os.path.join(original_images_dir, image_name)
        
        
        if not os.path.exists(image_path):
            print(f"Image {image_name} not found, skipping.")
            continue
        
        
        shutil.copy(image_path, os.path.join(new_images_dir, image_name))
        
        
        label_file_path = os.path.join(labels_dir, f"{os.path.splitext(image_name)[0]}.txt")
        with open(label_file_path, "w") as label_file:
            for _, row in group.iterrows():
            
                img_width, img_height = row["width"], row["height"]
                xmin, ymin, xmax, ymax = row["xmin"], row["ymin"], row["xmax"], row["ymax"]
                
                
                x_center = ((xmin + xmax) / 2) / img_width
                y_center = ((ymin + ymax) / 2) / img_height
                bbox_width = (xmax - xmin) / img_width
                bbox_height = (ymax - ymin) / img_height
                
                
                label_file.write(f"0 {x_center} {y_center} {bbox_width} {bbox_height}\n")

    print("Bounding box files created and images copied to new folder.")


    

    test_annotations=pd.read_csv(test_csv)
    test_annotations_grouped=test_annotations.groupby("Name")
    os.makedirs(test_images, exist_ok=True)

    for image_name, group in test_annotations_grouped:
 
        image_path = os.path.join(original_images_dir, image_name)
        
        
        if not os.path.exists(image_path):
            print(f"Image {image_name} not found, skipping.")
            continue
    
    
        shutil.copy(image_path, os.path.join(test_images, image_name))
    

if __name__=="main":

    """ ********* STEP 1- DATA PREPROCESSING ********* """

    csv_path = "train_HNzkrPW (1)/bbox_train.csv"
    original_images_dir = "train_HNzkrPW (1)/image_data"
    new_images_dir = "annotated_images/"
    labels_dir = "labels/"
    test_images="test_images/"
    test_csv='test_Rj9YEaI.csv'

    Data_preprocessing(csv_path,original_images_dir,new_images_dir,labels_dir,test_images,test_csv)



    """ ********* STEP 2- MODEL FINE TUNING ********* """ 

    YOLO_pretrained_path="yolo11n.pt"
    yamal_path="data.yaml"
    best_model_path=fine_tunning(YOLO_pretrained_path,yamal_path)

    """ ********* STEP 3- MODEL PREDICTION ********* """
    predict_path="test_images"
    model_prediction(best_model_path,predict_path)
