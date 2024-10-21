from ultralytics import YOLO
import cv2
import os 

THRESHHOLD = 0.7

#this function crops the images from the original images based on the yolo predictions
def crop_images_from_yolo_predictions(model , folder_path):
    objects = []

    results = model.predict(source= folder_path,
                            save_crop=False, stream=True, save=False)  
    
    for r in results:
        for box in r.boxes:
            tlx, tly, drx, dry, conf, cls = box.data.cpu().numpy()[0]
            tlx = int(tlx) ; tly = int(tly) ; drx = int(drx) ; dry = int(dry)
            if conf < THRESHHOLD: 
                continue
            original_image = r.orig_img
            crop = original_image[tly:dry, tlx:drx, ...]
            objects.append(crop)
    return objects


#this function crops the images from the original images based on the yolo predictions 
# and saves them in the destination folder
def crop_images_from_yolo_predictions_and_save(model , folder_path , dest_folder):

    os.makedirs(dest_folder, exist_ok=True)
    results = model.predict(source= folder_path,
                            save_crop=False, stream=True, save=False)  
    
    for r in results:
        for box in r.boxes:
            tlx, tly, drx, dry, conf, cls = box.data.cpu().numpy()[0]
            tlx = int(tlx) ; tly = int(tly) ; drx = int(drx) ; dry = int(dry)
            if conf < THRESHHOLD: 
                continue
            original_image = r.orig_img #get the original image
            crop = original_image[tly:dry, tlx:drx, ...]
            parent_folder = r.path.split('\\')[-2]
            filename = os.path.basename(r.path)
            coords = "_".join([str(x) for x in [tlx, tly, drx, dry]]) 

            crop_save_name = f"{parent_folder}---{filename}---{coords}.jpg"
            cv2.imwrite(os.path.join(dest_folder, crop_save_name), crop)
            


if __name__ == "__main__":

    model = YOLO(r"path/to/model.pt")
    folder_path = r"path/to/folder"
    dest_folder = r"path/to/destination/folder"
    crop_images_from_yolo_predictions_and_save(model , folder_path , dest_folder)


