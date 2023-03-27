import cv2
import os

def pad_images(directory_path, padded_size=(640, 640), border_color=(0, 0, 0)):
    # set the paths to the images and labels directories
    images_dir = os.path.join(directory_path, "images")
    labels_dir = os.path.join(directory_path, "labels")

    # loop over all the image files in the images directory
    for filename in os.listdir(images_dir):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            # set the paths to the image and label files
            image_path = os.path.join(images_dir, filename)
            label_path = os.path.join(labels_dir, os.path.splitext(filename)[0] + ".txt")

            # load the image
            image = cv2.imread(image_path)

            # get the current size of the image
            current_size = image.shape[:2]

            # calculate the amount of padding needed to make the image 640x640
            padding_x = max(0, padded_size[0] - current_size[1])
            padding_y = max(0, padded_size[1] - current_size[0])

            # pad the image
            padded_image = cv2.copyMakeBorder(image, padding_y // 2, padding_y - (padding_y // 2), padding_x // 2, padding_x - (padding_x // 2), cv2.BORDER_CONSTANT, value=border_color)

            # save the padded image
            cv2.imwrite(os.path.join(images_dir, os.path.splitext(filename)[0] + os.path.splitext(filename)[1]), padded_image)

            # update the label file
            if os.path.exists(label_path):
                with open(label_path, "r") as f:
                    label_info = f.read()

                # parse the label information
                label_parts = label_info.split(" ")
                className = label_parts[0]
                x_norm = float(label_parts[1])
                y_norm = float(label_parts[2])
                w_norm = float(label_parts[3])
                h_norm = float(label_parts[4])

                # calculate the new bounding box coordinates
                new_x_norm = (x_norm * current_size[1] + padding_x // 2) / padded_size[0]
                new_y_norm = (y_norm * current_size[0] + padding_y // 2) / padded_size[1]
                new_w_norm = min(1, w_norm * current_size[1] / padded_size[0])
                new_h_norm = min(1, h_norm * current_size[0] / padded_size[1])

                # write the updated label information to the file
                with open(os.path.join(labels_dir, os.path.splitext(filename)[0] + ".txt"), "w") as f:
                    f.write(f"{className} {new_x_norm} {new_y_norm} {new_w_norm} {new_h_norm}")
            else:
                print(f"Label file for {filename} does not exist.")

if __name__ == "__main__":

    directory_path = "Terraria4.1/train/"
    pad_images(directory_path, padded_size=(1920,1080))
    directory_path = "Terraria4.1/valid/"
    pad_images(directory_path, padded_size=(1920,1080))
    directory_path = "Terraria4.1/test/"
    pad_images(directory_path, padded_size=(1920,1080))
