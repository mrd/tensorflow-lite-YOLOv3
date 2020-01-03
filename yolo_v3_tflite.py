import time
import numpy as np
import sys
import os
import tensorflow as tf
from PIL import Image, ImageDraw

# see test() below for example

class YoloV3TFLite:
    def __init__(self, model_path, coco_names_path):
        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()

        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        if self.input_details[0]['dtype'] == np.float32:
            self.floating_model = True

        self.class_names = self.load_coco_names(coco_names_path)
        self.height = self.input_details[0]['shape'][1]
        self.width = self.input_details[0]['shape'][2]

    def handle_predictions(self, predictions, confidence=0.6, iou_threshold=0.5):
        boxes = predictions[:, :, :4]
        box_confidences = np.expand_dims(predictions[:, :, 4], -1)
        box_class_probs = predictions[:, :, 5:]

        box_scores = box_confidences * box_class_probs
        box_classes = np.argmax(box_scores, axis=-1)
        box_class_scores = np.max(box_scores, axis=-1)
        pos = np.where(box_class_scores >= confidence)

        boxes = boxes[pos]
        classes = box_classes[pos]
        scores = box_class_scores[pos]

        n_boxes, n_classes, n_scores = self.nms_boxes(boxes, classes, scores, iou_threshold)
        if n_boxes:
            boxes = np.concatenate(n_boxes)
            classes = np.concatenate(n_classes)
            scores = np.concatenate(n_scores)

            return boxes, classes, scores

        else:
            return None, None, None


    def nms_boxes(self, boxes, classes, scores, iou_threshold):
        nboxes, nclasses, nscores = [], [], []
        for c in set(classes):
            inds = np.where(classes == c)
            b = boxes[inds]
            c = classes[inds]
            s = scores[inds]

            x = b[:, 0]
            y = b[:, 1]
            w = b[:, 2]
            h = b[:, 3]

            areas = w * h
            order = s.argsort()[::-1]

            keep = []
            while order.size > 0:
                i = order[0]
                keep.append(i)

                xx1 = np.maximum(x[i], x[order[1:]])
                yy1 = np.maximum(y[i], y[order[1:]])
                xx2 = np.minimum(x[i] + w[i], x[order[1:]] + w[order[1:]])
                yy2 = np.minimum(y[i] + h[i], y[order[1:]] + h[order[1:]])

                w1 = np.maximum(0.0, xx2 - xx1 + 1)
                h1 = np.maximum(0.0, yy2 - yy1 + 1)

                inter = w1 * h1
                ovr = inter / (areas[i] + areas[order[1:]] - inter)
                inds = np.where(ovr <= iou_threshold)[0]
                order = order[inds + 1]

            keep = np.array(keep)

            nboxes.append(b[keep])
            nclasses.append(c[keep])
            nscores.append(s[keep])
        return nboxes, nclasses, nscores

    def load_coco_names(self, file_name):
        names = {}
        with open(file_name) as f:
            for id, name in enumerate(f):
                names[id] = name
        return names

    def letter_box_image(self, image: Image.Image, output_height: int, output_width: int, fill_value)-> np.ndarray:
        height_ratio = float(output_height)/image.size[1]
        width_ratio = float(output_width)/image.size[0]
        fit_ratio = min(width_ratio, height_ratio)
        fit_height = int(image.size[1] * fit_ratio)
        fit_width = int(image.size[0] * fit_ratio)
        fit_image = np.asarray(image.resize((fit_width, fit_height), resample=Image.BILINEAR))

        if isinstance(fill_value, int):
            fill_value = np.full(fit_image.shape[2], fill_value, fit_image.dtype)

        to_return = np.tile(fill_value, (output_height, output_width, 1))
        pad_top = int(0.5 * (output_height - fit_height))
        pad_left = int(0.5 * (output_width - fit_width))
        to_return[pad_top:pad_top+fit_height, pad_left:pad_left+fit_width] = fit_image
        return to_return


    def draw_boxes(self, boxes, classes, scores, img, cls_names, is_letter_box_image):
        draw = ImageDraw.Draw(img)
        detection_size = (self.height, self.width) 
        color = tuple(np.random.randint(0, 256, 3))
        for box, score, cls in zip(boxes, scores, classes):
            box = self.convert_to_original_size(box, np.array(detection_size),
                                                np.array(img.size),
                                                is_letter_box_image)
            draw.rectangle(box, outline=color)
            draw.text(box[:2], '{} {:.2f}%'.format(
                cls_names[cls], score * 100), fill=color)


    def convert_to_original_size(self, box, size, original_size, is_letter_box_image):
        if is_letter_box_image:
            box = box.reshape(2, 2)
            box[0, :] = self.letter_box_pos_to_original_pos(box[0, :], size, original_size)
            box[1, :] = self.letter_box_pos_to_original_pos(box[1, :], size, original_size)
        else:
            ratio = original_size / size
            box = box.reshape(2, 2) * ratio
        return list(box.reshape(-1))


    def letter_box_pos_to_original_pos(self, letter_pos, current_size, ori_image_size)-> np.ndarray:
        letter_pos = np.asarray(letter_pos, dtype=np.float)
        current_size = np.asarray(current_size, dtype=np.float)
        ori_image_size = np.asarray(ori_image_size, dtype=np.float)
        final_ratio = min(current_size[0]/ori_image_size[0], current_size[1]/ori_image_size[1])
        pad = 0.5 * (current_size - final_ratio * ori_image_size)
        pad = pad.astype(np.int32)
        to_return_pos = (letter_pos - pad) / final_ratio
        return to_return_pos

    # returns (input_image, original_image)
    def prepare_image_path(self, image_path):
        img = Image.open(image_path)
        return self.prepare_image(img), img

    def prepare_image(self, img):
        img_resized = self.letter_box_image(img, self.height, self.width, 128)
        img_resized = img_resized.astype(np.float32)
        return img_resized

    # runs prediction on prepared input_image (resized to height, width of
    # input tensor, converted to numpy array)
    def predict(self, input_img, confidence=0.3, iou_threshold=0.5):
        #t0 = time.time()
        self.interpreter.set_tensor(self.input_details[0]['index'], np.expand_dims(input_img, 0))
        self.interpreter.invoke()
        predictions = [self.interpreter.get_tensor(self.output_details[i]['index']) for i in range(len(self.output_details))]
        #t1 = time.time()
        #print("invoke time={}".format(t1 - t0))
        boxes, classes, scores = self.handle_predictions(predictions[0],
                                                         confidence=confidence,
                                                         iou_threshold=iou_threshold)
        return boxes, classes, scores

    # draws and saves on original_image
    def draw_boxes_and_save(self, boxes, classes, scores, orig_img, output_path):
        self.draw_boxes(boxes, classes, scores, orig_img, self.class_names, True)
        orig_img.save(output_path)

def test():
    t0 = time.time()
    yolo = YoloV3TFLite(os.path.join(os.getcwd(), 'yolo_v3.tflite'),
                        os.path.join(os.getcwd(), 'coco.names'))
    t1 = time.time()
    inp, img = yolo.prepare_image_path('example.jpg')
    boxes, classes, scores = yolo.predict(inp)
    t2 = time.time()
    print(boxes, classes, scores)
    print("load time={}, inference time={}".format(t1 - t0, t2 - t1))
    yolo.draw_boxes_and_save(boxes, classes, scores, img, 'output.jpg')

if __name__ == "__main__":
    test()

