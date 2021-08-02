import cv2
import numpy as np
import random
import math

from keras import layers
from keras import Model
from keras import losses

from keras import applications
import keras.backend as K
from tensorflow.python.keras.losses import mean_squared_error

"""
    Lessons Learnt
        1) np_array[...,n] (where n is an integer >=0) reshapes the array

    Research 
        1) shubham0204 for IOU regression componenet
        2) IOU regression loss https://github.com/Balupurohit23/IOU-for-bounding-box-regression-in-Keras/blob/master/iou_loss.py
        3) When writing a custom loss function with additional parameters you'll need this https://medium.com/@Bloomore/how-to-write-a-custom-loss-function-with-additional-arguments-in-keras-5f193929f7a0
        4) The article I was looking for
"""

""" ------------------------------------------------------ Dataset ----------------------------------------------------------- """
class generate_dataset():
    """ Creates a dataset given the """
    def __init__(self,dataset_config):
        self.img_w = dataset_config["ImageWidth"]
        self.img_h = dataset_config["ImageHeight"]
        self.img_channels = dataset_config["ImageChannels"]
        self.number_of_images = dataset_config["DatasetSize"]
         
        self.dataset = self.create_dataset()

    def create_dataset(self):
        """Creates a dataset of the given number of images such that it is a list of dictionaries"""
        """Format of dictionary {"Image":...,"GroundTruthBox":...,"ImageColour":...}"""
        return [self.create_random_colour_bbox_img() for v in range(self.number_of_images)]

    def generate_model_compatible_dataset(self, settings):
        """Generate dataset that is compatable with model"""
        """   CV2 input image (numpy uint8) (224,224)
              v
              Pre-processed input image (numpy float) (224,224)
              v
              Feature map after going through backbone CNN (7,7,512)
              v
              Region proposal output
        """
        """Assumption 1: There is only one size of image that is passed to the CNN (224,224)"""
        
        pre_processed_input_image_shape = pre_process_image_for_vgg(self.dataset[0]["HumanReadable"]["Image"],settings["ModelInputSize"]).shape
        feature_to_input, input_to_feature = get_conversions_between_input_and_feature(pre_processed_input_image_shape,settings["FeatureMapShape_Num_W_H_C"])
        anchor_boxes_input_coordinates_2d = get_input_coordinates_of_anchor_points(settings["FeatureMapShape_Num_W_H_C"],feature_to_input)

        # Update dataset with ModelCompatible item
        [ row.update({"ModelCompatible" : { 
            "Input" : settings["BackboneClassifier"].predict(pre_process_image_for_vgg(row["HumanReadable"]["Image"],settings["ModelInputSize"]))
            ,"Output" : generate_dataset.create_model_compatible_ground_truth(
                anchor_boxes_input_coordinates_2d
                ,row["HumanReadable"]["GroundTruthBox"]
                ,settings["OutputShape"][0]
            )
        }
        }) for row in self.dataset]

    def get_machine_formatted_dataset(self):
        """produces a dataset in a format which ML can train on split into x and y data"""
        x_data = np.array([row["ModelCompatible"]["Input"][0] for row in self.dataset])
        y_data = [
            np.array([row["ModelCompatible"]["Output"][0][0] for row in self.dataset])
            ,np.array([row["ModelCompatible"]["Output"][1][0] for row in self.dataset])
        ]
        return x_data, y_data

    @staticmethod # can be called without initialising custom_model. Assumption means this function is associated to the dataset
    def create_regression_model_compatible_ground_truth(anchor_boxes_input_coordinates_2d,gt_bbox):
        """Creates ground truth for regression component of RPN in parameterised coordinate space"""
        """Assumption 1: Dataset only contains 1 object per frame"""
        delta_anchor_boxes_human_readable =  [[ # Used for readability
            {
                "delta_x_c" : gt_bbox["xywh"]["x_c"] - anchor_bbox["xywh"]["x_c"]
                ,"delta_y_c" : gt_bbox["xywh"]["y_c"] - anchor_bbox["xywh"]["y_c"]
                ,"delta_w" : gt_bbox["xywh"]["w"] - anchor_bbox["xywh"]["w"]
                ,"delta_h" : gt_bbox["xywh"]["h"] - anchor_bbox["xywh"]["h"]
                }
            for anchor_bbox in list_of_anchor_boxes] for list_of_anchor_boxes in anchor_boxes_input_coordinates_2d]

        delta_parametarised_anchor_boxes_human_readable =  [[
            {
                "parm_delta_t_x" : parameterize_x_or_y_centre(gt_bbox["xywh"]["x_c"], anchor_bbox["xywh"]["x_c"], anchor_bbox["xywh"]["w"]) 
                ,"parm_delta_t_y" : parameterize_x_or_y_centre(gt_bbox["xywh"]["y_c"], anchor_bbox["xywh"]["y_c"], anchor_bbox["xywh"]["h"])
                ,"parm_delta_t_w" : parameterize_width_or_height(gt_bbox["xywh"]["w"], anchor_bbox["xywh"]["w"])
                ,"parm_delta_t_h" : parameterize_width_or_height(gt_bbox["xywh"]["h"], anchor_bbox["xywh"]["h"])
                }
            for anchor_bbox in list_of_anchor_boxes] for list_of_anchor_boxes in anchor_boxes_input_coordinates_2d]

        delta_regression_boxes_ground_truth = np.array([[[
            [
                float(deltas["parm_delta_t_x"])
                ,float(deltas["parm_delta_t_y"])
                ,float(deltas["parm_delta_t_w"])
                ,float(deltas["parm_delta_t_h"])
            ]
            for deltas in list_of_parm_deltas] for list_of_parm_deltas in delta_parametarised_anchor_boxes_human_readable]])
        return delta_regression_boxes_ground_truth 

    @staticmethod # can be called without initialising custom_model. Assumption means this function is associated to the dataset
    def create_classifier_model_compatible_ground_truth(anchor_boxes_input_coordinates_2d,ground_truth_bbox,output_shape):
        """Creates an array representing ground truth for selecting the anchor box with highest IOU"""
        """Assumption 1: Dataset only contains 1 object per frame"""
        iou_of_anchor_boxes_with_gt =  [[
            get_iou_from_bboxes(anchor_box,ground_truth_bbox)
            for anchor_box in list_of_anchor_boxes] for list_of_anchor_boxes in anchor_boxes_input_coordinates_2d]

        iou_of_anchor_boxes_with_gt_array = np.array(iou_of_anchor_boxes_with_gt)
        height_index = int(max(iou_of_anchor_boxes_with_gt_array.argmax(axis=0)))
        width_index = int(max(iou_of_anchor_boxes_with_gt_array.argmax(axis=1)))
        bbox_channel_index = 0 # There is no scale, aspect ratio adjustments. Hence channels will always be 0

        region_proposal_classifier_ground_truth = np.zeros(output_shape,np.float64)
        region_proposal_classifier_ground_truth[0][height_index][width_index][bbox_channel_index] = 1.0

        return region_proposal_classifier_ground_truth

    @staticmethod # can be called without initialising custom_model
    def create_model_compatible_ground_truth(anchor_boxes_input_coordinates_2d,gt_box,classifier_outputshape):
        """Creates ground truth aka the true output for the RPN"""
        classifier_ground_truth = generate_dataset.create_classifier_model_compatible_ground_truth(
            anchor_boxes_input_coordinates_2d
            ,gt_box
            ,classifier_outputshape)

        regression_ground_truth = generate_dataset.create_regression_model_compatible_ground_truth(
            anchor_boxes_input_coordinates_2d
            ,gt_box)
        return [classifier_ground_truth, regression_ground_truth] 

    def create_random_colour_bbox_img(self):
        """Creates a box of random colour and random points and returns an cv2 image"""
        img = np.zeros((self.img_h,self.img_w,self.img_channels),dtype=np.uint8)
        b_box_coords = self.generate_x1y1_x2y2()
        random_colour = generate_dataset.generate_random_colour()

        point1_top_left = (b_box_coords["p1p2"]["x1"],b_box_coords["p1p2"]["y1"])
        point2_bottom_right = (b_box_coords["p1p2"]["x2"],b_box_coords["p1p2"]["y2"])
        fill_box = -1
        cv2.rectangle(img,point1_top_left,point2_bottom_right,random_colour,fill_box)
        
        row_in_dataset = {
            "HumanReadable" : {
                "Image" : img
                ,"GroundTruthBox" : b_box_coords
                ,"ImageColourRBG" : random_colour
            }
        }
        return row_in_dataset

    def generate_x1y1_x2y2(self):
        """Creates random coordinates of a bounding box in two forms, {(x1,y1),(x2,y2)} & also in x_centre, y_centre, width & height"""
        """Note 1: Cannot have a width or height of 0 account for this"""
        x1 = random.randint(0,self.img_w-1)
        y1 = random.randint(0,self.img_h-1)
        x2 = random.randint(x1+1,self.img_w)
        y2 = random.randint(y1+1,self.img_h)

        ground_truth_bounding_box = {
            "p1p2" : {
                "x1" : x1 
                ,"y1" : y1
                ,"x2" : x2
                ,"y2" : y2
            }
            ,"xywh" : {
                "x_c" : int(round(x1 + (x2-x1)/2 ))
                ,"y_c" : int(round(y1 + (y2-y1)/2 ))
                ,"w" : x2-x1
                ,"h" : y2-y1
            }
        }
        return ground_truth_bounding_box

    def get_random_image(self):
        """Gets a random image from the human readble dataset"""
        return random.choice(self.dataset)["HumanReadable"]["Image"]

    @staticmethod 
    def generate_random_colour():
        """Generate a tuple of random colours in cv2, rbg"""
        r = random.randint(0,255)
        g = random.randint(0,255)
        b = random.randint(0,255)
        return (r,b,g)

""" -------------------------------------------------------- Model ----------------------------------------------------------- """

class generate_custom_model(): 
    """Generates the custom RPN model, trains and predicts"""
    def __init__(self,config):
        self.backbone_classifier = config["BackboneClassifier"]
        self.backbone_classifier_input_shape = config["ModelInputSize"]
        self.backbone_output_channels = config["BackboneClassifier"].layers[-1].output_shape[-1] # Assumes VGG16

        self.model = self.build_region_classifier()

    def build_region_classifier(self):
        """Creates the RPN classifier model"""
        assert type(self.backbone_output_channels) == int # Should be an integer

        self.anchor_boxes_per_point = 1 # Used for simplified model

        # Input layer, should have the same number of channels that the headless classifier produces
        feature_map_input = layers.Input(shape=(None,None,self.backbone_output_channels),name="RPN_Input_Same")
        # CNN component, ensure that padding is the same so that it has the same dimensions as input feature map
        convolution_3x3 = layers.Conv2D(filters=self.anchor_boxes_per_point,kernel_size=(3, 3),name="3x3",padding="same")(feature_map_input)
        # Output objectivness
        objectivness_output_scores = layers.Conv2D(filters=self.anchor_boxes_per_point, kernel_size=(1, 1),activation="sigmoid",kernel_initializer="uniform",name="scores1")(convolution_3x3)
        # Output deltas for bounding box
        deltas_output_scores = layers.Conv2D(filters=4*self.anchor_boxes_per_point,kernel_size=(1,1),activation="sigmoid",kernel_initializer="uniform",name="deltas1")(convolution_3x3)
        # Create model with input feature map and output
        model = Model(inputs=[feature_map_input], outputs=[objectivness_output_scores,deltas_output_scores])    
        # Set loss and compile
        model.compile(optimizer='adam', loss={'scores1':losses.binary_crossentropy,'deltas1':losses.huber})

        return model

    def get_feature_map_shape(self,input_image):
        """Provides the input shape of the RPN model in the format (number,height,width,channels)"""
        pre_processed_img = pre_process_image_for_vgg(input_image,self.backbone_classifier_input_shape)
        feature_map_output_from_backbone_classifier = self.backbone_classifier.predict(pre_processed_img)
        feature_map_shape_num_h_w_c = feature_map_output_from_backbone_classifier.shape
        return feature_map_shape_num_h_w_c

    def get_model_output_shape(self,input_image):
        """Gets the model output shape"""
        """Accounts for RPN which provides outputs as list"""
        pre_processed_img = pre_process_image_for_vgg(input_image,self.backbone_classifier_input_shape)
        feature_map_output_from_backbone_classifier = self.backbone_classifier.predict(pre_processed_img)
        
        model_output = self.model.predict(feature_map_output_from_backbone_classifier)
        if type(model_output) == list: # RPN outputs a list
            model_output_shapes = [model_output_element.shape for model_output_element in model_output]
        else:
            model_output_shapes = model_output.shape
        return model_output_shapes

    def predict_object(self,image_224_224,objectiveness_threshold=0.4):
        """Provides the bounding boxes for a given colour image of 224x224 with an objectivness_threshold that is default to 0.7"""
        """TODO adjust to handle multiple images"""
        pre_processed_input_image = pre_process_image_for_vgg(image_224_224,self.backbone_classifier_input_shape)
        feature_map = self.backbone_classifier.predict(pre_processed_input_image)

        feature_to_input, input_to_feature = get_conversions_between_input_and_feature(pre_processed_input_image.shape,feature_map.shape)
        anchor_boxes_input_coordinates_2d = get_input_coordinates_of_anchor_points(feature_map.shape,feature_to_input)

        predicted_output = self.model.predict(feature_map)
        [classification, parm_deltas] = [0,1]
        

        anchor_boxes_indicies_above_threshhold = np.argwhere(predicted_output[classification] > objectiveness_threshold)
        if len(anchor_boxes_indicies_above_threshhold) == 0:
            return []
        
        list_of_bboxes = []
        for a_box_index in anchor_boxes_indicies_above_threshhold:
            [img_idx, height_idx, width_idx, channel_idx] = [a_box_index[0], a_box_index[1], a_box_index[2], a_box_index[3]]

            selected_anchor_box = anchor_boxes_input_coordinates_2d[height_idx][width_idx]
            parm_bbox_refinement = predicted_output[parm_deltas][0][height_idx][width_idx]
            [t_x , t_y , t_w , t_h] = [0,1,2,3]

            delta_x_c = inverse_parameterize_x_or_y_centre(parm_bbox_refinement[t_x], selected_anchor_box["xywh"]["x_c"], selected_anchor_box["xywh"]["w"])
            delta_y_c = inverse_parameterize_x_or_y_centre(parm_bbox_refinement[t_y], selected_anchor_box["xywh"]["y_c"], selected_anchor_box["xywh"]["h"])
            delta_width = inverse_parameterize_width_or_height(parm_bbox_refinement[t_w], selected_anchor_box["xywh"]["w"])
            delta_height = inverse_parameterize_width_or_height(parm_bbox_refinement[t_h], selected_anchor_box["xywh"]["h"])

            pred_bbox = {
                "xywh" : {
                    "x_c" : round(selected_anchor_box["xywh"]["x_c"] + delta_x_c)
                    ,"y_c" : round(selected_anchor_box["xywh"]["y_c"] + delta_y_c)
                    ,"w" : round(selected_anchor_box["xywh"]["w"] + delta_width)
                    ,"h" : round(selected_anchor_box["xywh"]["h"] + delta_height)
                }
                ,"p1p2" : {
                    "x1" : round(selected_anchor_box["xywh"]["x_c"] + delta_x_c - (selected_anchor_box["xywh"]["w"] + delta_width)/2)
                    ,"y1" : round(selected_anchor_box["xywh"]["y_c"] + delta_y_c - (selected_anchor_box["xywh"]["h"] + delta_height)/2)
                    ,"x2" : round(selected_anchor_box["xywh"]["x_c"] + delta_x_c + (selected_anchor_box["xywh"]["w"] + delta_width)/2)
                    ,"y2" : round(selected_anchor_box["xywh"]["y_c"] + delta_y_c + (selected_anchor_box["xywh"]["h"] + delta_height)/2)
                }
            }

            objectiveness_prob = predicted_output[classification][0][height_idx][width_idx][channel_idx]

            list_of_bboxes.append({
                "Probability" : objectiveness_prob
                ,"BoundingBox" : pred_bbox
            })
        return list_of_bboxes
   
""" -------------------------------------------------------- UTILITY FUNCTIONS ----------------------------------------------------------- """
def parameterize_x_or_y_centre(ground_truth_x_or_y_centre, anchor_box_x_or_y_centre, anchor_box_w_or_h):
    """Parameterizes the delata box centre coordinate changes using t_x=(gt_x_c-abox_x_c)/abox_w and t_y=(gt_y_c-abox_y_c)/abox_h"""
    return (ground_truth_x_or_y_centre - anchor_box_x_or_y_centre)/anchor_box_w_or_h

def parameterize_width_or_height(ground_truth_w_or_h, anchor_box_w_or_h):
    """Parameterizes the delata box centre coordinate changes using t_w=log(gt_w/abox_w) and t_h=log(gt_w/abox_w)"""
    """Note 1: Ensure you do not have any ground truth boxes with a width or height of 0 otherwise it will break the parameterisation"""
    return math.log(ground_truth_w_or_h/anchor_box_w_or_h)

def inverse_parameterize_x_or_y_centre(parameter_t_x_or_y_centre, anchor_box_x_or_y_centre, anchor_box_w_or_h):
    """Inverse of x or y centre parameterisation aka ground_truth_x_centre=t_x*abox_w+abox_x_centre"""
    return parameter_t_x_or_y_centre*anchor_box_w_or_h + anchor_box_x_or_y_centre

def inverse_parameterize_width_or_height(parameter_t_w_or_h, anchor_box_w_or_h):
    """Inverse of width or height parameterisation aka ground_truth_width="""
    """Note 1: Ensure you do not have any ground truth boxes with a width or height of 0 otherwise it will break the parameterisation"""
    return math.exp(parameter_t_w_or_h) * anchor_box_w_or_h

def pre_process_image_for_vgg(img,input_size):
    """
        Resizes the image to input of VGGInputSize specified in the config dictionary
        Normalises the image
        Reshapes the image to an array of images e.g. [[img],[img],..]

        Inputs:
            img: a numpy array or array of numpy arrays that represent an image
            input_size: a tuple of (width, height)
    """
    if type(img) == np.ndarray: # Single image 
        resized_img = cv2.resize(img,input_size,interpolation = cv2.INTER_AREA)
        normalised_image = applications.vgg16.preprocess_input(resized_img)
        reshaped_to_array_of_images = np.array([normalised_image])
        return reshaped_to_array_of_images
    elif type(img) == list: # list of images
        img_list = img
        resized_img_list = [cv2.resize(image,input_size,interpolation = cv2.INTER_AREA) for image in img_list]
        resized_img_array = np.array(resized_img_list)
        normalised_images_array = applications.vgg16.preprocess_input(resized_img_array)
        return normalised_images_array

def get_conversions_between_input_and_feature(pre_processed_input_image_shape,feature_map_shape):
    """
        Finds the scale and offset from the feature map (output) of the CNN classifier to the pre-processed input image of the CNN
        Finds the inverse, pre-processed input image to feature map

        Input:
            pre_processed_input_image_shape: The 3/4d shape of the pre-processed input image that is passed to the backbone CNN classifier

        Returns a dictionary of values to easily pass variables
    """
    # Find shapes of feature maps and input images to the classifier CNN
    assert len(pre_processed_input_image_shape) in [3,4] # Either a 4d array with [:,height,width,channels] or just a single image [height,width,channels]
    assert len(feature_map_shape) in [3,4] # Either a 4d array with [:,height,width,channels] or just a single feature map [height,width,channels]
    if len(pre_processed_input_image_shape) == 3:
        img_height, img_width, _ = pre_processed_input_image_shape
    elif len(pre_processed_input_image_shape) == 4:
        _, img_height, img_width, _ = pre_processed_input_image_shape

    if len(feature_map_shape) == 3:
        features_height, features_width, _ = feature_map_shape
    elif len(feature_map_shape) == 4:
        _, features_height, features_width, _ = feature_map_shape

    # Find mapping from features map (output of backbone_model.predict) back to the input image
    feature_to_input_x_scale = img_width / features_width
    feature_to_input_y_scale = img_height / features_height

    # Put anchor points in the centre of 
    feature_to_input_x_offset = feature_to_input_x_scale/2
    feature_to_input_y_offset = feature_to_input_y_scale/2

    # Store as dictionary
    feature_to_input = {
        "x_scale": feature_to_input_x_scale
        ,"y_scale": feature_to_input_y_scale
        ,"x_offset" : feature_to_input_x_offset
        ,"y_offset" : feature_to_input_y_offset
    }

    # Find conversions from input image to feature map (CNN output)
    input_to_feature_x_scale = 1/feature_to_input_x_scale
    input_to_feature_y_scale = 1/feature_to_input_y_scale
    input_to_feature_x_offset = -feature_to_input_x_offset
    input_to_feature_y_offset = -feature_to_input_y_offset

    # Store as dictionary
    input_to_feature = {
        "x_scale": input_to_feature_x_scale
        ,"y_scale": input_to_feature_y_scale
        ,"x_offset" : input_to_feature_x_offset
        ,"y_offset" : input_to_feature_y_offset
    }

    return feature_to_input, input_to_feature

def get_input_coordinates_of_anchor_points(feature_map_shape,feature_to_input):
    """Maps the CNN output (Feature map) coordinates on the pre-processed input image space to the backbone CNN"""
    """Returns the coordinates as a 2d of dictionaries with the format {"x":x,"y":y}"""
    assert len(feature_map_shape) in [3,4] # Either a 4d array with [:,height,width,channels] or just a single feature map [height,width,channels]

    if len(feature_map_shape) == 3:
        features_height, features_width, _ = feature_map_shape
    elif len(feature_map_shape) == 4:
        _, features_height, features_width, _ = feature_map_shape

    # For the feature map (x,y) determine the anchors on the input image (x,y) as array 
    feature_to_input_coords_x  = [int(x_feature*feature_to_input["x_scale"]+feature_to_input["x_offset"]) for x_feature in range(features_width)]
    feature_to_input_coords_y  = [int(y_feature*feature_to_input["y_scale"]+feature_to_input["y_offset"]) for y_feature in range(features_height)]
    coordinate_of_anchor_points_2d = [[{
        "p1p2" : {
            "x1" : round(x_coord - feature_to_input["x_scale"]/2)
            ,"y1" : round(y_coord - feature_to_input["y_scale"]/2)
            ,"x2" : round(x_coord + feature_to_input["x_scale"]/2)
            ,"y2" : round(y_coord + feature_to_input["y_scale"]/2)
        }
        ,"xywh" : {
            "x_c" : round(x_coord)
            ,"y_c" : round(y_coord)
            ,"w" : round(feature_to_input["x_scale"])
            ,"h" : round(feature_to_input["y_scale"])
        }
        } for x_coord in feature_to_input_coords_x] for y_coord in feature_to_input_coords_y]

    return coordinate_of_anchor_points_2d

def get_iou_from_bboxes(bbox_1,bbox_2):
    """Finds the Intersection Over Union (IOU) of two bounding boxes"""
    """Taken from https://stackoverflow.com/questions/25349178/calculating-percentage-of-bounding-box-overlap-for-image-detector-evaluation """
    # determine the coordinates of the intersection rectangle
    x_left = max(bbox_1["p1p2"]['x1'], bbox_2["p1p2"]['x1'])
    y_top = max(bbox_1["p1p2"]['y1'], bbox_2["p1p2"]['y1'])
    x_right = min(bbox_1["p1p2"]['x2'], bbox_2["p1p2"]['x2'])
    y_bottom = min(bbox_1["p1p2"]['y2'], bbox_2["p1p2"]['y2'])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # The intersection of two axis-aligned bounding boxes is always an
    # axis-aligned bounding box
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # compute the area of both AABBs
    bbox_1_area = (bbox_1["p1p2"]['x2'] - bbox_1["p1p2"]['x1']) * (bbox_1["p1p2"]['y2'] - bbox_1["p1p2"]['y1'])
    bbox_2_area = (bbox_2["p1p2"]['x2'] - bbox_2["p1p2"]['x1']) * (bbox_2["p1p2"]['y2'] - bbox_2["p1p2"]['y1'])

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = intersection_area / float(bbox_1_area + bbox_2_area - intersection_area)
    return iou

""" -------------------------------------------------------- CONFIG ----------------------------------------------------------- """
config = {
    "Dataset" : 
    {
        "ImageWidth" : 224
        ,"ImageHeight" : 224
        ,"ImageChannels" : 3
        ,"DatasetSize" : 100
    }
    ,"Model" : {
        "ModelInputSize" : (224,224)
        # More settings added later in code
    }
}

if __name__ == "__main__":
    # Select classifier
    config["Model"]["BackboneClassifier"] = applications.VGG16(include_top=False,weights='imagenet')

    # Create dataset of random boxes in human readable format
    dataset = generate_dataset(config["Dataset"])
    
    # Create custom model and add to settings
    custom_model = generate_custom_model(config["Model"])
    config["Model"]["FeatureMapShape_Num_W_H_C"] = custom_model.get_feature_map_shape(dataset.get_random_image())
    config["Model"]["OutputShape"] = custom_model.get_model_output_shape(dataset.get_random_image())

    # Create machine dataset, this takes a while because it has to run through the CNN
    dataset.generate_model_compatible_dataset(config["Model"])

    # Split dataset & train model
    x_data, y_data = dataset.get_machine_formatted_dataset()
    history = custom_model.model.fit(x=x_data,y=y_data, batch_size=8, shuffle=True, epochs=16, verbose=1,validation_split=0.1)

    # Show Image and predictions from RPN
    for img_sel, row in enumerate(dataset.dataset):

        img = row["HumanReadable"]["Image"].copy()
        pred_bboxes = custom_model.predict_object(img,objectiveness_threshold=0.4)

        print(f'[img={img_sel}] Ground Truth = {dataset.dataset[img_sel]["HumanReadable"]["GroundTruthBox"]["p1p2"]}')
        for pred_bbox in pred_bboxes:
            print(f'[img={img_sel}]Predicted [{round(pred_bbox["Probability"]*100,4)}%] = {pred_bbox["BoundingBox"]["p1p2"]}')
            img = cv2.rectangle(
                img
                ,(pred_bbox["BoundingBox"]["p1p2"]["x1"],pred_bbox["BoundingBox"]["p1p2"]["y1"])
                ,(pred_bbox["BoundingBox"]["p1p2"]["x2"],pred_bbox["BoundingBox"]["p1p2"]["y2"])
                ,(255,255,255)
                ,3
            )

        cv2.imshow("Prediction Image",img)
        cv2.waitKey(250)
    print('finishing')