import cv2
import numpy as np
import random

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
        """Creates ground truth for regression component of RPN in input space coordinates"""
        """Assumption 1: Dataset only contains 1 object per frame"""
        delta_anchor_boxes_human_readable =  [[
            {
                "delta_x_c" : gt_bbox["xywh"]["x_c"] - anchor_bbox["xywh"]["x_c"]
                ,"delta_y_c" : gt_bbox["xywh"]["y_c"] - anchor_bbox["xywh"]["y_c"]
                ,"delta_w" : gt_bbox["xywh"]["w"] - anchor_bbox["xywh"]["w"]
                ,"delta_h" : gt_bbox["xywh"]["h"] - anchor_bbox["xywh"]["h"]
                }
            for anchor_bbox in list_of_anchor_boxes] for list_of_anchor_boxes in anchor_boxes_input_coordinates_2d]

        delta_regression_boxes_ground_truth = np.array([[[
            [
                float(deltas["delta_x_c"])
                ,float(deltas["delta_y_c"])
                ,float(deltas["delta_w"])
                ,float(deltas["delta_h"])
            ]
            for deltas in list_of_deltas] for list_of_deltas in delta_anchor_boxes_human_readable]])
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
        x1 = random.randint(0,self.img_w)
        y1 = random.randint(0,self.img_h)
        x2 = random.randint(x1,self.img_w)
        y2 = random.randint(y1,self.img_h)

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

    def calculate_iou(target_boxes,predicted_boxes):
        """Calculates IOU for loss function"""
        """TODO confirm Assumes that input matrix is in the form [[],[],[],[]]"""
        # TODO give credit to shubham0204 github repo
        # Get the area of intersection 
        top_left_x_intersect_matrix = K.maximum(target_boxes[...,0],predicted_boxes[...,0])
        top_left_y_intersect_matrix = K.maximum(target_boxes[...,1],predicted_boxes[...,1])
        bottom_right_x_intersect_matrix = K.minimum(target_boxes[...,2],predicted_boxes[...,2])
        bottom_right_y_intersect_matrix = K.minimum(target_boxes[...,3],predicted_boxes[...,3])

        intersect_width_matrix = K.maximum(0.0,bottom_right_x_intersect_matrix-top_left_x_intersect_matrix)
        intersect_height_matrix = K.maximum(0.0,bottom_right_y_intersect_matrix-top_left_y_intersect_matrix)

        intersection_area = intersect_width_matrix * intersect_height_matrix

        # Target boxes area
        target_boxes_top_left_x = target_boxes[...,0]
        target_boxes_bottom_right_x = target_boxes[...,2]
        target_boxes_width = target_boxes_bottom_right_x - target_boxes_top_left_x

        target_boxes_top_left_y = target_boxes[...,1]
        target_boxes_bottom_right_y = target_boxes[...,3]
        target_boxes_height = target_boxes_bottom_right_y - target_boxes_top_left_y

        target_boxes_area = target_boxes_width * target_boxes_height

        # predicted boxes area
        predicted_boxes_top_left_x = predicted_boxes[...,0]
        predicted_boxes_bottom_right_x = predicted_boxes[...,2]
        predicted_boxes_width = predicted_boxes_bottom_right_x - predicted_boxes_top_left_x

        predicted_boxes_top_left_y = predicted_boxes[...,1]
        predicted_boxes_bottom_right_y = predicted_boxes[...,3]
        predicted_boxes_height = predicted_boxes_bottom_right_y - predicted_boxes_top_left_y

        predicted_boxes_area = predicted_boxes_width * predicted_boxes_height

        # Calculate IOU (intersection over union)
        intersection_over_union = intersection_area / (target_boxes_area + predicted_boxes_area - intersection_area)

        return intersection_over_union

    @staticmethod # can be called without initialising object
    def custom_loss(y_ground_truth, y_predicted):
        """Calculates custom loss function to solve regression component of RPN (Region Proposal Network)"""
        mean_squared_error = losses.mean_squared_error(y_ground_truth, y_predicted)
        intersection_over_union = generate_custom_model.calculate_iou(y_ground_truth, y_predicted)
        custom_loss_values = mean_squared_error + (1 - intersection_over_union)
        return custom_loss_values

""" -------------------------------------------------------- UTILITY FUNCTIONS ----------------------------------------------------------- """
    
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
    """
        Maps the CNN output (Feature map) coordinates on the pre-processed input image space to the backbone CNN 
        Returns the coordinates as a 2d of dictionaries with the format {"x":x,"y":y}
    """
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

def iou_loss(y_true, y_pred):
    """IOU loss function for """
    # iou loss for bounding box prediction
    # input must be as [x1, y1, x2, y2]
    
    # AOG = Area of Groundtruth box
    AoG = K.abs(K.transpose(y_true)[2] - K.transpose(y_true)[0] + 1) * K.abs(K.transpose(y_true)[3] - K.transpose(y_true)[1] + 1)
    
    # AOP = Area of Predicted box
    AoP = K.abs(K.transpose(y_pred)[2] - K.transpose(y_pred)[0] + 1) * K.abs(K.transpose(y_pred)[3] - K.transpose(y_pred)[1] + 1)

    # overlaps are the co-ordinates of intersection box
    overlap_0 = K.maximum(K.transpose(y_true)[0], K.transpose(y_pred)[0])
    overlap_1 = K.maximum(K.transpose(y_true)[1], K.transpose(y_pred)[1])
    overlap_2 = K.minimum(K.transpose(y_true)[2], K.transpose(y_pred)[2])
    overlap_3 = K.minimum(K.transpose(y_true)[3], K.transpose(y_pred)[3])

    # intersection area
    intersection = (overlap_2 - overlap_0 + 1) * (overlap_3 - overlap_1 + 1)

    # area of union of both boxes
    union = AoG + AoP - intersection
    
    # iou calculation
    iou = intersection / union

    # bounding values of iou to (0,1)
    iou = K.clip(iou, 0.0 + K.epsilon(), 1.0 - K.epsilon())

    # loss for the iou value
    iou_loss = -K.log(iou)

    return iou_loss


""" -------------------------------------------------------- CONFIG ----------------------------------------------------------- """
config = {
    "Dataset" : 
    {
        "ImageWidth" : 224
        ,"ImageHeight" : 224
        ,"ImageChannels" : 3
        ,"DatasetSize" : 50
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
    history = custom_model.model.fit(x=x_data,y=y_data, batch_size=8, shuffle=True, epochs=4, verbose=1,validation_split=0.1)

    predicted_output = custom_model.model.predict(dataset.dataset[0]["ModelCompatible"]["Input"])