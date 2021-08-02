# Goal #
Draw a bounding box around an object of interest. Ideal architecture is using Faster-RCNN from stratch to understand how it works.

my email address is alexshellabear@gmail.com, what is yours?

## Deeper Explanation of Goal ##
I want to be able to draw bounding boxes around objects of interest. To simplify the problem I am trying to understand the region proposal network (RPN) component of Faster-RCNN. The below architecture is what I hope to implement
- Backbone CNN is VGG16, which will produce the feature map of shape (:,7,7,512)
- Region Proposal Network will produce two outputs, softmax classification of anchor location & regression to adjust anchor boxes

![](https://www.researchgate.net/profile/Young-Jin-Cha/publication/321341582/figure/fig2/AS:565755498463232@1511898030310/The-schematic-architecture-of-the-RPN.png)

Please note: This implementation will not use scale and aspect ratio so there will only be 1 channel for anchor boxes and output of RPN will be
- RPN Anchor Box Softmax output (:,7,7,1)
- RPN Regression Output (:,7,7,4) in the form [x delta,y delta,width delta,height delta]

# Overall Question #
1) **How do I make the regression component of RPN work?** I have the classification component working
- https://github.com/alexshellabear/Still-Simple-Region-Proposal-Network

## Explanation of Highest Level Code ##
This is comprised of 5 steps
1) Generate human readable dataset
2) Compile model
3) Generate model compatible dataset
4) Split data & train
5) Predict & show predictions

```Python
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
```

## Explanation of Dataset Generation ##
1 random box per image is drawn onto a black image of 224x224 pixels. 