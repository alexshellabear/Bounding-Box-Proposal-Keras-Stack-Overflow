# Goal #
Draw a bounding box around an object of interest. Ideal architecture is using Faster-RCNN from stratch to understand how it works.

## Deeper Explanation of Goal ##
I want to be able to draw bounding boxes around objects of interest. To simplify the problem I am trying to understand the region proposal network (RPN) component of Faster-RCNN. The below architecture is what I hope to implement
- Backbone CNN is VGG16, which will produce the feature map of shape (:,7,7,512)
- Region Proposal Network will produce two outputs, softmax classification of anchor location & regression to adjust anchor boxes

![](https://www.researchgate.net/profile/Young-Jin-Cha/publication/321341582/figure/fig2/AS:565755498463232@1511898030310/The-schematic-architecture-of-the-RPN.png)

Please note: This implementation will not use scale and aspect ratio so there will only be 1 channel for anchor boxes and output of RPN will be
- RPN Anchor Box Softmax output (:,7,7,1)
- RPN Regression Output (:,7,7,4) in the form [x delta,y delta,width delta,height delta]

# Overall Question #
1) **How do I make the regression component of RPN work?** I have the classification component working here https://github.com/alexshellabear/Still-Simple-Region-Proposal-Network

### Sub Questions ###
1) Is the RPN regression output in the input image coordinate space (i.e 224 pixels by 224 pixels)? An example of one the deltas may be ```[100.0 (move box 100 left),10.0 (move box 10 up),-100.0 (reduce box width by 100 pixels),-20.0 (reduce box height by 20 pixels)]```
2) What is the loss function for regression component of RPN?

## Explanation of Highest Level Code ##
This is comprised of 4 steps
1) generate human readable dataset
2) Compile model
3) generate model compatible dataset
4) split data & train

```Python
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
```

## Explanation of Dataset Generation ##