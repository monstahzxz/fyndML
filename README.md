# Fynd ML Challenge 2019
An entry to the Fynd ML open challenge 2019
## Introduction
An implementation of a CLDNN (Convolutional, Long Short-Term Memory, Fully Connected Deep Neural Networks) for the Fynd Dataset to classify footwear on the basis of closures.

## Problem Statement
1. Each product has multiple views (front, left, right, back) and only a few views have the information about the product through which you can make the ML model learn its features. Build ML model and explain your approach on how will you select the specific view from the given set of different views.
2. Identify the closure type of footwear from the data (CSV) provided which has imbalanced data distribution across 6 classes. You can collect more data using web scraping or different sources. Build ML model which effectively identifies the closure type of the footwear. 
Here is a link to the visual document to understand each class of the data.

## Dataset Description
Public dataset: Contains a CSV file with a column name ‘class’ and image URLs to different views.
### Example

id | view_1 | view_2 | view_3 | view_4 | view_5 | class
--- | --- | --- | --- | --- | --- | --- 
3213e9a8da734c368db6bed4b512tt411.jpg | *url_1* | *url_2* | *url_3* | *url_4* | *url_5* | zipper

* *Note: Each view gives a different POV of the same footwear*

### Closure Types
![Backstrap Type, Zipper Type, Hook & Look Type, Buckle Type, Lace Up Type, Slip-On Type](https://cdn-images-1.medium.com/max/1600/1*NVy-YMJ5w3dHSB9jeV31Sw.png)

* *Image taken from the original challenge [website](https://blog.gofynd.com/machine-learning-internship-challenge-2019-6b4e9dddb637)*

## Approach and Network Architecture
<dl>
  <dt>Takeaways from dataset</dt>
  <dd>A personal takeaway from the public dataset is the need for all views in the dataset. Instead of selecting a particular view on which our model should make predictions, we can use the combined information learnt from all the different views so that the classifier can make a better informed decision. Each view can contribute towards the learning of different aspects of the footwear like height, presence of laces, buckles, open ends and so on. A simple justification for this is the difference between the closure types. A <b>zipper</b> footwear can be easily identified from a view containing the zipper itself, the height of the part containing the zipper, absense of hooks, laces and so on. These informations can be extracted from the various views. For instance: </dd>
</dl>

*Zipper views*

<img src="https://vision-images-store.s3.amazonaws.com/internship/zipper/view_1/1a71911437d54b3980d3d81001ec19a4.jpg" height="200" width="200" title="View 1"/><img src="https://vision-images-store.s3.amazonaws.com/internship/zipper/view_5/1a71911437d54b3980d3d81001ec19a4.jpg" height="200" width="200" title="View 2"/>

From these two views we can gather necessary informations. From **view 1**, we can conclude the absence of any sorts of lace or buckle. And the **view 2** shows the zipper on the left part of the footwear.

<dl>
  <dt>Architecture of Model</dt>
  <dd>Model type - CLDNN<br/>
  <img src="https://3qeqpr26caki16dnhd19sv6by6v-wpengine.netdna-ssl.com/wp-content/uploads/2017/07/Convolutional-Neural-Network-Long-Short-Term-Memory-Network-Archiecture.png"/><br/>
    Image source: <a href="https://machinelearningmastery.com/cnn-long-short-term-memory-networks/">here</a><br/>
  </dd>
</dl>