# cs570-finalproject
cs570 finalproject


# Progress

1. <del> Train and test Autoencoder </del>
2. <del> Train and test CNN </del>
3. <del> Test Auto + CNN model for clean data </del>
4. <del> Test Auto + CNN model for natural noise data </del>
5. <del> Generate Adverserial example for CNN </del>
6. <del> Test Auto + CNN model for Adverserial example </del>



File Usage
==========
## convnet.py
> made by Sungwon Han
    Training CNN

## new_auto_cnn.py
> made by S & Y
    Test Auto encoder + CNN model: DA CNN or AGDA CNN
    
    

## cnn_Adverserial.py
> made by YongsuBaek

A few lines for generating and testing Adverserial inputs added to __convnet.py__.  
```{.python}
    from cleverhans.attacks_tf import fgm
    # Adverserial input generating
    adv_x = fgm(X, Y_conv, y=None, eps=Adv_eps, ord=np.inf, clip_min=None, clip_max=None)
    Ad_Y_conv = cnn(adv_x, keep_prob, 2, 2, 10, 20, 50, 25) # output of them
```
## draw.py
> made by Yongsu Baek
### * case1 : import

  it contains the function

      draw(img_array, filename)

input
 1. img_array : (8277,) array of image
 2. filename : file path + file name to save the image
  
return: None

### * case2 : with arguments
 1. with one arguments:

        python draw.py [path/file_name]

   will save the given (8277,) text file to result_img directory as same filename. Ex) test.txt -> test.png

 2. with two arguments:

        python draw.py [path/file_name] [path_to_destination/file_name]

   will save the given (8277,) text file to [path_to_destination/file_name] with latter filename.
