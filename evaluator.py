
def evaluate_model(model, test_input, tar):

    prediction = model(test_input, training=True)

    prediction_map = prediction[0]
    tar_map = tar[0]
    test_input_map = test_input[0]

    prediction_gray = rgb2gray(prediction_map)
    tar_gray = rgb2gray(tar_map)

    error = np.abs(prediction_gray[:,:]-tar_gray[:,:])

    #display_list = [test_input[0], tar[0], prediction[0], error]
    #title = ['Input Image', 'Ground Truth', 'Predicted Image', "Error Image"]
    
    #### mean squared error ####
    loss1 = sklearn.metrics.mean_squared_error(tar_gray, prediction_gray)
    loss1 = round(loss1, 3)

    ### structural similarity index #####
    loss2 = ssim(tar_gray, prediction_gray)
    loss2 = round(loss2, 3)
    
    ##### peak signal to noise ratio (PSNR) ########
    
    loss3 = 20*math.log(4095/(loss1+1), 10)
    loss3 = round(loss3, 3)

    ##### mean absolute error #######
    loss4 = sklearn.metrics.mean_absolute_error(tar_gray, prediction_gray)
    loss4 = round(loss4, 3)
    
    return float(loss1), float(loss2), float(loss3), float(loss4)

