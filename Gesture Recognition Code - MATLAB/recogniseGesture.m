function [ gest ] = recogniseGesture(Theta1, Theta2)

    pause(0.2);
    f = load('MatOCV Comm/flag.txt');
    if(f == 1)
        img = imread('MatOCV Comm/Gesture Input.jpg');
        X_in = double(img(:)');
        gest = predict(Theta1, Theta2, X_in);
        shape_output_file = fopen('MatOCV Comm/shape_prediction.txt', 'wt');
        fprintf(shape_output_file, '%d', gest);
        fclose(shape_output_file);
        flag = fopen('MatOCV Comm/flag.txt', 'wt');
        fprintf(flag, '%d', 0);
        fclose(flag);
    end
end

