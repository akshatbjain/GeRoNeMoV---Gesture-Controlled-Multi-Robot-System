clc;

load('dataset.mat');

num = load('num.txt');

elems = length(y);


i = 1;
while(i <= num)
    f = load('flag.txt');
    if(f == 1)
        img = imread('MatOCV Comm\training_eg.jpg');
        y(i+elems) = load('MatOCV Comm\output.txt');
        flag = fopen('MatOCV Comm\flag.txt', 'wt');
        fprintf(flag, '%d', 0);
        fclose(flag);
        X_new(:,:,i+elems) = img;
        i = i + 1;      
    end
end

flag = fopen('MatOCV Comm\flag.txt', 'wt');
fprintf(flag, '%d', 0);
fclose(flag);

fprintf('Data Acquired. Please Press a Key to verify data...\n');
pause;
for i = elems+1:elems+num
    imshow(imresize(X_new(:,:,i), [240, 320], 'nearest'));
    display(y(i));
    fprintf('\n');
    pause;
end

close all;

save dataset X_new y