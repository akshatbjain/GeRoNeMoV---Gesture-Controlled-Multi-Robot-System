clc;
clear all;
close all;

load('dataset.mat');

num = load('Comm Files/num.txt');

elems = length(y);

flag = fopen('Comm Files/flag.txt', 'wt');
fprintf(flag, '%d', 0);
fclose(flag);


i = 1;
while(i <= num)
    pause(0.5);
    f = load('Comm Files/flag.txt');
    if(f == 1)
        disp('Loaded... Go ahead...');
        img = imread('Comm Files/training_eg.jpg');
        y(i+elems) = load('Comm Files/output.txt');
        flag = fopen('Comm Files/flag.txt', 'wt');
        fprintf(flag, '%d', 0);
        fclose(flag);
        X_new(:,:,i+elems) = img;
        i = i + 1;      
    end
end

flag = fopen('Comm Files/flag.txt', 'wt');
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