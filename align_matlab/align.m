imglist=importdata('/home/lhx/MTCNN_face_detection_alignment-master/code/codes/MTCNNv1/casialist.txt');
load /home/lhx/MTCNN_face_detection_alignment-master/code/codes/MTCNNv1/calfwresult.mat
imgSize = [112, 96];
coord5points = [30.2946, 65.5318, 48.0252, 33.5493, 62.7299; ...
                51.6963, 51.5014, 71.7366, 92.3655, 92.2041];
            delcount=0;
            
for i=1:length(imglist)
    %i
    %if iscell(faces{i,1})
    tic
    image=imread(strcat('/home/lhx/images/',imglist{i}));

facial5points = [faces{i,2}{1}(1:5);faces{i,2}{1}(6:10)];
facial5points=double(facial5points);

Tfm =  cp2tform(facial5points', coord5points', 'similarity');
cropImg = imtransform(image, Tfm, 'XData', [1 imgSize(2)],...
                                  'YData', [1 imgSize(1)], 'Size', imgSize);
                              
%figure(1);
%imshow(cropImg);

% transform image, obtaining the original face and the horizontally flipped one
if size(cropImg, 3) < 3
   cropImg(:,:,2) = cropImg(:,:,1);
   cropImg(:,:,3) = cropImg(:,:,1);
end
% cropImg = single(cropImg);
% cropImg = (cropImg - 127.5)/128;
% cropImg = permute(cropImg, [2,1,3]);
% cropImg = cropImg(:,:,[3,2,1]);

% cropImg_(:,:,1) = flipud(cropImg(:,:,1));
% cropImg_(:,:,2) = flipud(cropImg(:,:,2));
% cropImg_(:,:,3) = flipud(cropImg(:,:,3));
if mod(i,1000)==0
    i
toc
end
imwrite(cropImg,strcat('/home/lhx/cropcasia/',imglist{i}));
    %else
    %    delete(strcat('/home/lhx/cropcasia/',imglist{i}));
     %   delcount=delcount+1
   % end
    
        
%pause
end