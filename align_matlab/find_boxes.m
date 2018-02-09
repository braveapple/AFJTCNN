clear;
%list of images
imglist=importdata('./casialist.txt');
%imglist.textdata{1}='/data/1.jpg';
%minimum size of face
minsize=20;

%path of toolbox
caffe_path='/home/hins/caffe/matlab';
pdollar_toolbox_path='/data/lhx/toolbox-master/';
caffe_model_path='./model';
addpath(genpath(caffe_path));
addpath(genpath(pdollar_toolbox_path));

%use cpu
%caffe.set_mode_cpu();
gpu_id=2;
caffe.set_mode_gpu();	
caffe.set_device(gpu_id);

%three steps's threshold
threshold=[0.6 0.7 0.7]

%scale factor
factor=0.709;

%load caffe models
prototxt_dir =strcat(caffe_model_path,'/det1.prototxt');
model_dir = strcat(caffe_model_path,'/det1.caffemodel');
PNet=caffe.Net(prototxt_dir,model_dir,'test');
prototxt_dir = strcat(caffe_model_path,'/det2.prototxt');
model_dir = strcat(caffe_model_path,'/det2.caffemodel');
RNet=caffe.Net(prototxt_dir,model_dir,'test');	
prototxt_dir = strcat(caffe_model_path,'/det3.prototxt');
model_dir = strcat(caffe_model_path,'/det3.caffemodel');
ONet=caffe.Net(prototxt_dir,model_dir,'test');
faces=cell(0);	
tic
for i=1:length(imglist)
    if mod(i,100)==0
    i
    toc
    end
	img=imread(strcat('/data/CASIA-WebFace/',imglist{i}));
    if size(img,3)==1
        img2=zeros(size(img,1),size(img,2),3);
        img2(:,:,1)=img;
        img2(:,:,2)=img;
        img2(:,:,3)=img;
        img=img2;
    end
	%we recommend you to set minsize as x * short side
	%minl=min([size(img,1) size(img,2)]);
	%minsize=fix(minl*0.1)
    
    [boudingboxes points]=detect_face(img,minsize,PNet,RNet,ONet,threshold,false,factor);
 
	%show detection result
    if size(boudingboxes,1)==0
        faces{i,1}=0;
        faces{i,2}=0;
    else
        j=1;
	numbox=size(boudingboxes,1);
    [ro co ch]=size(img);
    for jj=1:numbox
        if  boudingboxes(jj,1)-ro/2<0 &&boudingboxes(jj,2)-co/2<0 &&boudingboxes(jj,3)-ro/2>0 &&boudingboxes(jj,4)-co/2>0
            j=jj;
        end
    end
    
    faces{i,1}={boudingboxes(j,:)};
    point=points(:,j);
	faces{i,2}={point'};
%  	imshow(img)
% 	hold on; 
% 	for j=1:numbox
% 		plot(points(1:5,j),points(6:10,j),'g.','MarkerSize',10);
% 		r=rectangle('Position',[boudingboxes(j,1:2) boudingboxes(j,3:4)-boudingboxes(j,1:2)],'Edgecolor','g','LineWidth',3);
%     end
%     hold off; 
% 	pause
    end
end
save casiaresult faces