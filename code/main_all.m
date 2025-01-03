clear ;close all;clc;
 for i=1:1
    view_num=3;
    % get data
    train_path1 = ['../data/Train/temp/all/temp',num2str(i,'%02d'),'.mat'];
    load(train_path1);
    X1_src=X;
    [n_src,fea_num(1)]=size(X1_src);
    
    train_path2 = ['../data/Train/stat/all/stat',num2str(i,'%02d'),'.mat'];
    load(train_path2);
    X2_src=X;
    [n_src,fea_num(2)]=size(X2_src);
    
    train_path3 = ['../data/Train/spect/PSD/PSD',num2str(i,'%02d'),'.mat'];
    load(train_path3);
    X3_src=X;
    [n_src,fea_num(3)]=size(X3_src);
    
    X_src_fea_num=max(fea_num);
    X_src=zeros(n_src,X_src_fea_num,view_num);
    X_src(:,1:fea_num(1),1)=X1_src;
    X_src(:,1:fea_num(2),2)=X2_src;
    X_src(:,1:fea_num(3),3)=X3_src;
    
    
    
    train_Y_path = ['../data/train_Y/train_Y_s',num2str(i,'%02d'),'.mat'];
    load(train_Y_path);
    [~,Y_src] = max(Y,[],2);
    clear Y ;
    
    
    test_path1 = ['../data/test/temp/all/temp',num2str(i,'%02d'),'.mat'];
    load(test_path1);
    X1_tar=X;
    
    test_path2 = ['../data/test/stat/all/stat',num2str(i,'%02d'),'.mat'];
    load(test_path2);
    X2_tar=X;
    
    test_path3 =  ['../data/test/spect/PSD/PSD',num2str(i,'%02d'),'.mat'];
    load(test_path3);
    X3_tar=X;
    
    
    [n_tar,~]=size(X3_tar);
    
    X_tar=zeros(n_tar,X_src_fea_num,view_num);
    X_tar(:,1:fea_num(1),1)=X1_tar;
    X_tar(:,1:fea_num(2),2)=X2_tar;
    X_tar(:,1:fea_num(3),3)=X3_tar;
    
    
    train_Y_path = ['../data/test_Y/test_Y_s',num2str(i,'%02d'),'.mat'];
    load(train_Y_path);
    [~,Y_tar] = max(Y,[],2);
    % process data
    [X,X_l,Y_l,X_u,Y_u,Y_index] = process_1(X_src,Y_src,X_tar,Y_tar,fea_num);
    % main program
    [Max_acc,BestBeta,BestDelta,BestLambda,BestAlpha,BestTheta,BestS,BestF_U,Y_predict,BestIter,f_v] = MVGSF(X,X_l,Y_l,X_u,Y_u,Y_index,fea_num);
    
    % save result
    save_path = ['../result/all_',num2str(i,'%02d')];
    save(save_path,'Max_acc','BestBeta','BestDelta','BestLambda','BestAlpha','BestTheta','BestS','BestF_U','Y_predict','Y_u','BestIter','f_v');
    
    
 end