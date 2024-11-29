function [X,X_l,Y_l,X_u,Y_u,Y_index] = process_1(X_src,Y_src,X_tar,Y_tar,fea_num)
[Y_sort,index] = sort(Y_src);
[~,~,view_num]=size(X_src);
for v=1:view_num
    X_sort = X_src(index,1:fea_num(v),v);
    X_src(:,1:fea_num(v),v) = X_sort;
end
Y_src = Y_sort;


[Y_sort,index] = sort(Y_tar);
[~,~,view_num]=size(X_tar);
for v=1:view_num
    X_sort = X_tar(index,1:fea_num(v),v);
    X_tar(:,1:fea_num(v),v)= X_sort;
end

Y_tar = Y_sort;
[n_l,fea_l,view_num] = size(X_src);
[n_u,fea_u,~] = size(X_tar);
X_l=zeros(n_l,fea_l,view_num);
X_u=zeros(n_u,fea_u,view_num);
for v=1:view_num
    X_l(:,:,v) = X_src(:,:,v);
    X_u(:,:,v) = X_tar(:,:,v);
    col_max = max(X_l(:,1:fea_num(v),v), [], 1); 
    col_min = min(X_l(:,1:fea_num(v),v), [], 1); 
    col_range = col_max - col_min;
    X_l(:,1:fea_num(v),v) = (X_l(:,1:fea_num(v),v) - col_min) ./ col_range;
    X_l(:,1:fea_num(v),v) =X_l(:,1:fea_num(v),v)*fea_num(v);
    mean_X_l1=repmat(mean(X_l(:,1:fea_num(v),v),1),[n_l,1]);
    mean_X_l2=repmat(mean(X_l(:,1:fea_num(v),v),1),[n_u,1]);
    X_l(:,1:fea_num(v),v)= X_l(:,1:fea_num(v),v)-mean_X_l1;
    X_u(:,1:fea_num(v),v) = (X_u(:,1:fea_num(v),v) - col_min) ./ col_range;
    X_u(:,1:fea_num(v),v) =X_u(:,1:fea_num(v),v)*fea_num(v);
    X_u(:,1:fea_num(v),v)= X_u(:,1:fea_num(v),v)-mean_X_l2;
end
Y_l = Y_src;
Y_u = Y_tar;
X = [X_l; X_u];
[n,~,~] = size(X);
Y = [Y_l;Y_u];
[~,Y_index] = sort(Y);
Y_l = LabelConvert(Y_l);
