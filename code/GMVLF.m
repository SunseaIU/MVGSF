function [Max_acc,BestBeta,BestDelta,BestLambda,BestAlpha,BestTheta,BestS,BestF_U,Y_predict,BestIter,f_v] = GMVLF(X,X_l,Y_l,X_u,Y_u,Y_index,d)
% =========================================================================
% Input:
% X: n*d is the all data, which include source data and target data.
% X_l: l*d is the labeled data feature.
% Y_l: l*c is the labeled data ground truth.
% X_u: u*d is the unlabeled data feature.
% Y_u: u*one is the unlabeled data ground truth (one represents the number 1).
% d: one*view_num is the number of features in each view.
%
% Output:
% Max_acc: is the accuracy of emotion recognition by use SPGO model.
% BestBeta: is the value of parameter beta at maximum accuracy.
% BestDelta: is the value of parameter delta at maximum accuracy.
% BestLambda: is the value of parameter lambda at maximum accuracy.
% BestAlpha: is the value of view importance at maximum accuracy.
% BestTheta: is the value of feature importance at maximum accuracy.
% BestS: is the adjacency matrix of adaptive graph.
% BestF_U: is the prediction accuracy for each class.  
% Y_predict: is the prediction of unlabeled data.
% BestIter: is the value of Iter at maximum accuracy.
% f_v:is the value of the objective function.
% 
% 
% =========================================================================
[n,~,~] = size(X);
[l,c] = size(Y_l);
[u,~] = size(X_u);
[~,view_num]=size(d);
% set the parameter selection range to {2^-10, 2^-8,бн,2^10}
betanums = [-2];
deltanums = [0];
lambdanums = [-3];
% Default MaxIteration is 50;
MaxIteration = 50;
Max_acc = 0;
F_mean = ones(u,c)/c;
F_new = [Y_l;F_mean];
max_d = max(d);
S=zeros(n,n);
for beta_index = 1:length(betanums)
    for delta_index = 1:length(deltanums)
        for lambda_index = 1:length(lambdanums)
            beta = 2^betanums(beta_index);
            delta = 2^deltanums(delta_index);
            lambda = 2^lambdanums(lambda_index);
            alpha=ones(1,view_num) * (1/view_num);
            theta=zeros(max_d,max_d,view_num);
            %% Initialization
            for v=1:view_num
                theta(1:d(v),1:d(v),v)=diag(ones(d(v), 1) * (1/d(v)));
            end
            D_X=zeros(n,n,view_num);
            for v=1:view_num
                d_n=1:d(v);
                D_X(:,:,v)=getDmatrix(X(:,d_n,v)*theta(d_n,d_n,v));
            end
            distX=zeros(n,n);
            for v=1:view_num
                distX=distX+alpha(v)*D_X(:,:,v);
            end
            for i = 1:n
                temp=(-1/(2*beta))*distX(i,:);
                S(i,:) =EProjSimplex_new(temp);
            end
            S = (S+S')/2;
            D_s = diag(sum(S));
            L_s = D_s - S;
            F=F_new;
            f_v=zeros(1,MaxIteration);
            f_v1=zeros(1,MaxIteration);
            f_v2=zeros(1,MaxIteration);
            f_v3=zeros(1,MaxIteration);
            f_v4=zeros(1,MaxIteration);
            acc=zeros(1,MaxIteration);
            %% Loop Iteration
            for iter = 1:MaxIteration
                %Update F
                for i = l+1:n
                    sumf_j1 = sum(F(1:l,:).* S(i,1:l)');
                    sumf_j = (sumf_j1 - S(i,i)*F(i,:));
                    F(i,:) = EProjSimplex_new(sumf_j);
                end
                % calculate the accuracy
                F_U = F(l+1:n,:);
                [~,Max_index] = max(F_U,[],2);
                acc(iter) = length(find(Max_index==Y_u))/u;
                [f_v(iter),f_v1(iter),f_v2(iter),f_v3(iter),f_v4(iter)]=getFunctionValue(beta,lambda,delta,alpha,D_X,S,F,L_s,view_num);
                %Update alpha
                alpha=updateAlpha(D_X,S,n,view_num,delta);
                %Update S
                D_TX=zeros(n,n);
                for v=1:view_num
                    D_TX=D_TX+alpha(v)*D_X(:,:,v);
                end
                D_F = getDmatrix(F);
                D_F(l+1:n,l+1:n)=1;
                D = D_TX + lambda*D_F;
                S=zeros(n);
                for i = 1:n
                    temp=D(i,:)/(-2*beta);
                    S(i,:) = EProjSimplex_new(temp);
                end
                %Update L_s
                S = (S+S')/2;
                D_s = diag(sum(S));
                L_s = D_s - S;
                %Update Theta
                for v=1:view_num
                    d_n=1:d(v);
                    theta(d_n,d_n,v)=updateTheta(X(:,d_n,v),L_s,d(v));
                end
                for v=1:view_num
                    d_n=1:d(v);
                    D_X(:,:,v)=getDmatrix(X(:,d_n,v)*theta(d_n,d_n,v));
                end
                if(acc(iter) >=Max_acc)
                    Max_acc = acc(iter);
                    BestBeta = beta;
                    BestDelta = delta;
                    BestLambda = lambda;
                    BestF_U = F_U;
                    BestS = solveS(S,Y_index);
                    BestAlpha = alpha;
                    BestTheta = theta;
                    Y_predict = Max_index;
                    BestIter = iter;
                end
                if(mod(iter,10)==1)
                    fprintf('beta: %.4f delta:%.4f lambda: %.4f the best acc: %.4f,acc:%.4f  f_v:%.4f  \n',beta,delta,lambda,Max_acc,acc(iter),f_v(iter));
                end
            end
            fprintf('next\n');
            fprintf('beta: %.4f delta: %.4f lambda: %.4f the best acc: %.4f \n',BestBeta,BestDelta,BestLambda,Max_acc);
        end
    end
    
end
end




