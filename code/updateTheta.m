function theta = updateTheta(X,L_s,d)
theta= diag(zeros(d, 1));
M = X'*L_s*X;
sum_diagM=0;
for i=1:d
   sum_diagM=sum_diagM+1/(M(i,i)); 
end
for i=1:d
    theta(i,i)=1/(sum_diagM*M(i,i));
end
end