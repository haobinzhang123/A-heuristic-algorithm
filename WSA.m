function [Group_Best_Score,Group_Best_Pos,WAR_curve]=WSA0(N,T,lb,ub,dim,fobj)
ub = ub.*ones(dim,1);
lb = lb.*ones(dim,1);
W1n=0.4*N;
W2n=0.2*N;
W3n=0.4*N;
%% initialization
WSA_curve=zeros(1,T);
random_nums = rand(dim,N);
uniform_nums = cumsum(random_nums,2) ./ max(cumsum(random_nums,2),[],2);
[m, t] = size(uniform_nums);
for i = 1:m
    uniform_nums(i, :) = uniform_nums(i, randperm(t));
end
W=lb+uniform_nums.*(ub-lb);

for i=1:N
    Fitness_W(i)=fobj((W(:,i))');
end
Fitness_W1= Fitness_W(1:W1n);
Fitness_W2= Fitness_W(W1n+1:W1n+W2n);
Fitness_W3= Fitness_W(W1n+W2n+1:N);
[A,index]=sort(Fitness_W);
Best_Pos=W(:,index(1));
Best_Score=A(1);
W1_Pos=W(:,index(1));
W1_Score=A(1);
W2_Pos=W(:,index(1));
W2_Score=A(1);
W3_Pos=W(:,index(1));
W3_Score=A(1);
Group_Best_Pos=Best_Pos;
Group_Best_Score=Best_Score;
W1=W(:,index(1:W1n));
W2=W(:,index(W1n+1:W2n+W1n));
W3=W(:,index(W2n+W1n+1:N));
F=[Best_Score];
WB=[Best_Pos];
t=0;
L=0;
%% Start Iteration
while t<T
    Wmax=max(W,[],2);
    Wmin=min(W,[],2);
    for j=1:N
        newW(:,j)=Wmin+rand(dim,1).*(Wmax-Wmin);
    end
    for i=1:N
        ub_flag=newW(:,i)>=ub;
        lb_flag=newW(:,i)<=lb;
        newW(:,i)=newW(:,i).*(~(ub_flag+lb_flag))+(lb+rand(dim,1).*(ub-lb)).*ub_flag+(lb+rand(dim,1).*(ub-lb)).*lb_flag;
    end
    Fm=mean(Fitness_W);
    for i=1:N
        f=fobj(newW(:,i)');
        if f<=Fm
            Fitness_W(i)=f;
            W(:,i)=newW(:,i);
        end
    end
    [F_x1,index]=sort(Fitness_W);
    if F_x1(1)< Best_Score
        Best_Pos=W(:,index(1));
        Best_Score=F_x1(1);
    end
    if L>1||t>0.1*T
        %%  emit electromagnetic waves
        theta=-(5*t/T-2)./sqrt(25+25*(5*t/T-2).^2)+0.8;
        WPos_copy = repmat(Best_Pos, 1, size(W1,2));
        distances = abs(W1 - WPos_copy);
        [~, index] = sort(distances,2);
        new_W1 = zeros(size(W1));
        random_values1 = randn(size(W1, 1), size(W1, 2));
        random_values = sort(random_values1, 2);
        for j = 1:size(W1, 2)
            for i=1:size(W1, 1)
                new_W1(i,j) = Best_Pos(i) + (W1(i,j) - Best_Pos(i)) .* (1 + random_values(i,index(i,j)))./theta;
            end
        end
        for i=1:W1n
            ub_flag=new_W1(:,i)>ub;
            lb_flag=new_W1(:,i)<lb;
            new_W1(:,i)=new_W1(:,i).*(~(ub_flag+lb_flag))+(lb+rand(dim,1).*(ub-lb)).*ub_flag+(lb+rand(dim,1).*(ub-lb)).*lb_flag;
        end
        for i=1:W1n
            f=fobj((new_W1(:,i))');
            if f<max(Fitness_W1)
                Fitness_W1(i)=f;
                W1(:,i)=new_W1(:,i);
            end
            if f<W1_Score
                W1_Score=f;
                W1_Pos=new_W1(:,i);
            end
        end
        if W1_Score<Best_Score
            Best_Score=W1_Score;
            Best_Pos=W1_Pos;
        end
        for i=1:W1n
            A =  W1(:,i);
            indices = randperm(dim, floor(dim*rand+1));
            C = zeros(dim, 1);
            C(indices) = A(indices);
            C(setdiff(1:dim, indices)) = Best_Pos(setdiff(1:dim, indices));
            new_W1(:,i)=C;
        end
        for i=1:W1n
            f=fobj((new_W1(:,i))');
            if f<Fitness_W1(i)
                Fitness_W1(i)=f;
                W1(:,i)=new_W1(:,i);
            end
            if f<W1_Score
                W1_Score=f;
                W1_Pos=new_W1(:,i);
            end
        end
        if W1_Score<Best_Score
            Best_Score=W1_Score;
            Best_Pos=W1_Pos;
        end
        if W1_Score<Best_Score
            Best_Score=W1_Score;
            Best_Pos=W1_Pos;
        end
        [~,indexc]=sort(Fitness_W1);
        W1=W1(:,indexc);
        half_lengthc = floor(size(W1, 2)/2);
        %% reflected electromagnetic waves
        [~,index1]=sort(Fitness_W2);
        W2= W2(:, index1);
        for i=1:W2n
            t1=0.75+exp(-i/W2n) ;
            W2(:,i)=-t1*randn*cos((pi*i)/W2n)*(Best_Pos-W2(:,i))+0.8*W2(:,i);
            ub_flag=W2(:,i)>ub;
            lb_flag=W2(:,i)<lb;
            W2(:,i)=(W2(:,i).*(~(ub_flag+lb_flag)))+ub.*ub_flag+lb.*lb_flag;
        end
        for i=1:W2n
            Fitness_W2(i)=fobj((W2(:,i))');
            if Fitness_W2(i)<W2_Score
                W2_Score=Fitness_W2(i);
                W2_Pos=W2(:,i);
            end
        end
        if W2_Score<Best_Score
            Best_Score=W2_Score;
            Best_Pos=W2_Pos;
        end
        [~,indexb]=sort(Fitness_W2);
        W2=W2(:,indexb);
        half_lengthb = floor(size(W2, 2)/2);
        
        %%  receive electromagnetic waves
        for i=1:W3n
            Fitness_W3(i)=fobj((W3(:,i))');
        end
        lamda=((2*t)/T-0.7)/(0.78+abs((2*t)/T-0.7))+1;
        for i=1:W3n
            if rand>0.05
                u1=0.6+(1.2-0.5)*sin((pi*t)/(2*T));
                new_W3(:,i)=W3(:,i)+u1*rand*(Best_Pos-W3(:,i))+randn*cos((pi*i)/W3n)*(W3_Pos-W3(:,i));
            else
                kernel= 1/size(WB,2)*ones(1, size(WB,2));
                Px = conv2(WB, kernel, 'valid');
                new_W3(:,i)=W3(:,i)+lamda*rand*(Best_Pos-W3(:,i))+0.5*rand*(1-lamda)*(Px-W3(:,i));
            end
            ub_flag=new_W3(:,i)>ub;
            lb_flag=new_W3(:,i)<lb;
            new_W3(:,i)= new_W3(:,i).*(~(ub_flag+lb_flag))+(lb+rand(dim,1).*(ub-lb)).*ub_flag+(lb+rand(dim,1).*(ub-lb)).*lb_flag;
        end
        for i=1:W3n
            f=fobj((new_W3(:,i))');
            if f<Fitness_W3(i)
                Fitness_W3(i)=f;
                W3(:,i)=new_W3(:,i);
            end
            if f<W3_Score
                W3_Score=f;
                W3_Pos=new_W3(:,i);
            end
        end
        if W3_Score<Best_Score
            Best_Score=W3_Score;
            Best_Pos=W3_Pos;
        end
        [~,indexd]=sort(Fitness_W3);
        W3=W3(:,indexd);
        half_lengthd = floor(size(W3, 2)/2);
        W2(:, half_lengthb+1:end) = W3(:, 1:half_lengthd/2);
        W1(:, half_lengthc+1:end) = W3(:, 1:half_lengthd);
        W=[W2,W1,W3];
        
        %% Fitted gradient descent method
        alpha=0.3*ones(1,N);
        G1=[];
        delta = 0.000001;
        for j=1:N
            g1 = zeros(dim,1);
            for i = 1:dim
                W_plus=W(:,j);
                W_plus(i) = W(i,j) + delta;
                W_minus=W(:,j);
                W_minus(i) = W(i,j) - delta;
                g1(i) = (fobj(W_plus') - fobj(W_minus')) / (2*delta);
            end
            G1=[G1,g1];
        end
        
        for j=1:N
            nW(:,j) = W(:,j) - alpha(j)* G1(:,j);
            f=fobj(nW(:,j)');
            if f<Fitness_W(j)
                alpha(j)=alpha(j)*1.6;
            else
                alpha(j)=alpha(j)/1.6;
            end
        end
        
        for j=1:N
            newW(:,j) = W(:,j) - alpha(j)* G1(:,j);
        end
        
        for i=1:N
            f=fobj((newW(:,i))');
            if f<Fitness_W(i)
                Fitness_W(i)=f;
                W(:,i)=newW(:,i);
            end
            if f<Best_Score
                Best_Score=f;
                Best_Pos=newW(:,i);
            end
        end
        
    end
    Group_Best_Score=Best_Score;
    Group_Best_Pos=Best_Pos;
    disp( ['Best_Score',num2str(Group_Best_Score)])
    F=[F,Group_Best_Score];
    WB=[WB,Best_Pos];
    if t>1
        if F(t)==F(t-1)
            L=L+1;
        end
    end
    t=t+1; 
    WAR_curve(t)=Group_Best_Score;
end
end

