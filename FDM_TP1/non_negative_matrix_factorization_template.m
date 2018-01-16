%% Factorization-Based Data Modeling --  TP 1: non-negative matrix factorization - Umut Simsekli

clear 
close all
clc

%% Data generation

% the size of the matrices will be X: I x J, W: I x K, H: K x J
I = 10;
J = 20;
K = 3;

% initialize it from random numbers ("rand" only generates random numbers between 0 and 1)
Wtrue = 2 * rand(I,K); 
Htrue = 2 * rand(K,J);

% optional: make the true factor matrices more sparse (we decide on how
% we generate the data)

randomMask1 = (rand(I,K)<0.5); %try to understand what this line is doing
randomMask2 = (rand(K,J)<0.5); %you can play with the number 0.5 and see what happens

Wtrue = Wtrue .* randomMask1;
Htrue = Htrue .* randomMask2;

% generate the data (multiply the matrices and add some more NONNEGATIVE noise)
dataNoise = 1;
X = Wtrue*Htrue + dataNoise * rand(I,J);

% visualize the data with the true factor matrices

figure, %open a new figure -- google the functions that you don't know or get help from your teaching assistant

subplot(3,3,[2 3]);
imagesc(Htrue); axis xy;
colorbar;
title('True H');

subplot(3,3,[4 7]);
imagesc(Wtrue); axis xy;
colorbar;
title('True W');

subplot(3,3,[5 6 8 9]);
imagesc(X); axis xy;
colorbar;
title('X');



%% Algorithm: Multiplicative Update Rules

%Initialize the factor matrices -- you can choose a better way if you have
Wmur = 2 * rand(I,K); %make sure they are initialized non-negative
Hmur = 2 * rand(K,J);

MaxIterMur = 100;

%record the objective function values
obj_mur = zeros(MaxIterMur,1);
O=ones(I,J);
figure, 
eps=0.01;
%eps in order to avoid divide by 0
for i = 1:MaxIterMur
    
    %Update W
    %todo
    Xhat=Wmur*Hmur+eps;
    tmp=((X./Xhat)*transpose(Hmur))./(O*transpose(Hmur));
    Wmur=times(Wmur,tmp);
    %Update H
    %todo
    Xhat=Wmur*Hmur+eps;
    tmp=(transpose(Wmur)*(X./Xhat))./(transpose(Wmur)*O);
    Hmur=times(Hmur,tmp);
    
    Xhat = Wmur * Hmur;
    Xhat = Xhat + eps; %for numerical stability
    tmp=X.*log(X./Xhat)-X+Xhat;
    obj_mur(i) = sum(tmp(:));%todo
    
    
    % visualize the iterations
    subplot(3,3,[2 3]);
    imagesc(Hmur); axis xy;
    colorbar;
    title('H MUR');

    subplot(3,3,[4 7]);
    imagesc(Wmur); axis xy;
    colorbar;
    title('W MUR');

    subplot(3,3,[5 6 8 9]);
    imagesc(Xhat); axis xy;
    colorbar;
    title('Xhat');
    
    drawnow; %you need to use this function if you want to plot in a loop
    
    %pause(1); 
    
    disp(i);
    
end

figure, 
plot(obj_mur);
xlabel('Iterations');
ylabel('Objective Value');
title('MUR');
