function plotData(X, y)
%PLOTDATA Plots the data points X and y into a new figure 
%   PLOTDATA(x,y) plots the data points with + for the positive examples
%   and o for the negative examples. X is assumed to be a Mx2 matrix.

% Create New Figure
figure; hold on;

% % Initial implementation
% X0 = [];
% X1 = [];
% 
% for i=1:size(y,1)
%     if y(i) == 1
%         X1 = [X1 ; X(i,:)];
%     elseif y(i) == 0
%         X0 = [X0 ; X(i,:)];
%     end
% end
% 
% plot(X0(:,1),X0(:,2),'ko','MarkerFaceColor','y');
% plot(X1(:,1),X1(:,2),'k+');

% *****************************************************

% Alternate implementation

% Find Indices of Positive and Negative Examples
pos = find(y==1); neg = find(y == 0);

% Plot Examples
plot(X(pos, 1), X(pos, 2), 'k+','LineWidth', 2, ...
'MarkerSize', 7);
plot(X(neg, 1), X(neg, 2), 'ko', 'MarkerFaceColor', 'y', ...
'MarkerSize', 7);

hold off;

end
