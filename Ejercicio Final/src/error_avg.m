% ----------------------------------------------------------------------
% Teoría de Detección y Estimación - FIUBA
%
% Trabajo Final
%
% Fecha de entrega: 14 de Julio de 2014
%
% Alumno: Sebastián Sampayo
%
% Test de error promedio
% ----------------------------------------------------------------------

close all;
clear all;


k_knn = [1, 10, 50, 100];
N_knn = length(k_knn);
error_total_parzen = 0;
error_total_knn = zeros(N_knn, 1);
k_knnr = [1, 11, 51];
N_knnr = length(k_knnr);
error_total_knnr = zeros(N_knnr, 1);
M = 1000;

N = 1e3;
mu2 = 2;
sigma2 = 1;
P1 = 0.4;
P2 = 0.6;
n1 = 1e3;
h1_optimum = 0.1;
h2_optimum = 0.6;
n2 = 1e3;
N_new = 1e2;

for i=1:M

% -------------- a) Generación de la muestra ---------------

% X1 ~ Uniforme(0,10)
X1 = rand(N, 1) * 10;
% X2 ~ Gaussiana(2,1)
X2 = normrnd(mu2, sigma2, N, 1);
% ----------------------------------------------------------


% ----------------- b) Ventanas de Parzen ------------------
[x1_parzen, p1_parzen] = parzen_estimate(X1, h1_optimum, n1);
[x2_parzen, p2_parzen] = parzen_estimate(X2, h2_optimum, n1);
% ----------------------------------------------------------


% ----------------- c) Kn vecinos más cercanos ------------------
p1_knn = zeros(n2, N_knn);
p2_knn = zeros(n2, N_knn);
x1_knn = zeros(n2, N_knn);
x2_knn = zeros(n2, N_knn);

for j = 1:N_knn
    [x1_knn(:,j), p1_knn(:,j)] = knn_estimate(X1, k_knn(j), n2);
    [x2_knn(:,j), p2_knn(:,j)] = knn_estimate(X2, k_knn(j), n2);
end
% ----------------------------------------------------------



% ------------------ d) Clasificación Bayesiana ---------------
X1_new = rand(N_new, 1) * 10;
X2_new = normrnd(mu2, sigma2, N_new, 1);

% --- Caso Parzen ---
class1 = bayesian_classificate(X1_new,x1_parzen,p1_parzen,...
    x2_parzen,p2_parzen,P1,P2);
class2 = bayesian_classificate(X2_new,x1_parzen,p1_parzen,...
    x2_parzen,p2_parzen,P1,P2);

% Error debido a elegir clase 2 cuando correspondía clase 1
error_rate1 = sum(class1)/N_new;
% Error debido a elegir clase 1 cuando correspondía clase 2
error_rate2 = 1 - sum(class2)/N_new;
% Error total
% error_total_parzen = error_rate1*P1 + error_rate2*P2;
error_total_parzen = error_total_parzen + error_rate1*P1 + error_rate2*P2;
% fprintf('Error total - Parzen: %.4f\n', error_total_parzen)
% ------------------
% --- Caso Knn ---

for j=1:N_knn
    class1 = bayesian_classificate(X1_new,x1_knn(:,j),p1_knn(:,j),...
        x2_knn(:,j),p2_knn(:,j),P1,P2);
    class2 = bayesian_classificate(X2_new,x1_knn(:,j),p1_knn(:,j),...
        x2_knn(:,j),p2_knn(:,j),P1,P2);

    % Error debido a elegir clase 2 cuando correspondía clase 1
    error_rate1 = sum(class1)/N_new;
    % Error debido a elegir clase 1 cuando correspondía clase 2
    error_rate2 = 1 - sum(class2)/N_new;
    % Error total
    error_total_knn(j) = error_total_knn(j) + error_rate1*P1 + error_rate2*P2;
%     fprintf('Error total - KNN, k = %i: %.4f\n', k_knn(j), error_total_knn(j))
end
% ------------------

% ----------------------------------------------------------



% ------------------ e) Clasificación KNNR --------------------

for j=1:N_knnr
    class1 = knnr_classificate(X1_new, X1, X2, k_knnr(j));
    class2 = knnr_classificate(X2_new, X1, X2, k_knnr(j));
    
    % Error debido a elegir clase 2 cuando correspondía clase 1
    error_rate1 = sum(class1)/N_new;
    % Error debido a elegir clase 1 cuando correspondía clase 2
    error_rate2 = 1 - sum(class2)/N_new;
    % Error total
    error_total_knnr(j) = error_total_knnr(j) + error_rate1*P1 + error_rate2*P2;
%     fprintf('Error total - KNNR, k = %i: %.4f\n', k_knnr(j), error_total_knnr(j))
end
% ----------------------------------------------------------


end

error_total_parzen = error_total_parzen/M;
error_total_knn = error_total_knn/M;
error_total_knnr = error_total_knnr/M;

fprintf('\n--- Errores promedio ---\n')
fprintf('Iteraciones: %i\n', M)
fprintf('# Muestras de entrenamiento: %i\n', N)
fprintf('\nClasificación Bayesiana\n')
fprintf('Error promedio - Parzen: %.4f\n', error_total_parzen)
for j=1:N_knn
    fprintf('Error promedio - KNN, k = %i: %.4f\n', k_knn(j), error_total_knn(j))
end
fprintf('\nClasificación por regla de los K vecinos más cercanos\n');
for j=1:N_knnr
    fprintf('Error promedio - KNNR, k = %i: %.4f\n', k_knnr(j), error_total_knnr(j))
end