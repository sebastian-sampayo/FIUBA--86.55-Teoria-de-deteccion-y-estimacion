% ----------------------------------------------------------------------
% Teoría de Detección y Estimación - FIUBA
%
% Trabajo Final
%
% Fecha de entrega: 14 de Julio de 2014
%
% Alumno: Sebastián Sampayo
%
% Archivo principal
% ----------------------------------------------------------------------

close all;
clear all;

% -------------- a) Generación de la muestra ---------------
N = 1e4;
% X1 ~ Uniforme(0,10)
X1 = rand(N, 1) * 10;
% X2 ~ Gaussiana(2,1)
mu2 = 2;
sigma2 = 1;
X2 = normrnd(mu2, sigma2, N, 1);
% Probabilidades a priori:
P1 = 0.4;
P2 = 0.6;
% ----------------------------------------------------------


% ----------------- b) Ventanas de Parzen ------------------
% Primero creo un espacio donde graficar las ventanas de Parzen.
% Para esto, utilizo la mayor cantidad de muestras posibles que me permite
% la memoria del programa.
n = 1e6;

% ------ X1 ------
% Para elegir 'h' adopto el criterio de dividir los datos de entrenamiento
% en 2 y hacer "maximum likelihood". En una realizo la estimación 
% con un cierto 'h' y en la otra
% calculo el likelihood para dicho 'h'. Luego itero con distintos valores
% de 'h'. Finalmente, busco el 'h' que maximiza el likelihood.
X1_1 = X1(round(N/10):end); % Utilizo una mayor proporción de muestras para estimar
X1_2 = X1(1:round(N/10)); % que para validar.

% -- Selección de valores de h para calcular el likelihood.
N_h = 4;
% Se toma un valor inicial y final de h arbitrario
min_h = (max(X1)-min(X1))/sqrt(N)/2;
max_h = (max(X1)-min(X1))/12;
h = linspace(min_h, max_h, N_h);
likelihood = zeros(N_h, 1);
s = cell(1,N_h+1); % Leyendas del gráfico

figure(1)
hold all
for i=1:N_h
    [x1, p1] = parzen_estimate(X1_1, h(i), n);
    plot(x1,p1)
    % Muestrear los datos de validación
    min1 = min(x1);
    max1 = max(x1);
    n1 = length(x1);
    Ts1 = (max1-min1)/(n1-1);
    x_index = round((X1_2-min1)/Ts1) + 1;
    x_index(x_index>n1)=n1;
    x_index(x_index<1)=1;
    % Calcular el likelihood
    likelihood(i) = sum(log(p1(x_index)));
    s(i) = cellstr(sprintf('h = %.4f',h(i))); % Leyenda del gráfico
end
p1_true = ones(length(x1),1)/10;
plot(x1, p1_true, 'k--');
s(N_h+1) = cellstr('p1(x)_{true}');
grid on
xlabel('x');
ylabel('p1(x)');
title('Estimación por Ventanas de Parzen para X1');
legend(s);
% print('-dpng', 'est_parzen_1.png');

% Maximizar el Likelihood
h1_optimum = h(likelihood==max(likelihood));

figure(2)
hold all
plot(h, likelihood, '-*')
plot(h1_optimum, max(likelihood), 'or')
grid on
xlabel('h');
ylabel('Likelihood (log)');
title('Estimación de longitud "h" por Maximum Likelihood para X1');
% print('-dpng', 'likelihood_1.png');

% Estimar nuevamente con el h óptimo según ML.
[x1_parzen, p1_parzen] = parzen_estimate(X1_1, h1_optimum, n);
p1_true = ones(length(x1_parzen),1)/10;
% ---------------


% ----- X2 ------
% Para elegir 'h' adopto el criterio de dividir los datos de entrenamiento
% en 2 y hacer "maximum likelihood". En una realizo la estimación 
% con un cierto 'h' y en la otra
% calculo el likelihood para dicho 'h'. Luego itero con distintos valores
% de 'h'. Finalmente, busco el 'h' que maximiza el likelihood.
X2_1 = X2(round(N/10):end); % Utilizo una mayor proporción de muestras para estimar
X2_2 = X2(1:round(N/10)); % que para validar.

% -- Selección de valores de h para calcular el likelihood.
N_h = 4;
min_h = (max(X2)-min(X2))/sqrt(N)/2;
max_h = (max(X2)-min(X2))/6;
h = linspace(min_h, max_h, N_h);
likelihood = zeros(N_h, 1);
s = cell(1,N_h+1);

figure(3)
hold all
for i=1:N_h
    [x2, p2] = parzen_estimate(X2_1, h(i), n);
    plot(x2,p2)
    % Muestrear los datos de validación
    min2 = min(x2);
    max2 = max(x2);
    n2 = length(x2);
    Ts = (max2-min2)/(n2-1);
    x_index = round((X2_2-min2)/Ts) + 1;
    x_index(x_index>n2)=n2;
    x_index(x_index<1)=1;
    % Calcular el likelihood
    likelihood(i) = sum(log(p2(x_index)));
    s(i) = cellstr(sprintf('h = %.4f',h(i)));
end
p2_true = normpdf(x2,mu2,sigma2);
plot(x2, p2_true, 'k--');
s(N_h+1) = cellstr('p2(x)_{true}');
grid on
xlabel('x');
ylabel('p2(x)');
title('Estimación por Ventanas de Parzen para X2');
legend(s);
% print('-dpng', 'est_parzen_2.png');

% Maximizar el Likelihood
h2_optimum = h(likelihood==max(likelihood));

figure(4)
hold all
plot(h, likelihood, '-*')
plot(h2_optimum, max(likelihood), 'or')
grid on
xlabel('h');
ylabel('Likelihood (log)');
title('Estimación de longitud "h" por Maximum Likelihood para X2');
% print('-dpng', 'likelihood_2.png');

% Estimar nuevamente con el h óptimo según ML.
[x2_parzen, p2_parzen] = parzen_estimate(X2_1, h2_optimum, n);
p2_true = normpdf(x2_parzen,mu2,sigma2);
% ---------------

% ----------------------------------------------------------


% ----------------- c) Kn vecinos más cercanos ------------------
% Primero creo un espacio donde graficar las ventanas de Parzen.
% Para esto, utilizo la mayor cantidad de muestras posibles que me permite
% la memoria del programa.
n = 1e5;

k_knn = [1, 10, 50, 100];
N_knn = length(k_knn);
p1_knn = zeros(n, N_knn);
p2_knn = zeros(n, N_knn);
x1_knn = zeros(n, N_knn);
x2_knn = zeros(n, N_knn);

for j = 1:N_knn
    [x1_knn(:,j), p1_knn(:,j)] = knn_estimate(X1, k_knn(j), n);
    [x2_knn(:,j), p2_knn(:,j)] = knn_estimate(X2, k_knn(j), n);
end
% ----------------------------------------------------------



% ------------------ d) Clasificación Bayesiana ---------------
% La idea es comparar las funciones discriminantes y clasificar según
% aquella que de mayor. Las funciones discriminantes son las densidades
% de probabilidad estimadas pesadas por sus respectivas probabilidades
% a priori.
% Si xn es la nueva muestra, entonces la regla de decisión será:
% Elijo la clase 1 si: p1(xn)*P1 > p2(xn)*P2
% Sino, elijo la clase 2.
fprintf('\nClasificación Bayesiana\n')
% Primero generar las nuevas muestras de cada clase:
N_new = 1e2;
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
error_total_parzen = error_rate1*P1 + error_rate2*P2;
fprintf('Error total - Parzen: %.4f\n', error_total_parzen)
% ------------------
% --- Caso Knn ---
error_total_knn = zeros(N_knn, 1);
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
    error_total_knn(j) = error_rate1*P1 + error_rate2*P2;
    fprintf('Error total - KNN, k = %i: %.4f\n', k_knn(j), error_total_knn(j))
end
% ------------------

% ----------------------------------------------------------



% ------------------ e) Clasificación KNNR --------------------
fprintf('\nClasificación por regla de los K vecinos más cercanos\n');
k_knnr = [1, 11, 51];
N_knnr = length(k_knnr);
error_total_knnr = zeros(N_knnr, 1);
for j=1:N_knnr
    class1 = knnr_classificate(X1_new, X1, X2, k_knnr(j));
    class2 = knnr_classificate(X2_new, X1, X2, k_knnr(j));
    
    % Error debido a elegir clase 2 cuando correspondía clase 1
    error_rate1 = sum(class1)/N_new;
    % Error debido a elegir clase 1 cuando correspondía clase 2
    error_rate2 = 1 - sum(class2)/N_new;
    % Error total
    error_total_knnr(j) = error_rate1*P1 + error_rate2*P2;
    fprintf('Error total - KNNR, k = %i: %.4f\n', k_knnr(j), error_total_knnr(j))
end


% ----------------------------------------------------------


% ------------- Error ideal -----------------
% Error en el caso ideal de clasificación bayesiana con las 
% densidades de probabilidad reales:
error_ideal = 1/10*P1*(3.8) + ((1-normcdf(3.9,2,1)) + normcdf(0.1,2,1)-normcdf(0,2,1))*P2;
fprintf('\nError ideal: %.4f\n', error_ideal)
% -------------------------------------------


% ------------------ Gráficos ---------------
figure(5)
s = cell(1, 4);
hold all
plot(x1_parzen,p1_parzen*P1)
plot(x2_parzen,p2_parzen*P2)
plot(x1_parzen, p1_true*P1, 'k--');
plot(x2_parzen, p2_true*P2, 'k--');

s(1) = cellstr('P(w1)*p1(x|w1)_{Parzen}');
s(2) = cellstr('P(w2)*p2(x|w2)_{Parzen}');
s(3) = cellstr('P(w1)*p1(x|w1)_{true}');
s(4) = cellstr('P(w2)*p2(x|w2)_{true}');
legend(s);
title('Estimación por Ventanas de Parzen');
xlabel('x');
grid on
% print('-dpng', 'est_parzen_3.png');


figure(6)
s = cell(1, 4);
hold all
plot(x1_knn(:,2), p1_knn(:,2)*P1)
plot(x2_knn(:,2), p2_knn(:,2)*P2)
plot(x1_parzen, p1_true*P1, 'k--');
plot(x2_parzen, p2_true*P2, 'k--');

s(1) = cellstr('P(w1)*p1(x|w1)_{knn - k = 10}');
s(2) = cellstr('P(w2)*p2(x|w2)_{knn - k = 10}');
s(3) = cellstr('P(w1)*p1(x|w1)_{true}');
s(4) = cellstr('P(w2)*p2(x|w2)_{true}');
legend(s);
title('Estimación por KNN');
xlabel('x');
grid on
% print('-dpng', 'KNN10.png');


figure(7)
s = cell(1, 4);
hold all
plot(x1_knn(:,4), p1_knn(:,4)*P1)
plot(x2_knn(:,4), p2_knn(:,4)*P2)
plot(x1_parzen, p1_true*P1, 'k--');
plot(x2_parzen, p2_true*P2, 'k--');

s(1) = cellstr('P(w1)*p1(x|w1)_{knn - k = 100}');
s(2) = cellstr('P(w2)*p2(x|w2)_{knn - k = 100}');
s(3) = cellstr('P(w1)*p1(x|w1)_{true}');
s(4) = cellstr('P(w2)*p2(x|w2)_{true}');
legend(s);
title('Estimación por KNN');
xlabel('x');
grid on
% print('-dpng', 'KNN100.png');
% ----------------------------------------------------------
