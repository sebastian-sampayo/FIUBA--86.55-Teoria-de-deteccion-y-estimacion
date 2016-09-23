% ----------------------------------------------------------------------
% Teoría de Detección y Estimación - FIUBA
%
% Trabajo Final
%
% Fecha de entrega: 14 de Julio de 2014
%
% Alumno: Sebastián Sampayo
% ----------------------------------------------------------------------

close all;
clear all;

p_gaussian = @(x, mu, sigma) ...
    (1/(sqrt(2*pi)*sigma)*exp(-1/2*((x-mu)/sigma).^2));

% ---- a) Generación de la muestra ----
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
% -------------------------------------

% ---- b) Ventanas de Parzen ----
% Primero creo un espacio donde graficar las ventanas de Parzen.
% Para esto, utilizo la mayor cantidad de muestras posibles que me permite
% la memoria del programa.
n = 1e6;
% - X2 -
% Para elegir 'h' adopto el criterio de dividir los datos de entrenamiento
% en 2 y hacer "maximum likelihood". En una realizo la estimación 
% con un cierto 'h' y en la otra
% calculo el likelihood para dicho 'h'. Luego itero con distintos valores
% de 'h'. Finalmente, busco el 'h' que maximiza el likelihood.
X2_1 = X2(N/3:end); % Utilizo una mayor proporción de muestras para estimar
X2_2 = X2(1:N/3); % que para validar.

% Selección de valores de h para calcular el likelihood.
N_h = 30;
min_h = (max(X2)-min(X2))/sqrt(N)/2;
max_h = (max(X2)-min(X2))/6;
h = linspace(min_h, max_h, N_h);
likelihood = zeros(N_h, 1);

figure(1)
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
end
plot(x2, p_gaussian(x2,mu2,sigma2), 'k--');
grid on

figure(2)
plot(h, likelihood, '-*')
grid on
% Estimar nuevamente con el h óptimo según ML.
h2_optimum = h(likelihood==max(likelihood));
[x2, p2] = parzen_estimate(X2_1, h2_optimum, n);




% ------ Clasificación Bayesiana -------
% La idea es comparar las funciones discriminantes y clasificar según
% aquella que de mayor. Las funciones discriminantes son las densidades
% de probabilidad estimadas pesadas por sus respectivas probabilidades
% a priori.
% Si xn es la nueva muestra, entonces la regla de decisión será:
% Elijo la clase 1 si: p1(xn)*P1 > p2(xn)*P2
% Sino, elijo la clase 2.






% La idea es crear N cestos (cuyos límites son "límites_inferiores" y
% "límites_superiores") y contar cuantas muestras hay en cada cesto realiando
% comparaciones ("cuenta"). Luego hacer un gráfico de barras centradas en cada muestra
% con la altura correspondiente a la cantidad de muestras en cada cesto
% h = 1/sqrt(N);
% X2_sorted = sort(X2);
% limites_inferiores = X2_sorted - h/2;
% limites_superiores = X2_sorted + h/2;
