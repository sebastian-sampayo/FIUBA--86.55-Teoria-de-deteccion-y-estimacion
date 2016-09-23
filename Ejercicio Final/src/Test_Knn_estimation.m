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


% ----------------- c) Kn vecinos más cercanos ------------------
% Primero creo un espacio donde graficar las ventanas de Parzen.
% Para esto, utilizo la mayor cantidad de muestras posibles que me permite
% la memoria del programa.
n = 1e4;

k = 500;
[x1, p1] = knn_estimate(X1, k, n);
[x2, p2] = knn_estimate(X2, k, n);

figure(1)
hold all;
plot(x1,p1)
plot(x2,p2)
plot(x1, ones(length(x1),1)/10, 'k--');
plot(x2, p_gaussian(x2,mu2,sigma2), 'k--');
grid on