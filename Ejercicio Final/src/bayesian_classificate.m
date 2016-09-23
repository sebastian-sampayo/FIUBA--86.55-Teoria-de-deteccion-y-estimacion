% ----------------------------------------------------------------------
% Teoría de Detección y Estimación - FIUBA
%
% Trabajo Final - 2014
%
% Alumno: Sebastián Sampayo
% ----------------------------------------------------------------------

% --- Función de Clasificación Bayesiana entre 2 clases ---
% 
% class = bayesian_classificate(data, x0, p0, x1, p1, P0, P1)
%
% data: muestras a clasificar (puede ser un escalar o un vector)
% x0/1: espacio creado donde se evalúa 'p'. (p = p(x))
% p0/1: función de densidad de probabilidad de la clase 0/1
% P0/1: Probabilidad a priori de clase 0/1
% class: dato clasificado, 
%       class = 0 si clasifica clase 0 (p0(data)*P0 > p1(data)*P1)
%       class = 1 si clasifica clase 1 (p0(data)*P0 < p1(data)*P1)
% 
% Ejemplo:
% x0 = linspace(0,10,1e3);
% x1 = linspace(-4, 6, 1e3);
% p0_true = ones(length(x0),1)/10;
% p1_true = normpdf(x1,2,1);
% data = normrnd(2,1);
% bayesian_classificate(data, x0, p0, x1, p1, 0.5, 0.5)

function class = bayesian_classificate(data, x0, p0, x1, p1, P0, P1)
    min0 = min(x0);
    max0 = max(x0);
    n0 = length(x0);
    Ts0 = (max0-min0)/(n0-1);
    min1 = min(x1);
    max1 = max(x1);
    n1 = length(x1);
    Ts1 = (max1-min1)/(n1-1);
    
    % Condiciones de borde, esto se hace para que en caso de que 'data'
    % sea un valor fuera de los límites de x, p(data) sea 0.
    p0 = [p0;0];
    p1 = [p1;0];

    x_index0 = round((data-min0)/Ts0) + 1;
    x_index0(x_index0>n0)=n0+1;
    x_index0(x_index0<1)=n0+1;

    x_index1 = round((data-min1)/Ts1) + 1;
    x_index1(x_index1>n1)=n1+1;
    x_index1(x_index1<1)=n1+1;
    
    class = p0(x_index0)*P0 < p1(x_index1)*P1;
end