% ----------------------------------------------------------------------
% Teoría de Detección y Estimación - FIUBA
%
% Trabajo Final - 2014
%
% Alumno: Sebastián Sampayo
% ----------------------------------------------------------------------

% --- Función estimadora por ventanas de Parzen ---
% 
% [x, p] = parzen_estimate(data, h, n) 
%
% p: función de densidad de probabilidad estimada
% x: espacio creado donde se evalúa 'p'. (p = p(x))
% data: muestras de entrenamiento
% h: longitud de la ventana de Parzen
% n: cantidad de muestras del espacio creado para 'p' (length(p)~n)
% 
% Ejemplo:
% data = rand(1e4,1);
% [x, p] = parzen_estimate(data, 0.1, 1e5);
% plot(x,p)

function [x, p] = parzen_estimate(data, h, n)
    min2 = min(data);
    max2 = max(data);
    
    % Período de meustreo
    Ts = (max2-min2)/(n-1);
    % Samplear 'h' (cantidad de muestras en una longitud 'h')
    h_sampled = round(h/Ts) + 1;
    % Crear la ventana
    window = ones(h_sampled,1);
    % Crear un vector de ceros
    x2 = zeros(n,1);
    % Samplear los datos (hacer equivalencia "dato real - índice del vector")
    data_sampled = round((data - min2) / Ts) + 1;
    data_sampled(data_sampled==0) = 1;
    % Poner deltas en el vector de ceros donde cayeron los datos
    x2(data_sampled) = 1;
    % Convolucionar la ventana con los datos y escalar.
    p = conv(x2, window)/length(data)/h;
    x = linspace(min2-h/2, max2+h/2, n+h_sampled-1)';
end